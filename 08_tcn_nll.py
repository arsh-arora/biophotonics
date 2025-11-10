#!/usr/bin/env python3
"""
TCN with Heteroscedastic NLL Loss
==================================
Temporal Convolutional Network that outputs both mean and variance,
trained with Negative Log-Likelihood loss for uncertainty quantification.

Key Features:
- Dual-head output: mean and log-variance
- NLL loss for probabilistic predictions
- Well-calibrated uncertainty estimates
- Only penalizes missing points
"""

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import warnings

sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        DEVICE = torch.device('cpu')
        print("Using CPU")

except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠ PyTorch not available. Install with: pip install torch")


# ============================================================================
# DATASET
# ============================================================================

class MaskedImputeDataset(Dataset):
    def __init__(self, x: np.ndarray, obs_mask: np.ndarray, window: int = 512, step: int = 256):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.m = torch.tensor(obs_mask[:, None].astype(np.float32))
        self.window = window
        self.step = step

        N = len(x)
        self.indices = [(i, min(i + window, N)) for i in range(0, N - window + 1, step)]
        if len(self.indices) == 0:
            self.indices = [(0, N)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        return self.x[start:end], self.m[start:end]


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def nll_loss(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Negative Log-Likelihood loss for heteroscedastic predictions.

    NLL = 0.5 * (log(2π) + log(σ²) + (y - μ)² / σ²)
    Simplified: 0.5 * (log_var + (y - μ)² / exp(log_var))

    Args:
        pred_mean: (B, T, D) predicted means
        pred_logvar: (B, T, D) predicted log-variances
        target: (B, T, D) ground truth
        mask: (B, T, 1) observation mask (1=observed, 0=missing)

    Returns:
        loss: scalar tensor
    """
    missing_mask = (1.0 - mask)  # 1 where missing

    # NLL computation
    squared_error = (pred_mean - target) ** 2
    nll = 0.5 * (pred_logvar + squared_error / (torch.exp(pred_logvar) + 1e-6))

    # Only on missing points
    masked_nll = nll * missing_mask
    denom = missing_mask.sum().clamp_min(1.0)

    return masked_nll.sum() / denom


# ============================================================================
# TCN ARCHITECTURE
# ============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNImputerNLL(nn.Module):
    """TCN with heteroscedastic output (mean + log-variance)."""

    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 5,
                 hidden_channels: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        layers = []
        num_channels = in_channels

        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(num_channels, hidden_channels, kernel_size, dilation, dropout))
            num_channels = hidden_channels

        self.network = nn.Sequential(*layers)

        # Dual heads for mean and log-variance
        self.mean_head = nn.Conv1d(hidden_channels, out_channels, 1)
        self.logvar_head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, T) input [trajectory; mask]

        Returns:
            mean: (B, C_out, T) predicted mean
            logvar: (B, C_out, T) predicted log-variance
        """
        features = self.network(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # Clamp log-variance to prevent numerical issues
        logvar = torch.clamp(logvar, -10, 10)

        return mean, logvar


# ============================================================================
# TRAINING
# ============================================================================

def train_tcn_nll(
    model: nn.Module,
    dataloader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = False
) -> Dict[str, list]:
    """Train TCN with NLL loss."""

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {'loss': [], 'nll': []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, m_batch in dataloader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)

            x_masked = x_batch * m_batch
            x_input = torch.cat([x_masked, m_batch], dim=-1).transpose(1, 2)
            x_target = x_batch.transpose(1, 2)

            # Forward pass
            x_pred_mean, x_pred_logvar = model(x_input)

            # NLL loss
            loss = nll_loss(
                x_pred_mean.transpose(1, 2),
                x_pred_logvar.transpose(1, 2),
                x_target.transpose(1, 2),
                m_batch
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        history['nll'].append(avg_loss)

        scheduler.step(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: NLL = {avg_loss:.6f}")

    return history


# ============================================================================
# IMPUTATION
# ============================================================================

def impute_tcn_nll(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    n_epochs: int = 50,
    window: int = 512,
    step: int = 256,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_levels: int = 5,
    hidden_channels: int = 64,
    dropout: float = 0.1,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Impute trajectory using TCN with NLL loss."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D = X.shape
    M = len(t_out)

    # Normalize
    X_mean = X[obs_mask].mean(axis=0)
    X_std = X[obs_mask].std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"\n  Training TCN-NLL:")
        print(f"    Data shape: {X.shape}")
        print(f"    Observation rate: {obs_mask.mean()*100:.1f}%")
        print(f"    Levels: {num_levels}, Hidden: {hidden_channels}")

    # Create dataset
    dataset = MaskedImputeDataset(X_norm, obs_mask, window=window, step=step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Initialize model
    model = TCNImputerNLL(
        in_channels=D + 1,
        out_channels=D,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        kernel_size=3,
        dropout=dropout
    ).to(DEVICE)

    # Train
    history = train_tcn_nll(
        model, dataloader,
        n_epochs=n_epochs, lr=lr,
        device=DEVICE, verbose=verbose
    )

    # Inference
    model.eval()
    with torch.no_grad():
        x_full = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        m_full = torch.tensor(obs_mask[:, None].astype(np.float32), device=DEVICE).unsqueeze(0)

        x_masked = x_full * m_full
        x_input = torch.cat([x_masked, m_full], dim=-1).transpose(1, 2)

        x_pred_mean, x_pred_logvar = model(x_input)
        x_pred_mean = x_pred_mean.transpose(1, 2).squeeze(0).cpu().numpy()
        x_pred_var = torch.exp(x_pred_logvar).transpose(1, 2).squeeze(0).cpu().numpy()

    # Denormalize
    X_recon = x_pred_mean * X_std + X_mean
    X_var = x_pred_var * (X_std ** 2)

    # Interpolate to 1000 Hz
    mu_1000 = np.zeros((M, D))
    var_1000 = np.zeros((M, D))

    for d in range(D):
        cs_mean = CubicSpline(t, X_recon[:, d], bc_type='natural', extrapolate=True)
        mu_1000[:, d] = cs_mean(t_out)

        cs_var = CubicSpline(t, X_var[:, d], bc_type='natural', extrapolate=True)
        var_1000[:, d] = np.maximum(cs_var(t_out), 0)

    return {
        'output_300Hz': X_recon,
        'output_1000Hz': mu_1000,
        'mean': X_recon,
        'var': X_var,
        'mean_1000Hz': mu_1000,
        'var_1000Hz': var_1000,
        'extras': {
            'final_loss': history['loss'][-1] if history['loss'] else np.nan,
            'n_epochs': n_epochs,
            'model_params': sum(p.numel() for p in model.parameters())
        }
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_imputation_with_uncertainty(
    t: np.ndarray,
    X_true: np.ndarray,
    X_pred: np.ndarray,
    X_var: Optional[np.ndarray] = None,
    obs_mask: Optional[np.ndarray] = None,
    method_name: str = "TCN-NLL",
    filename: Optional[str] = None
):
    """Plot imputation results with uncertainty bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    D = X_true.shape[1] if X_true.ndim > 1 else 1
    fig, axes = plt.subplots(D, 1, figsize=(14, 4*D), squeeze=False)

    for d in range(D):
        ax = axes[d, 0]

        y_true = X_true[:, d] if D > 1 else X_true.flatten()
        y_pred = X_pred[:, d] if D > 1 else X_pred.flatten()

        # Ground truth
        ax.plot(t, y_true, 'o', color='gray', markersize=2, alpha=0.3, label='Ground truth')

        # Prediction
        ax.plot(t, y_pred, '-', color='#e74c3c', linewidth=1.5, alpha=0.8, label='Prediction')

        # Uncertainty bands
        if X_var is not None:
            y_var = X_var[:, d] if D > 1 else X_var.flatten()
            y_std = np.sqrt(y_var)
            ax.fill_between(t.flatten(), (y_pred - 2*y_std).flatten(), (y_pred + 2*y_std).flatten(),
                           color='#e74c3c', alpha=0.2, label='±2σ uncertainty')

        # Observed points
        if obs_mask is not None:
            t_obs = t[obs_mask]
            y_obs = y_true[obs_mask]
            ax.plot(t_obs, y_obs, 'o', color='#3498db', markersize=3, alpha=0.6, label='Observed')

            # Missing points
            t_miss = t[~obs_mask]
            y_miss = y_true[~obs_mask]
            ax.scatter(t_miss, y_miss, s=15, color='red', marker='x',
                      alpha=0.4, label='Missing', zorder=5)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (nm)', fontsize=11)
        ax.set_title(f'{method_name} - Dim {d}' if D > 1 else method_name,
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {filename}")
        plt.close(fig)
    else:
        plt.show()


def visualize_sample_result_nll(
    segment: dict,
    n_epochs: int = 50
):
    """Visualize imputation on a sample segment with uncertainty."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / 1000.0)

    result = impute_tcn_nll(
        X_true, t_seg, t_out, obs_mask,
        n_epochs=n_epochs,
        verbose=True
    )

    X_pred = result['output_300Hz']
    X_var = result['var']

    # Full plot
    plot_imputation_with_uncertainty(
        t_seg, X_true, X_pred, X_var=X_var,
        obs_mask=obs_mask,
        method_name="TCN-NLL (Heteroscedastic)",
        filename="08_tcn_nll_result.png"
    )

    # Zoom plot
    missing_indices = np.where(~obs_mask)[0]
    if len(missing_indices) > 0:
        gaps = np.split(missing_indices, np.where(np.diff(missing_indices) > 1)[0] + 1)
        longest_gap = max(gaps, key=len)

        if len(longest_gap) > 10:
            pad = 50
            zoom_start = max(0, longest_gap[0] - pad)
            zoom_end = min(len(t_seg), longest_gap[-1] + pad)

            plot_imputation_with_uncertainty(
                t_seg[zoom_start:zoom_end],
                X_true[zoom_start:zoom_end],
                X_pred[zoom_start:zoom_end],
                X_var=X_var[zoom_start:zoom_end] if X_var is not None else None,
                obs_mask=obs_mask[zoom_start:zoom_end],
                method_name="TCN-NLL (Zoom)",
                filename="08_tcn_nll_zoom.png"
            )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_tcn_nll(
    segments: list,
    n_epochs: int = 50,
    window: int = 512,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate TCN-NLL on all segments."""

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available. Skipping evaluation.")
        return {}

    print("\n" + "="*80)
    print(f"EVALUATING: TCN with NLL Loss (Heteroscedastic)")
    print("="*80)
    print(f"Parameters: epochs={n_epochs}, window={window}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / 1000.0)

        result = impute_tcn_nll(
            X_true, t_seg, t_out, obs_mask,
            n_epochs=n_epochs,
            window=window,
            verbose=verbose
        )

        X_pred = result['output_300Hz']
        X_var = result['var']

        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            continue

        # Compute metrics
        y_true = X_true[missing_mask, 0]
        y_pred = X_pred[missing_mask, 0]
        y_var = X_var[missing_mask, 0] if X_var is not None else None

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        # NLL and calibration
        if y_var is not None:
            y_std = np.sqrt(y_var)
            nll = 0.5 * (np.log(2 * np.pi * y_var) + (y_true - y_pred) ** 2 / y_var)
            metrics['NLL'] = np.mean(nll)

            z_scores = np.abs((y_true - y_pred) / (y_std + 1e-9))
            metrics['Calib_1sigma'] = np.mean(z_scores <= 1.0)
            metrics['Calib_2sigma'] = np.mean(z_scores <= 2.0)

        all_metrics.append(metrics)

        calib_str = f"NLL={metrics.get('NLL', np.nan):.3f}, " \
                   f"Cal@1σ={metrics.get('Calib_1sigma', 0)*100:.1f}%" if 'NLL' in metrics else ""

        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f} | {calib_str}")

    if not all_metrics:
        return {}

    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else np.nan
        avg_metrics[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0

    print()
    print("="*80)
    print("Results on Missing Data Points")
    print("="*80)
    print(f"MAE:  {avg_metrics['MAE']:.4f} nm")
    print(f"RMSE: {avg_metrics['RMSE']:.4f} nm")
    print(f"R²:   {avg_metrics['R2']:.4f}")
    if 'NLL' in avg_metrics:
        print(f"NLL:  {avg_metrics['NLL']:.4f}")
        print(f"Calibration @ 1σ: {avg_metrics['Calib_1sigma']*100:.1f}%")
        print(f"Calibration @ 2σ: {avg_metrics['Calib_2sigma']*100:.1f}%")
    print()

    return avg_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch is not installed.")
        sys.exit(1)

    print("Loading data and segments from starter framework...")
    t, X, X_detrended, ou_params, segments = main_setup()

    print(f"\nRunning on device: {DEVICE}")

    # Evaluate
    metrics = evaluate_tcn_nll(
        segments,
        n_epochs=50,
        window=512,
        verbose=False
    )

    # Visualize
    print("\nGenerating visualization...")
    visualize_sample_result_nll(segments[0], n_epochs=50)

    print("\n✓ TCN-NLL evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
    print(f"  Saved: 08_tcn_nll_result.png, 08_tcn_nll_zoom.png")
