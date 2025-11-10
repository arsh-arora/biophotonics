#!/usr/bin/env python3
"""
TCN with Ensemble Loss
======================
Temporal Convolutional Network trained with weighted ensemble of multiple losses:
1. Reconstruction loss (MSE on missing points)
2. Smoothness loss (temporal derivatives)
3. Spectral loss (PSD consistency)
4. Physics loss (OU autocorrelation)

Multiple weight configurations are tested to find optimal balance.
"""

import sys
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal
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
# WEIGHT CONFIGURATIONS
# ============================================================================

LOSS_CONFIGS = {
    'balanced': {
        'name': 'Balanced',
        'weights': {'recon': 1.0, 'smooth': 0.5, 'spectral': 0.3, 'physics': 0.2}
    },
    'recon_heavy': {
        'name': 'Reconstruction-Heavy',
        'weights': {'recon': 2.0, 'smooth': 0.3, 'spectral': 0.1, 'physics': 0.1}
    },
    'smooth_heavy': {
        'name': 'Smoothness-Heavy',
        'weights': {'recon': 1.0, 'smooth': 1.5, 'spectral': 0.3, 'physics': 0.2}
    },
    'physics_heavy': {
        'name': 'Physics-Heavy',
        'weights': {'recon': 1.0, 'smooth': 0.3, 'spectral': 0.8, 'physics': 0.8}
    },
    'spectral_heavy': {
        'name': 'Spectral-Heavy',
        'weights': {'recon': 1.0, 'smooth': 0.2, 'spectral': 1.5, 'physics': 0.3}
    }
}


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
# ENSEMBLE LOSS
# ============================================================================

class EnsembleLoss(nn.Module):
    """
    Weighted ensemble of multiple loss terms.
    """

    def __init__(self, weights: Dict[str, float], fs: float = 300.0, tau_ms: float = 20.0):
        super().__init__()
        self.weights = weights
        self.fs = fs
        self.tau = tau_ms / 1000.0  # Convert to seconds

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MSE on missing points."""
        missing_mask = (1.0 - mask)
        squared_errors = (pred - target) ** 2
        masked_errors = squared_errors * missing_mask
        denom = missing_mask.sum().clamp_min(1.0)
        return masked_errors.sum() / denom

    def smoothness_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Penalize large temporal derivatives (encourages smooth predictions).
        Uses Huber loss for robustness.
        """
        # First-order differences
        dy_pred = pred[:, 1:] - pred[:, :-1]
        dy_target = target[:, 1:] - target[:, :-1]
        mask_diff = mask[:, 1:] * mask[:, :-1]

        # Huber loss on derivatives
        residual = dy_pred - dy_target
        delta = 1.0
        abs_res = residual.abs()
        huber = torch.where(abs_res <= delta,
                           0.5 * (abs_res ** 2),
                           delta * (abs_res - 0.5 * delta))

        masked_huber = huber * mask_diff
        denom = mask_diff.sum().clamp_min(1.0)
        return masked_huber.sum() / denom

    def spectral_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        PSD divergence: match power spectral density.
        """
        # Fill missing target values with prediction (stop gradient on target)
        target_filled = target * mask + pred.detach() * (1 - mask)
        pred_filled = pred

        B, T = pred.shape
        if T < 16:  # Too short for spectral analysis
            return torch.tensor(0.0, device=pred.device)

        window = torch.hann_window(T, device=pred.device, dtype=pred.dtype)

        # Compute FFT
        Y_true = torch.fft.rfft(target_filled * window, dim=-1)
        Y_pred = torch.fft.rfft(pred_filled * window, dim=-1)

        # Power spectral density
        Py_true = (Y_true.abs() ** 2).mean(dim=0)
        Py_pred = (Y_pred.abs() ** 2).mean(dim=0)

        # Log-MSE divergence (more stable than raw PSD difference)
        loss = ((torch.log(Py_pred + 1e-12) - torch.log(Py_true + 1e-12)) ** 2).mean()

        return loss

    def physics_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        OU physics prior: match autocovariance of increments.
        Theoretical OU has exponential decay: exp(-k*dt/τ)
        """
        # Compute increments
        dy_target = target[:, 1:] - target[:, :-1]
        dy_pred = pred[:, 1:] - pred[:, :-1]
        mask_inc = mask[:, 1:] * mask[:, :-1]

        B, T_inc = dy_pred.shape
        if T_inc < 10:
            return torch.tensor(0.0, device=pred.device)

        # Compute autocovariance at few lags
        max_lag = min(20, T_inc // 2)
        dt = 1.0 / self.fs

        loss = 0.0
        for lag in range(1, max_lag):
            if lag >= T_inc:
                break

            dy_target_1 = dy_target[:, :-lag]
            dy_target_2 = dy_target[:, lag:]
            dy_pred_1 = dy_pred[:, :-lag]
            dy_pred_2 = dy_pred[:, lag:]

            m = mask_inc[:, :-lag] * mask_inc[:, lag:]

            # Empirical autocovariance
            acov_target = ((dy_target_1 * dy_target_2) * m).sum() / (m.sum() + 1e-8)
            acov_pred = ((dy_pred_1 * dy_pred_2) * m).sum() / (m.sum() + 1e-8)

            # Theoretical OU decay
            acov_ou = torch.exp(torch.tensor(-lag * dt / self.tau, device=pred.device))

            # Penalize deviation from OU form
            loss += ((acov_pred / (acov_target.abs() + 1e-8) - acov_ou) ** 2)

        return loss / max_lag

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted ensemble loss.

        Args:
            pred: (B, T) predictions
            target: (B, T) ground truth
            mask: (B, T) observation mask

        Returns:
            loss: weighted total loss
            loss_dict: individual loss components
        """
        # Compute individual losses
        L_recon = self.reconstruction_loss(pred, target, mask)
        L_smooth = self.smoothness_loss(pred, target, mask)
        L_spectral = self.spectral_loss(pred, target, mask)
        L_physics = self.physics_loss(pred, target, mask)

        # Weighted combination
        loss = (
            self.weights['recon'] * L_recon +
            self.weights['smooth'] * L_smooth +
            self.weights['spectral'] * L_spectral +
            self.weights['physics'] * L_physics
        )

        loss_dict = {
            'total': loss.item(),
            'recon': L_recon.item(),
            'smooth': L_smooth.item(),
            'spectral': L_spectral.item(),
            'physics': L_physics.item()
        }

        return loss, loss_dict


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


class TCNImputer(nn.Module):
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
        self.output_head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        out = self.output_head(features)
        return out


# ============================================================================
# TRAINING
# ============================================================================

def train_tcn_ensemble(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: EnsembleLoss,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = False
) -> Dict[str, list]:
    """Train TCN with ensemble loss."""

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {
        'loss': [],
        'recon': [],
        'smooth': [],
        'spectral': [],
        'physics': []
    }

    for epoch in range(n_epochs):
        epoch_losses = {key: 0.0 for key in history.keys()}
        n_batches = 0

        for x_batch, m_batch in dataloader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)

            x_masked = x_batch * m_batch
            x_input = torch.cat([x_masked, m_batch], dim=-1).transpose(1, 2)

            # Forward pass
            x_pred = model(x_input)

            # Compute ensemble loss
            y_pred = x_pred.squeeze(1)
            y_true = x_batch.squeeze(-1)
            mask = m_batch.squeeze(-1)

            loss, loss_dict = loss_fn(y_pred, y_true, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            for key in history.keys():
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
                elif key == 'loss':
                    epoch_losses[key] += loss_dict['total']
            n_batches += 1

        # Average losses
        for key in history.keys():
            avg_loss = epoch_losses[key] / n_batches
            history[key].append(avg_loss)

        scheduler.step(history['loss'][-1])

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: "
                  f"Loss={history['loss'][-1]:.6f}, "
                  f"Recon={history['recon'][-1]:.6f}, "
                  f"Smooth={history['smooth'][-1]:.6f}, "
                  f"Spectral={history['spectral'][-1]:.6f}, "
                  f"Physics={history['physics'][-1]:.6f}")

    return history


# ============================================================================
# IMPUTATION
# ============================================================================

def impute_tcn_ensemble(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    config_key: str = 'balanced',
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
    """Impute trajectory using TCN with ensemble loss."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    X = np.asarray(X, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    t_out = np.asarray(t_out, dtype=np.float32)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D = X.shape
    M = len(t_out)

    # Get configuration
    config = LOSS_CONFIGS[config_key]
    weights = config['weights']

    # Normalize
    X_mean = X[obs_mask].mean(axis=0)
    X_std = X[obs_mask].std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    fs_in = 1.0 / (t[1] - t[0]) if len(t) > 1 else 300.0

    if verbose:
        print(f"\n  Training TCN-Ensemble ({config['name']}):")
        print(f"    Data shape: {X.shape}")
        print(f"    Observation rate: {obs_mask.mean()*100:.1f}%")
        print(f"    Weights: Recon={weights['recon']:.1f}, Smooth={weights['smooth']:.1f}, "
              f"Spectral={weights['spectral']:.1f}, Physics={weights['physics']:.1f}")

    # Create dataset
    dataset = MaskedImputeDataset(X_norm, obs_mask, window=window, step=step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Initialize model
    model = TCNImputer(
        in_channels=D + 1,
        out_channels=D,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        kernel_size=3,
        dropout=dropout
    ).to(DEVICE)

    # Initialize ensemble loss
    loss_fn = EnsembleLoss(weights=weights, fs=fs_in, tau_ms=20.0)

    # Train
    history = train_tcn_ensemble(
        model, dataloader, loss_fn,
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

        x_pred = model(x_input).transpose(1, 2).squeeze(0).cpu().numpy()

    # Denormalize
    X_recon = x_pred * X_std + X_mean

    # Interpolate to 1000 Hz
    mu_1000 = np.zeros((M, D))
    for d in range(D):
        cs = CubicSpline(t, X_recon[:, d], bc_type='natural', extrapolate=True)
        mu_1000[:, d] = cs(t_out)

    return {
        'output_300Hz': X_recon,
        'output_1000Hz': mu_1000,
        'mean': None,
        'var': None,
        'extras': {
            'final_loss': history['loss'][-1] if history['loss'] else np.nan,
            'final_recon': history['recon'][-1] if history['recon'] else np.nan,
            'final_smooth': history['smooth'][-1] if history['smooth'] else np.nan,
            'final_spectral': history['spectral'][-1] if history['spectral'] else np.nan,
            'final_physics': history['physics'][-1] if history['physics'] else np.nan,
            'config': config_key,
            'weights': weights,
            'n_epochs': n_epochs,
            'model_params': sum(p.numel() for p in model.parameters())
        }
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ensemble_comparison(
    t: np.ndarray,
    X_true: np.ndarray,
    X_pred: np.ndarray,
    obs_mask: Optional[np.ndarray] = None,
    config_name: str = "Balanced",
    filename: Optional[str] = None
):
    """Plot imputation results for single configuration."""
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
        ax.plot(t, y_pred, '-', color='#2ecc71', linewidth=1.5, alpha=0.8, label='Prediction')

        # Observed/Missing points
        if obs_mask is not None:
            t_obs = t[obs_mask]
            y_obs = y_true[obs_mask]
            ax.plot(t_obs, y_obs, 'o', color='#3498db', markersize=3, alpha=0.6, label='Observed')

            t_miss = t[~obs_mask]
            y_miss = y_true[~obs_mask]
            ax.scatter(t_miss, y_miss, s=15, color='red', marker='x',
                      alpha=0.4, label='Missing', zorder=5)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (nm)', fontsize=11)
        ax.set_title(f'TCN-Ensemble ({config_name}) - Dim {d}' if D > 1 else f'TCN-Ensemble ({config_name})',
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


def plot_all_configurations_comparison(
    segment: dict,
    results: Dict[str, Dict],
    filename: str = "08_tcn_ensemble_comparison.png"
):
    """Plot comparison of all configurations on same segment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    n_configs = len(results)
    fig, axes = plt.subplots(n_configs, 1, figsize=(14, 3.5*n_configs))

    if n_configs == 1:
        axes = [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, (config_key, result) in enumerate(results.items()):
        ax = axes[idx]
        config = LOSS_CONFIGS[config_key]
        X_pred = result['X_pred']

        # Ground truth
        ax.plot(t_seg, X_true[:, 0], 'o', color='gray', markersize=2, alpha=0.3, label='Ground truth')

        # Prediction
        ax.plot(t_seg, X_pred[:, 0], '-', color=colors[idx % len(colors)],
               linewidth=1.5, alpha=0.8, label='Prediction')

        # Observed/Missing
        t_obs = t_seg[obs_mask]
        y_obs = X_true[obs_mask, 0]
        ax.plot(t_obs, y_obs, 'o', color='#3498db', markersize=2, alpha=0.4, label='Observed')

        t_miss = t_seg[~obs_mask]
        y_miss = X_true[~obs_mask, 0]
        ax.scatter(t_miss, y_miss, s=10, color='red', marker='x', alpha=0.4, label='Missing', zorder=5)

        # Metrics
        mae = result['metrics']['MAE']
        rmse = result['metrics']['RMSE']
        r2 = result['metrics']['R2']

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Position (nm)', fontsize=10)
        ax.set_title(f"{config['name']}: MAE={mae:.2f} nm, RMSE={rmse:.2f} nm, R²={r2:.4f}",
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {filename}")
    plt.close(fig)


def visualize_all_configurations(
    segment: dict,
    n_epochs: int = 50,
    window: int = 512
):
    """Generate visualizations for all configurations."""
    print("\nGenerating visualizations for all configurations...")

    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']
    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / 1000.0)

    results = {}

    for config_key in LOSS_CONFIGS.keys():
        config = LOSS_CONFIGS[config_key]
        print(f"  Processing {config['name']}...")

        result = impute_tcn_ensemble(
            X_true, t_seg, t_out, obs_mask,
            config_key=config_key,
            n_epochs=n_epochs,
            window=window,
            verbose=False
        )

        X_pred = result['output_300Hz']

        # Compute metrics
        missing_mask = ~obs_mask
        y_true = X_true[missing_mask, 0]
        y_pred = X_pred[missing_mask, 0]

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

        results[config_key] = {
            'X_pred': X_pred,
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        }

        # Individual plot
        plot_ensemble_comparison(
            t_seg, X_true, X_pred,
            obs_mask=obs_mask,
            config_name=config['name'],
            filename=f"08_tcn_ensemble_{config_key}.png"
        )

    # Comparison plot
    print("\n  Creating comparison plot...")
    plot_all_configurations_comparison(
        segment, results,
        filename="08_tcn_ensemble_comparison.png"
    )

    print("\n  ✓ Visualizations complete")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_tcn_ensemble(
    segments: list,
    config_key: str = 'balanced',
    n_epochs: int = 50,
    window: int = 512,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate TCN-Ensemble on all segments."""

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available. Skipping evaluation.")
        return {}

    config = LOSS_CONFIGS[config_key]

    print("\n" + "="*80)
    print(f"EVALUATING: TCN with Ensemble Loss - {config['name']}")
    print("="*80)
    print(f"Weights: Recon={config['weights']['recon']:.1f}, "
          f"Smooth={config['weights']['smooth']:.1f}, "
          f"Spectral={config['weights']['spectral']:.1f}, "
          f"Physics={config['weights']['physics']:.1f}")
    print(f"Parameters: epochs={n_epochs}, window={window}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / 1000.0)

        result = impute_tcn_ensemble(
            X_true, t_seg, t_out, obs_mask,
            config_key=config_key,
            n_epochs=n_epochs,
            window=window,
            verbose=verbose
        )

        X_pred = result['output_300Hz']

        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            continue

        # Compute metrics
        y_true = X_true[missing_mask, 0]
        y_pred = X_pred[missing_mask, 0]

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        all_metrics.append(metrics)

        extras = result['extras']
        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f} | "
              f"Loss={extras['final_loss']:.4f} "
              f"(R:{extras['final_recon']:.3f}, "
              f"S:{extras['final_smooth']:.3f}, "
              f"Sp:{extras['final_spectral']:.3f}, "
              f"Ph:{extras['final_physics']:.3f})")

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
    print()

    return avg_metrics


def compare_all_configurations(segments: list, n_epochs: int = 50, window: int = 512):
    """Compare all weight configurations."""

    print("\n" + "="*80)
    print("COMPARING ALL ENSEMBLE LOSS CONFIGURATIONS")
    print("="*80)

    results = {}

    for config_key in LOSS_CONFIGS.keys():
        metrics = evaluate_tcn_ensemble(
            segments,
            config_key=config_key,
            n_epochs=n_epochs,
            window=window,
            verbose=False
        )
        results[config_key] = metrics

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Ensemble Loss Configuration Comparison")
    print("="*80)
    print(f"{'Configuration':<25} {'MAE (nm)':<12} {'RMSE (nm)':<12} {'R²':<10}")
    print("-" * 80)

    for config_key, metrics in results.items():
        config_name = LOSS_CONFIGS[config_key]['name']
        mae = metrics.get('MAE', np.nan)
        rmse = metrics.get('RMSE', np.nan)
        r2 = metrics.get('R2', np.nan)
        print(f"{config_name:<25} {mae:<12.3f} {rmse:<12.3f} {r2:<10.4f}")

    print("="*80)

    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1].get('MAE', float('inf')))
    print(f"\n✓ Best Configuration: {LOSS_CONFIGS[best_config[0]]['name']}")
    print(f"  MAE={best_config[1]['MAE']:.3f} nm, RMSE={best_config[1]['RMSE']:.3f} nm, R²={best_config[1]['R2']:.4f}")

    return results


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

    # Compare all configurations
    results = compare_all_configurations(
        segments,
        n_epochs=50,
        window=512
    )

    # Generate visualizations
    print("\n" + "="*80)
    visualize_all_configurations(
        segments[0],
        n_epochs=50,
        window=512
    )

    print("\n✓ Ensemble evaluation complete with visualizations")
    print(f"  Individual plots: 08_tcn_ensemble_{{config}}.png")
    print(f"  Comparison plot: 08_tcn_ensemble_comparison.png")
