#!/usr/bin/env python3
"""
Bidirectional TCN Ensemble for Trajectory Imputation
=====================================================
Ensemble of multiple bidirectional TCN models for improved predictions.

Architecture:
- Multiple independent bidirectional TCN models
- Different random initializations
- Ensemble predictions (mean and variance)
- Uncertainty quantification from ensemble spread

Features:
- Reduced overfitting through model averaging
- Better uncertainty estimates
- More robust predictions
- Captures model uncertainty

Ensemble Strategy:
- Train K models independently
- Predictions: mean of ensemble
- Uncertainty: variance across ensemble + individual variances

References:
- Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty.
- Dietterich, T. G. (2000). Ensemble methods in machine learning.
"""

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Dict, List
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils import weight_norm
    TORCH_AVAILABLE = True

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None


class MaskedImputeDataset(Dataset):
    def __init__(self, x: np.ndarray, obs_mask: np.ndarray,
                 window: int = 512, step: int = 256):
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


def masked_mse_loss(pred, target, mask):
    missing_mask = (1.0 - mask)
    squared_errors = (pred - target) ** 2
    masked_errors = squared_errors * missing_mask
    denom = missing_mask.sum().clamp_min(1.0)
    return masked_errors.sum() / denom


class TemporalBlock(nn.Module):
    """Temporal block for bidirectional TCN."""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )

        self.downsample = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BidirectionalTCN(nn.Module):
    """Single bidirectional TCN model for ensemble."""
    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 5,
                 hidden_channels: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super(BidirectionalTCN, self).__init__()

        # Forward network
        forward_layers = []
        num_channels = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            forward_layers.append(
                TemporalBlock(num_channels, hidden_channels,
                            kernel_size, dilation, dropout)
            )
            num_channels = hidden_channels
        self.forward_network = nn.Sequential(*forward_layers)

        # Backward network
        backward_layers = []
        num_channels = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            backward_layers.append(
                TemporalBlock(num_channels, hidden_channels,
                            kernel_size, dilation, dropout)
            )
            num_channels = hidden_channels
        self.backward_network = nn.Sequential(*backward_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        forward_features = self.forward_network(x)

        x_backward = torch.flip(x, dims=[2])
        backward_features = self.backward_network(x_backward)
        backward_features = torch.flip(backward_features, dims=[2])

        combined = torch.cat([forward_features, backward_features], dim=1)
        out = self.output_head(combined)
        return out


class BidirectionalTCNEnsemble:
    """Ensemble of bidirectional TCN models."""
    def __init__(self, n_models: int, in_channels: int, out_channels: int,
                 num_levels: int = 5, hidden_channels: int = 64,
                 kernel_size: int = 3, dropout: float = 0.1):
        self.n_models = n_models
        self.models = []

        for i in range(n_models):
            model = BidirectionalTCN(
                in_channels=in_channels,
                out_channels=out_channels,
                num_levels=num_levels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.models.append(model)

    def train_ensemble(self, dataloader, n_epochs=50, lr=1e-3, device=DEVICE, verbose=False):
        """Train all models in ensemble."""
        histories = []

        for i, model in enumerate(self.models):
            if verbose:
                print(f"\n  Training ensemble member {i+1}/{self.n_models}")

            model = model.to(device)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )

            history = {'loss': []}

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for x_batch, m_batch in dataloader:
                    x_batch = x_batch.to(device)
                    m_batch = m_batch.to(device)

                    x_masked = x_batch * m_batch
                    x_input = torch.cat([x_masked, m_batch], dim=-1).transpose(1, 2)
                    x_target = x_batch.transpose(1, 2)

                    x_pred = model(x_input)

                    loss = masked_mse_loss(
                        x_pred.transpose(1, 2),
                        x_target.transpose(1, 2),
                        m_batch
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / n_batches
                history['loss'].append(avg_loss)
                scheduler.step()

                if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                    print(f"      Epoch {epoch+1:3d}/{n_epochs}: Loss = {avg_loss:.6f}")

            histories.append(history)

        return histories

    def predict_ensemble(self, x_input, device=DEVICE):
        """
        Get ensemble predictions.

        Returns:
            mean: (B, D, T) ensemble mean
            var: (B, D, T) ensemble variance (epistemic uncertainty)
        """
        all_predictions = []

        for model in self.models:
            model.eval()
            model = model.to(device)

            with torch.no_grad():
                pred = model(x_input)
                all_predictions.append(pred.cpu().numpy())

        # Stack predictions
        all_predictions = np.stack(all_predictions, axis=0)  # (K, B, D, T)

        # Ensemble mean and variance
        ensemble_mean = np.mean(all_predictions, axis=0)  # (B, D, T)
        ensemble_var = np.var(all_predictions, axis=0)  # (B, D, T)

        return ensemble_mean, ensemble_var


def impute_bitcn_ensemble(X, t, t_out, obs_mask, n_models=5, n_epochs=50,
                         window=512, step=256, batch_size=16, lr=1e-3,
                         num_levels=5, hidden_channels=64, dropout=0.1, verbose=False):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D = X.shape
    M = len(t_out)

    X_mean = X[obs_mask].mean(axis=0)
    X_std = X[obs_mask].std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    dataset = MaskedImputeDataset(X_norm, obs_mask, window=window, step=step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create and train ensemble
    ensemble = BidirectionalTCNEnsemble(
        n_models=n_models,
        in_channels=D + 1,
        out_channels=D,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        kernel_size=3,
        dropout=dropout
    )

    if verbose:
        n_params = sum(p.numel() for p in ensemble.models[0].parameters())
        print(f"  Ensemble size: {n_models} models")
        print(f"  Parameters per model: {n_params:,}")
        print(f"  Total parameters: {n_params * n_models:,}")

    histories = ensemble.train_ensemble(dataloader, n_epochs=n_epochs, lr=lr,
                                       device=DEVICE, verbose=verbose)

    # Inference
    x_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)
    m_tensor = torch.tensor(obs_mask[:, None].astype(np.float32)).unsqueeze(0)
    x_masked = x_tensor * m_tensor
    x_input = torch.cat([x_masked, m_tensor], dim=-1).transpose(1, 2).to(DEVICE)

    ensemble_mean, ensemble_var = ensemble.predict_ensemble(x_input, device=DEVICE)

    # Extract and denormalize
    mean_recon = ensemble_mean[0].transpose(1, 0)  # (N, D)
    var_recon = ensemble_var[0].transpose(1, 0)  # (N, D)

    X_recon_300Hz = mean_recon * X_std + X_mean
    var_recon_300Hz = var_recon * (X_std ** 2)

    # Interpolate to 1000 Hz
    X_recon_1000Hz = np.zeros((M, D))
    var_1000Hz = np.zeros((M, D))

    for d in range(D):
        cs_mean = CubicSpline(t, X_recon_300Hz[:, d])
        X_recon_1000Hz[:, d] = cs_mean(t_out)

        cs_var = CubicSpline(t, var_recon_300Hz[:, d])
        var_1000Hz[:, d] = np.maximum(cs_var(t_out), 1e-6)

    return {
        'output_300Hz': X_recon_300Hz,
        'output_1000Hz': X_recon_1000Hz,
        'mean': X_recon_1000Hz,
        'var': var_1000Hz,
        'extras': {
            'histories': histories,
            'final_loss': np.mean([h['loss'][-1] for h in histories if h['loss']])
        }
    }


def evaluate_bitcn_ensemble(segments, n_models=5, n_epochs=50, window=512,
                           config=CONFIG, verbose=False):
    all_true, all_pred, all_var = [], [], []

    for i, seg in enumerate(segments):
        if verbose:
            print(f"\n  Segment {i+1}/{len(segments)}")

        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_bitcn_ensemble(X_true, t_seg, t_out, obs_mask,
                                      n_models=n_models, n_epochs=n_epochs,
                                      window=window, verbose=verbose)

        y_true_1000 = np.interp(t_out, t_seg, X_true)
        obs_mask_1000 = np.interp(t_out, t_seg, obs_mask.astype(float)) > 0.5
        eval_mask = ~obs_mask_1000

        all_true.append(y_true_1000[eval_mask])
        all_pred.append(result['output_1000Hz'][:, 0][eval_mask])
        all_var.append(result['var'][:, 0][eval_mask])

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    y_var_all = np.concatenate(all_var)

    residual = y_true_all - y_pred_all
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    # Calibration with ensemble variance
    z_scores = np.abs(residual) / np.sqrt(y_var_all + 1e-6)
    calib_1sigma = np.mean(z_scores <= 1.0)
    calib_2sigma = np.mean(z_scores <= 2.0)

    print("\n" + "="*80)
    print("Results on Missing Points @ 1000Hz:")
    print("="*80)
    print(f"MAE:  {mae:.3f} nm")
    print(f"RMSE: {rmse:.3f} nm")
    print(f"R²:   {r2:.4f}")
    print(f"Calibration @ 1σ: {calib_1sigma*100:.1f}% (ideal: 68.3%)")
    print(f"Calibration @ 2σ: {calib_2sigma*100:.1f}% (ideal: 95.4%)")

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Calib_1sigma': calib_1sigma,
        'Calib_2sigma': calib_2sigma
    }


def plot_results(t, y_true, y_pred, y_std, obs_mask, eval_mask, method_name, filename=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, y_true, 'k-', linewidth=1.2, label='Ground Truth', alpha=0.7)
    axes[0].plot(t[obs_mask], y_true[obs_mask], 'go', markersize=2, label='Observed', alpha=0.5)
    axes[0].plot(t[eval_mask], y_pred[eval_mask], 'r-', linewidth=1.0, label='Ensemble Mean', alpha=0.8)
    axes[0].fill_between(t[eval_mask],
                        (y_pred - 2*y_std)[eval_mask],
                        (y_pred + 2*y_std)[eval_mask],
                        color='r', alpha=0.2, label='±2σ (Ensemble)')
    axes[0].set_ylabel('Position (nm)', fontsize=11)
    axes[0].set_title(f'{method_name} - Full Trajectory', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    residual = y_true - y_pred
    axes[1].plot(t, residual, 'b-', linewidth=0.8, alpha=0.6)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1].fill_between(t, -2*y_std, 2*y_std, color='r', alpha=0.2)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Residual (nm)', fontsize=11)
    axes[1].set_title('Prediction Error with Ensemble Uncertainty', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_zoom(t, y_true, y_pred, y_std, obs_mask, eval_mask, method_name, filename=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    zoom_mask = t <= 0.5
    t_zoom = t[zoom_mask]
    y_true_zoom = y_true[zoom_mask]
    y_pred_zoom = y_pred[zoom_mask]
    y_std_zoom = y_std[zoom_mask]
    obs_zoom = obs_mask[zoom_mask]
    eval_zoom = eval_mask[zoom_mask]

    ax.plot(t_zoom, y_true_zoom, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.7)
    ax.plot(t_zoom[obs_zoom], y_true_zoom[obs_zoom], 'go', markersize=4, label='Observed', alpha=0.6)
    ax.plot(t_zoom[eval_zoom], y_pred_zoom[eval_zoom], 'r-', linewidth=1.2, label='Ensemble Mean', alpha=0.9)
    ax.fill_between(t_zoom[eval_zoom],
                   (y_pred_zoom - 2*y_std_zoom)[eval_zoom],
                   (y_pred_zoom + 2*y_std_zoom)[eval_zoom],
                   color='r', alpha=0.2, label='±2σ (Ensemble)')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (nm)', fontsize=11)
    ax.set_title(f'{method_name} - Zoomed View (First 0.5s)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def visualize_sample_result(segment, n_models=5, n_epochs=50, config=CONFIG):
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_bitcn_ensemble(X_true, t_seg, t_out, obs_mask,
                                  n_models=n_models, n_epochs=n_epochs, verbose=True)

    y_true_1000hz = np.interp(t_out, t_seg, X_true)
    y_pred_1000hz = result['output_1000Hz'][:, 0]
    y_std_1000hz = np.sqrt(result['var'][:, 0])

    obs_mask_1000 = np.interp(t_out, t_seg, obs_mask.astype(float)) > 0.5
    eval_mask = ~obs_mask_1000

    residual = y_true_1000hz[eval_mask] - y_pred_1000hz[eval_mask]
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_true_1000hz[eval_mask] - np.mean(y_true_1000hz[eval_mask]))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    print(f"\nSample Segment Metrics:")
    print(f"  MAE:  {mae:.3f} nm")
    print(f"  RMSE: {rmse:.3f} nm")
    print(f"  R²:   {r2:.4f}")

    plot_results(t_out, y_true_1000hz, y_pred_1000hz, y_std_1000hz, obs_mask_1000, eval_mask,
                'Bidirectional TCN Ensemble', filename='15_tcn_bidirectional_ensemble_result.png')
    plot_zoom(t_out, y_true_1000hz, y_pred_1000hz, y_std_1000hz, obs_mask_1000, eval_mask,
             'Bidirectional TCN Ensemble', filename='15_tcn_bidirectional_ensemble_zoom.png')


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch is not installed. Install with: pip install torch")
        sys.exit(1)

    print("Loading data and segments...")
    t, X, X_detrended, ou_params, segments = main_setup()
    print(f"Running on device: {DEVICE}")

    metrics = evaluate_bitcn_ensemble(segments, n_models=3, n_epochs=50,
                                     window=512, config=CONFIG, verbose=False)

    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], n_models=3, n_epochs=50, config=CONFIG)

    print("\n✓ Bidirectional TCN Ensemble evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
