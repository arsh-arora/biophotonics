#!/usr/bin/env python3
"""
Bidirectional Multi-Scale TCN for Trajectory Imputation
========================================================
Multi-scale bidirectional TCN with parallel branches at different dilation rates.

Architecture:
- Multiple parallel TCN branches with different dilations
- Each branch captures patterns at different time scales
- Forward and backward processing for each branch
- Fusion of multi-scale features

Features:
- Coarse scale: large dilations (long-range dependencies)
- Fine scale: small dilations (local patterns)
- Bidirectional: full sequence context
- Multi-scale fusion: weighted combination

References:
- Bai, S., et al. (2018). Empirical evaluation of TCNs.
- Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation. ICLR 2016.
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
    """Temporal block with specific dilation rate."""
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


class MultiScaleBranch(nn.Module):
    """Single branch processing at a specific scale."""
    def __init__(self, in_channels: int, hidden_channels: int,
                 dilation_list: List[int], kernel_size: int = 3, dropout: float = 0.1):
        super(MultiScaleBranch, self).__init__()

        layers = []
        num_channels = in_channels

        for dilation in dilation_list:
            layers.append(
                TemporalBlock(num_channels, hidden_channels,
                            kernel_size, dilation, dropout)
            )
            num_channels = hidden_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BidirectionalMultiScaleTCN(nn.Module):
    """Bidirectional Multi-Scale TCN with parallel branches."""
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super(BidirectionalMultiScaleTCN, self).__init__()

        # Define multiple scales (different dilation patterns)
        # Fine scale: small dilations [1, 2, 4]
        # Medium scale: moderate dilations [2, 4, 8]
        # Coarse scale: large dilations [4, 8, 16]
        self.scale_dilations = {
            'fine': [1, 2, 4],
            'medium': [2, 4, 8],
            'coarse': [4, 8, 16]
        }

        # Forward branches for each scale
        self.forward_branches = nn.ModuleDict({
            scale: MultiScaleBranch(in_channels, hidden_channels,
                                   dilations, kernel_size, dropout)
            for scale, dilations in self.scale_dilations.items()
        })

        # Backward branches for each scale
        self.backward_branches = nn.ModuleDict({
            scale: MultiScaleBranch(in_channels, hidden_channels,
                                   dilations, kernel_size, dropout)
            for scale, dilations in self.scale_dilations.items()
        })

        # Fusion layer (combine all scales and directions)
        n_branches = len(self.scale_dilations) * 2  # forward + backward for each scale
        self.fusion = nn.Sequential(
            weight_norm(nn.Conv1d(hidden_channels * n_branches, hidden_channels * 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(hidden_channels * 2, hidden_channels, 1)),
            nn.ReLU()
        )

        # Output head
        self.output_head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T) input

        Returns:
            out: (B, C_out, T) output
        """
        all_features = []

        # Process each scale in forward direction
        for scale, branch in self.forward_branches.items():
            features = branch(x)
            all_features.append(features)

        # Process each scale in backward direction
        x_backward = torch.flip(x, dims=[2])
        for scale, branch in self.backward_branches.items():
            features = branch(x_backward)
            features = torch.flip(features, dims=[2])
            all_features.append(features)

        # Concatenate all multi-scale bidirectional features
        combined = torch.cat(all_features, dim=1)  # (B, H*6, T)

        # Fuse features
        fused = self.fusion(combined)  # (B, H, T)

        # Output
        out = self.output_head(fused)
        return out


def train_model(model, dataloader, n_epochs=50, lr=1e-3, device=DEVICE, verbose=False):
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

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: Loss = {avg_loss:.6f}")

    return history


def impute_bitcn_multiscale(X, t, t_out, obs_mask, n_epochs=50, window=512,
                           step=256, batch_size=16, lr=1e-3, hidden_channels=64,
                           dropout=0.1, verbose=False):
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

    model = BidirectionalMultiScaleTCN(
        in_channels=D + 1,
        out_channels=D,
        hidden_channels=hidden_channels,
        kernel_size=3,
        dropout=dropout
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Scales: fine [1,2,4], medium [2,4,8], coarse [4,8,16]")

    history = train_model(model, dataloader, n_epochs=n_epochs, lr=lr,
                         device=DEVICE, verbose=verbose)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)
        m_tensor = torch.tensor(obs_mask[:, None].astype(np.float32)).unsqueeze(0)

        x_masked = x_tensor * m_tensor
        x_input = torch.cat([x_masked, m_tensor], dim=-1).transpose(1, 2).to(DEVICE)

        x_pred = model(x_input)
        x_recon = x_pred.transpose(1, 2).squeeze(0).cpu().numpy()

    X_recon_300Hz = x_recon * X_std + X_mean

    X_recon_1000Hz = np.zeros((M, D))
    for d in range(D):
        cs = CubicSpline(t, X_recon_300Hz[:, d])
        X_recon_1000Hz[:, d] = cs(t_out)

    return {
        'output_300Hz': X_recon_300Hz,
        'output_1000Hz': X_recon_1000Hz,
        'mean': None,
        'var': None,
        'extras': {'history': history, 'final_loss': history['loss'][-1] if history['loss'] else np.nan}
    }


def evaluate_bitcn_multiscale(segments, n_epochs=50, window=512, config=CONFIG, verbose=False):
    all_true, all_pred = [], []

    for i, seg in enumerate(segments):
        if verbose:
            print(f"\n  Segment {i+1}/{len(segments)}")

        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_bitcn_multiscale(X_true, t_seg, t_out, obs_mask,
                                        n_epochs=n_epochs, window=window, verbose=verbose)

        y_true_1000 = np.interp(t_out, t_seg, X_true)
        obs_mask_1000 = np.interp(t_out, t_seg, obs_mask.astype(float)) > 0.5
        eval_mask = ~obs_mask_1000

        all_true.append(y_true_1000[eval_mask])
        all_pred.append(result['output_1000Hz'][:, 0][eval_mask])

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    residual = y_true_all - y_pred_all
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    print("\n" + "="*80)
    print("Results on Missing Points @ 1000Hz:")
    print("="*80)
    print(f"MAE:  {mae:.3f} nm")
    print(f"RMSE: {rmse:.3f} nm")
    print(f"R²:   {r2:.4f}")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def plot_results(t, y_true, y_pred, obs_mask, eval_mask, method_name, filename=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, y_true, 'k-', linewidth=1.2, label='Ground Truth', alpha=0.7)
    axes[0].plot(t[obs_mask], y_true[obs_mask], 'go', markersize=2, label='Observed', alpha=0.5)
    axes[0].plot(t[eval_mask], y_pred[eval_mask], 'r-', linewidth=1.0, label='Imputed', alpha=0.8)
    axes[0].set_ylabel('Position (nm)', fontsize=11)
    axes[0].set_title(f'{method_name} - Full Trajectory', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    residual = y_true - y_pred
    axes[1].plot(t, residual, 'b-', linewidth=0.8, alpha=0.6)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1].fill_between(t, residual, 0, alpha=0.3)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Residual (nm)', fontsize=11)
    axes[1].set_title('Prediction Error', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_zoom(t, y_true, y_pred, obs_mask, eval_mask, method_name, filename=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    zoom_mask = t <= 0.5
    t_zoom = t[zoom_mask]
    y_true_zoom = y_true[zoom_mask]
    y_pred_zoom = y_pred[zoom_mask]
    obs_zoom = obs_mask[zoom_mask]
    eval_zoom = eval_mask[zoom_mask]

    ax.plot(t_zoom, y_true_zoom, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.7)
    ax.plot(t_zoom[obs_zoom], y_true_zoom[obs_zoom], 'go', markersize=4, label='Observed', alpha=0.6)
    ax.plot(t_zoom[eval_zoom], y_pred_zoom[eval_zoom], 'r-', linewidth=1.2, label='Imputed', alpha=0.9)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (nm)', fontsize=11)
    ax.set_title(f'{method_name} - Zoomed View (First 0.5s)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)


def visualize_sample_result(segment, n_epochs=50, config=CONFIG):
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_bitcn_multiscale(X_true, t_seg, t_out, obs_mask,
                                    n_epochs=n_epochs, verbose=True)

    y_true_1000hz = np.interp(t_out, t_seg, X_true)
    y_pred_1000hz = result['output_1000Hz'][:, 0]

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

    plot_results(t_out, y_true_1000hz, y_pred_1000hz, obs_mask_1000, eval_mask,
                'Bidirectional Multi-Scale TCN', filename='15_tcn_bidirectional_multiscale_result.png')
    plot_zoom(t_out, y_true_1000hz, y_pred_1000hz, obs_mask_1000, eval_mask,
             'Bidirectional Multi-Scale TCN', filename='15_tcn_bidirectional_multiscale_zoom.png')


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch is not installed. Install with: pip install torch")
        sys.exit(1)

    print("Loading data and segments...")
    t, X, X_detrended, ou_params, segments = main_setup()
    print(f"Running on device: {DEVICE}")

    metrics = evaluate_bitcn_multiscale(segments, n_epochs=50, window=512,
                                       config=CONFIG, verbose=False)

    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], n_epochs=50, config=CONFIG)

    print("\n✓ Bidirectional Multi-Scale TCN evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
