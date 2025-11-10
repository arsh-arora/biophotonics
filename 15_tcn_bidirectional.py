#!/usr/bin/env python3
"""
Bidirectional Temporal Convolutional Network (Bi-TCN) for Trajectory Imputation
================================================================================
Bidirectional deep learning approach using dilated convolutions for masked reconstruction.

Architecture:
- Forward TCN: processes sequence left-to-right
- Backward TCN: processes sequence right-to-left
- Fusion: concatenates or averages forward/backward features
- Input: [trajectory, observation_mask] concatenated
- Output: reconstructed trajectory
- Loss: MSE only on missing points

Key Features:
- Accesses both past and future context (non-causal)
- Larger effective receptive field
- Better for offline imputation tasks
- Handles variable-length gaps
- No explicit physics model

Advantages over Causal TCN:
- Full sequence context for imputation
- Improved accuracy on long gaps
- Better temporal smoothness

Training Strategy:
1. Normalize trajectory data
2. Create overlapping windows with random gaps
3. Train forward and backward TCNs
4. Combine predictions for reconstruction
5. Full-sequence inference for test data

References:
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic
  convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- Lea, C., et al. (2017). Temporal convolutional networks for action segmentation
  and detection. CVPR 2017.
"""

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True

    # Device selection with GPU priority
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
    """
    Dataset for training Bi-TCN with windowed samples.

    Creates overlapping windows from trajectory with observation masks.
    """
    def __init__(
        self,
        x: np.ndarray,
        obs_mask: np.ndarray,
        window: int = 512,
        step: int = 256
    ):
        """
        Args:
            x: (N, D) trajectory data
            obs_mask: (N,) boolean mask
            window: window size in samples
            step: stride between windows
        """
        self.x = torch.tensor(x, dtype=torch.float32)  # (N, D)
        self.m = torch.tensor(obs_mask[:, None].astype(np.float32))  # (N, 1)
        self.window = window
        self.step = step

        # Create window indices
        N = len(x)
        self.indices = [(i, min(i + window, N)) for i in range(0, N - window + 1, step)]

        # If no full windows, use entire sequence
        if len(self.indices) == 0:
            self.indices = [(0, N)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            x_window: (T, D) trajectory window
            m_window: (T, 1) mask window
        """
        start, end = self.indices[idx]
        return self.x[start:end], self.m[start:end]


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    MSE loss only on missing (unobserved) positions.

    Args:
        pred: (B, T, D) predictions
        target: (B, T, D) ground truth
        mask: (B, T, 1) observation mask (1 = observed, 0 = missing)

    Returns:
        loss: scalar tensor
    """
    missing_mask = (1.0 - mask)  # 1 where missing, 0 where observed
    squared_errors = (pred - target) ** 2

    # Masked loss
    masked_errors = squared_errors * missing_mask
    denom = missing_mask.sum().clamp_min(1.0)

    return masked_errors.sum() / denom


# ============================================================================
# BIDIRECTIONAL TCN ARCHITECTURE
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Temporal block with two dilated convolutions + residual connection.

    For bidirectional TCN, we don't need causal constraints (no chomping).
    Uses symmetric padding to preserve sequence length.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super(TemporalBlock, self).__init__()

        # Symmetric padding for non-causal processing
        padding = (kernel_size - 1) * dilation // 2

        # Two convolutional layers
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential network
        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )

        # Skip connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) tensor

        Returns:
            out: (B, C_out, T) tensor
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BidirectionalTCNImputer(nn.Module):
    """
    Bidirectional Temporal Convolutional Network for trajectory imputation.

    Processes sequence in both forward and backward directions, then fuses.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 5,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        fusion: str = 'concat'  # 'concat' or 'average'
    ):
        """
        Args:
            in_channels: input channels (trajectory_dim + 1 for mask)
            out_channels: output channels (trajectory_dim)
            num_levels: number of temporal blocks
            hidden_channels: hidden channel dimension
            kernel_size: convolution kernel size
            dropout: dropout probability
            fusion: how to combine forward/backward ('concat' or 'average')
        """
        super(BidirectionalTCNImputer, self).__init__()

        self.fusion = fusion

        # Forward TCN
        forward_layers = []
        num_channels = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            forward_layers.append(
                TemporalBlock(
                    num_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            num_channels = hidden_channels
        self.forward_network = nn.Sequential(*forward_layers)

        # Backward TCN (same architecture)
        backward_layers = []
        num_channels = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            backward_layers.append(
                TemporalBlock(
                    num_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            num_channels = hidden_channels
        self.backward_network = nn.Sequential(*backward_layers)

        # Output head
        if fusion == 'concat':
            self.output_head = nn.Conv1d(hidden_channels * 2, out_channels, 1)
        else:  # average
            self.output_head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) input tensor [trajectory; mask]

        Returns:
            out: (B, C_out, T) reconstructed trajectory
        """
        # Forward pass
        forward_features = self.forward_network(x)  # (B, H, T)

        # Backward pass (flip time dimension)
        x_backward = torch.flip(x, dims=[2])
        backward_features = self.backward_network(x_backward)
        backward_features = torch.flip(backward_features, dims=[2])  # (B, H, T)

        # Fusion
        if self.fusion == 'concat':
            combined = torch.cat([forward_features, backward_features], dim=1)  # (B, 2H, T)
        else:  # average
            combined = (forward_features + backward_features) / 2.0  # (B, H, T)

        # Output
        out = self.output_head(combined)
        return out


# ============================================================================
# TRAINING
# ============================================================================

def train_bitcn_model(
    model: nn.Module,
    dataloader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = False
) -> Dict[str, list]:
    """
    Train Bi-TCN model with masked reconstruction loss.

    Args:
        model: Bi-TCN model
        dataloader: training dataloader
        n_epochs: number of training epochs
        lr: learning rate
        device: device to train on
        verbose: print training progress

    Returns:
        dict with training history
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'loss': []}

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, m_batch in dataloader:
            x_batch = x_batch.to(device)  # (B, T, D)
            m_batch = m_batch.to(device)  # (B, T, 1)

            # Prepare input: concatenate trajectory and mask
            # Mask missing values with zeros
            x_masked = x_batch * m_batch

            # Transpose to (B, C, T) for Conv1d
            x_input = torch.cat([x_masked, m_batch], dim=-1).transpose(1, 2)  # (B, D+1, T)
            x_target = x_batch.transpose(1, 2)  # (B, D, T)

            # Forward pass
            x_pred = model(x_input)  # (B, D, T)

            # Compute loss (transpose back to (B, T, D) for loss function)
            loss = masked_mse_loss(
                x_pred.transpose(1, 2),
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

        scheduler.step(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: Loss = {avg_loss:.6f}")

    return history


# ============================================================================
# IMPUTATION
# ============================================================================

def impute_bitcn(
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
    fusion: str = 'concat',
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using Bi-TCN with masked reconstruction.

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds
        obs_mask: (N,) boolean mask
        n_epochs: number of training epochs
        window: window size for training
        step: stride for window sampling
        batch_size: training batch size
        lr: learning rate
        num_levels: number of TCN levels
        hidden_channels: hidden channel dimension
        dropout: dropout probability
        fusion: 'concat' or 'average'
        verbose: print training progress

    Returns:
        dict with:
            - output_300Hz: (N, D) reconstructed trajectory at native rate
            - output_1000Hz: (M, D) interpolated trajectory at target rate
            - mean: None (deterministic)
            - var: None
            - extras: dict with training info
    """
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

    # Normalize data
    X_mean = X[obs_mask].mean(axis=0)
    X_std = X[obs_mask].std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Create dataset
    dataset = MaskedImputeDataset(X_norm, obs_mask, window=window, step=step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = BidirectionalTCNImputer(
        in_channels=D + 1,
        out_channels=D,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        kernel_size=3,
        dropout=dropout,
        fusion=fusion
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Fusion strategy: {fusion}")

    # Train model
    history = train_bitcn_model(
        model, dataloader, n_epochs=n_epochs, lr=lr,
        device=DEVICE, verbose=verbose
    )

    # Inference on full sequence
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)  # (1, N, D)
        m_tensor = torch.tensor(obs_mask[:, None].astype(np.float32)).unsqueeze(0)  # (1, N, 1)

        x_masked = x_tensor * m_tensor
        x_input = torch.cat([x_masked, m_tensor], dim=-1).transpose(1, 2).to(DEVICE)  # (1, D+1, N)

        x_pred = model(x_input)  # (1, D, N)
        x_recon = x_pred.transpose(1, 2).squeeze(0).cpu().numpy()  # (N, D)

    # Denormalize
    X_recon_300Hz = x_recon * X_std + X_mean

    # Interpolate to 1000 Hz using cubic spline
    X_recon_1000Hz = np.zeros((M, D))
    for d in range(D):
        cs = CubicSpline(t, X_recon_300Hz[:, d])
        X_recon_1000Hz[:, d] = cs(t_out)

    return {
        'output_300Hz': X_recon_300Hz,
        'output_1000Hz': X_recon_1000Hz,
        'mean': None,
        'var': None,
        'extras': {
            'history': history,
            'final_loss': history['loss'][-1] if history['loss'] else np.nan
        }
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_bitcn(
    segments: list,
    n_epochs: int = 50,
    window: int = 512,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate Bi-TCN on all segments."""

    all_true = []
    all_pred = []
    all_eval_mask = []

    for i, seg in enumerate(segments):
        if verbose:
            print(f"\n  Segment {i+1}/{len(segments)}")

        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_bitcn(
            X_true, t_seg, t_out, obs_mask,
            n_epochs=n_epochs,
            window=window,
            verbose=verbose
        )

        # Interpolate ground truth to 1000 Hz for eval
        y_true_1000 = np.interp(t_out, t_seg, X_true)

        # Create evaluation mask at 1000 Hz
        obs_mask_1000 = np.interp(t_out, t_seg, obs_mask.astype(float)) > 0.5
        eval_mask = ~obs_mask_1000

        all_true.append(y_true_1000[eval_mask])
        all_pred.append(result['output_1000Hz'][:, 0][eval_mask])
        all_eval_mask.append(eval_mask)

    # Aggregate metrics
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

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(t, y_true, y_pred, obs_mask, eval_mask, method_name, filename=None):
    """Plot full trajectory results."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: trajectory
    axes[0].plot(t, y_true, 'k-', linewidth=1.2, label='Ground Truth', alpha=0.7)
    axes[0].plot(t[obs_mask], y_true[obs_mask], 'go', markersize=2,
                label='Observed', alpha=0.5)
    axes[0].plot(t[eval_mask], y_pred[eval_mask], 'r-', linewidth=1.0,
                label='Imputed', alpha=0.8)
    axes[0].set_ylabel('Position (nm)', fontsize=11)
    axes[0].set_title(f'{method_name} - Full Trajectory', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Bottom: residual
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
    else:
        plt.show()


def plot_zoom(t, y_true, y_pred, obs_mask, eval_mask, method_name, filename=None):
    """Plot zoomed-in view."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Select zoom region (first 0.5 seconds)
    zoom_mask = t <= 0.5
    t_zoom = t[zoom_mask]
    y_true_zoom = y_true[zoom_mask]
    y_pred_zoom = y_pred[zoom_mask]
    obs_zoom = obs_mask[zoom_mask]
    eval_zoom = eval_mask[zoom_mask]

    ax.plot(t_zoom, y_true_zoom, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.7)
    ax.plot(t_zoom[obs_zoom], y_true_zoom[obs_zoom], 'go', markersize=4,
           label='Observed', alpha=0.6)
    ax.plot(t_zoom[eval_zoom], y_pred_zoom[eval_zoom], 'r-', linewidth=1.2,
           label='Imputed', alpha=0.9)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (nm)', fontsize=11)
    ax.set_title(f'{method_name} - Zoomed View (First 0.5s)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_sample_result(
    segment: dict,
    n_epochs: int = 50,
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_bitcn(
        X_true, t_seg, t_out, obs_mask,
        n_epochs=n_epochs,
        verbose=True
    )

    # Interpolate ground truth
    y_true_1000hz = np.interp(t_out, t_seg, X_true)
    y_pred_1000hz = result['output_1000Hz'][:, 0]

    # Masks
    obs_mask_1000 = np.interp(t_out, t_seg, obs_mask.astype(float)) > 0.5
    eval_mask = ~obs_mask_1000

    # Metrics
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

    # Generate plots
    plot_results(t_out, y_true_1000hz, y_pred_1000hz, obs_mask_1000, eval_mask,
                'Bidirectional TCN', filename='15_tcn_bidirectional_result.png')
    plot_zoom(t_out, y_true_1000hz, y_pred_1000hz, obs_mask_1000, eval_mask,
             'Bidirectional TCN', filename='15_tcn_bidirectional_zoom.png')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch is not installed. Please install with:")
        print("  pip install torch")
        sys.exit(1)

    print("Loading data and segments from starter framework...")
    t, X, X_detrended, ou_params, segments = main_setup()

    print(f"\nRunning on device: {DEVICE}")

    # Evaluate
    metrics = evaluate_bitcn(
        segments,
        n_epochs=50,
        window=512,
        config=CONFIG,
        verbose=False
    )

    # Visualize
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], n_epochs=50, config=CONFIG)

    print("\n✓ Bidirectional TCN evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
