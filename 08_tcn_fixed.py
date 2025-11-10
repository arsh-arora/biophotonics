#!/usr/bin/env python3
"""
Temporal Convolutional Network (TCN) for Trajectory Imputation
===============================================================
Deep learning approach using causal dilated convolutions for masked reconstruction.

Architecture:
- Causal 1D convolutions (no future information leakage)
- Dilated convolutions for large receptive field
- Residual connections for gradient flow
- Input: [trajectory, observation_mask] concatenated
- Output: reconstructed trajectory
- Loss: MSE only on missing points

Key Features:
- Handles variable-length gaps
- Learns temporal patterns from data
- No explicit physics model
- Requires training data with diverse gap patterns

Training Strategy:
1. Normalize trajectory data
2. Create overlapping windows with random gaps
3. Train to reconstruct missing points
4. Full-sequence inference for test data

References:
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic
  convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- Oord, A. v. d., et al. (2016). WaveNet: A generative model for raw audio.
  arXiv:1609.03499.
"""

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional
import warnings

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
    Dataset for training TCN with windowed samples.

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
# TCN ARCHITECTURE
# ============================================================================

class Chomp1d(nn.Module):
    """
    Removes rightmost padding to ensure causality.

    After causal padding, we have future information on the right.
    This module removes it to maintain causality.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor

        Returns:
            x: (B, C, T - chomp_size) tensor
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal block with two causal dilated convolutions + residual connection.

    Structure:
        x -> [Conv -> Chomp -> ReLU -> Conv -> Chomp -> ReLU] -> + -> ReLU -> out
        |___________________ skip connection ___________________|
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

        # Causal padding: pad on left side only
        padding = (kernel_size - 1) * dilation

        # Two convolutional layers with chomping
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential network
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
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


class TCNImputer(nn.Module):
    """
    Temporal Convolutional Network for trajectory imputation.

    Uses stacked temporal blocks with exponentially increasing dilation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 5,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: input channels (trajectory_dim + 1 for mask)
            out_channels: output channels (trajectory_dim)
            num_levels: number of temporal blocks
            hidden_channels: hidden channel dimension
            kernel_size: convolution kernel size
            dropout: dropout probability
        """
        super(TCNImputer, self).__init__()

        layers = []
        num_channels = in_channels

        # Stack temporal blocks with exponentially increasing dilation
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    num_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            num_channels = hidden_channels

        self.network = nn.Sequential(*layers)

        # Output head (1x1 conv)
        self.output_head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) input tensor [trajectory; mask]

        Returns:
            out: (B, C_out, T) reconstructed trajectory
        """
        features = self.network(x)
        out = self.output_head(features)
        return out


# ============================================================================
# TRAINING
# ============================================================================

def train_tcn_model(
    model: nn.Module,
    dataloader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = False
) -> Dict[str, list]:
    """
    Train TCN model with masked reconstruction loss.

    Args:
        model: TCN model
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
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

def impute_tcn(
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
    """
    Impute trajectory using TCN with masked reconstruction.

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

    if verbose:
        print(f"\n  Training TCN:")
        print(f"    Data shape: {X.shape}")
        print(f"    Observation rate: {obs_mask.mean()*100:.1f}%")
        print(f"    Window: {window}, Step: {step}")
        print(f"    Levels: {num_levels}, Hidden: {hidden_channels}")

    # Create dataset and dataloader
    dataset = MaskedImputeDataset(X_norm, obs_mask, window=window, step=step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if verbose:
        print(f"    Training samples: {len(dataset)}")

    # Initialize model
    model = TCNImputer(
        in_channels=D + 1,  # trajectory + mask
        out_channels=D,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        kernel_size=5,
        dropout=dropout
    )

    # Train model
    history = train_tcn_model(
        model, dataloader,
        n_epochs=n_epochs,
        lr=lr,
        device=DEVICE,
        verbose=verbose
    )

    # Full-sequence inference
    model.eval()
    with torch.no_grad():
        # Prepare full sequence
        x_full = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, N, D)
        m_full = torch.tensor(obs_mask[:, None].astype(np.float32), device=DEVICE).unsqueeze(0)  # (1, N, 1)

        # Mask input
        x_masked = x_full * m_full

        # Concatenate and transpose
        x_input = torch.cat([x_masked, m_full], dim=-1).transpose(1, 2)  # (1, D+1, N)

        # Forward pass
        x_pred = model(x_input).transpose(1, 2).squeeze(0).cpu().numpy()  # (N, D)

    # Denormalize
    X_recon = x_pred * X_std + X_mean

    # Interpolate to 1000 Hz
    mu_1000 = np.zeros((M, D))
    for d in range(D):
        try:
            cs = CubicSpline(t, X_recon[:, d], bc_type='natural', extrapolate=True)
            mu_1000[:, d] = cs(t_out)
        except Exception as e:
            warnings.warn(f"Spline failed for dim {d}: {e}")
            mu_1000[:, d] = np.interp(t_out, t, X_recon[:, d])

    # Diagnostics
    extras = {
        'final_loss': history['loss'][-1] if history['loss'] else np.nan,
        'n_epochs': n_epochs,
        'n_train_samples': len(dataset),
        'model_params': sum(p.numel() for p in model.parameters())
    }

    return {
        'output_300Hz': X_recon,
        'output_1000Hz': mu_1000,
        'mean': None,
        'var': None,
        'extras': extras
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_tcn(
    segments: list,
    n_epochs: int = 50,
    window: int = 512,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate TCN on all segments."""

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available. Skipping evaluation.")
        return {}

    print("\n" + "="*80)
    print(f"EVALUATING: TCN (Temporal Convolutional Network)")
    print("="*80)
    print(f"Parameters: epochs={n_epochs}, window={window}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_tcn(
            X_true, t_seg, t_out, obs_mask,
            n_epochs=n_epochs,
            window=window,
            verbose=verbose
        )

        X_pred = result['output_300Hz']

        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            continue

        metrics = compute_metrics(
            X_true, X_pred, y_var=None, mask=missing_mask
        )

        all_metrics.append(metrics)

        params = result['extras']
        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f} | Loss={params['final_loss']:.6f}")

    if not all_metrics:
        return {}

    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else np.nan
        avg_metrics[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0

    print()
    print("Average metrics:")
    print(f"  MAE:  {avg_metrics['MAE']:.3f} ± {avg_metrics['MAE_std']:.3f} nm")
    print(f"  RMSE: {avg_metrics['RMSE']:.3f} ± {avg_metrics['RMSE_std']:.3f} nm")
    print(f"  R²:   {avg_metrics['R2']:.4f} ± {avg_metrics['R2_std']:.4f}")
    print()

    return avg_metrics


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

    result = impute_tcn(
        X_true, t_seg, t_out, obs_mask,
        n_epochs=n_epochs,
        verbose=True
    )

    X_pred = result['output_300Hz']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=None,
        obs_mask=obs_mask,
        method_name="TCN (Temporal Convolutional Network)",
        filename="08_tcn_result.png"
    )

    # Zoom
    missing_indices = np.where(~obs_mask)[0]
    if len(missing_indices) > 0:
        gaps = np.split(missing_indices, np.where(np.diff(missing_indices) > 1)[0] + 1)
        longest_gap = max(gaps, key=len)

        if len(longest_gap) > 10:
            pad = 50
            zoom_start = max(0, longest_gap[0] - pad)
            zoom_end = min(len(t_seg), longest_gap[-1] + pad)

            plot_imputation_comparison(
                t_seg[zoom_start:zoom_end],
                X_true[zoom_start:zoom_end],
                X_pred[zoom_start:zoom_end],
                X_var=None,
                obs_mask=obs_mask[zoom_start:zoom_end],
                method_name="TCN (Zoom)",
                filename="08_tcn_zoom.png"
            )


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
    metrics = evaluate_tcn(
        segments,
        n_epochs=50,
        window=512,
        config=CONFIG,
        verbose=False
    )

    # Visualize
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], n_epochs=50, config=CONFIG)

    print("\n✓ TCN evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
