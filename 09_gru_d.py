#!/usr/bin/env python3
"""
GRU-D: Gated Recurrent Unit with Decay for Missing Data Imputation
====================================================================
Bidirectional RNN that explicitly models missing data patterns and time intervals.

Key Features:
1. Decay mechanism: values decay toward empirical mean based on time since last observation
2. Input representation: [value_filled, mask, time_delta]
3. Bidirectional processing: forward and backward RNNs averaged
4. Mask-aware training: loss only on missing points

Model:
    Input at time t:
        - x_t: observed value (or filled with previous/mean)
        - m_t: mask (1 = observed, 0 = missing)
        - δ_t: time since last observation

    Decay:
        γ_t = exp(-max(0, W_γ δ_t + b_γ))
        x̃_t = m_t · x_t + (1 - m_t) · (γ_t · x̄ + (1 - γ_t) · x_mean)

    where x̄ is the last observed value.

References:
- Che, Z., et al. (2018). Recurrent neural networks for multivariate time series
  with missing values. Scientific reports, 8(1), 6085.
- Cao, W., et al. (2018). BRITS: Bidirectional recurrent imputation for time series.
  NeurIPS 2018.
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

class TimeSeriesImputeDataset(Dataset):
    """
    Dataset for GRU-D training with time-aware features.

    Prepares (value, mask, time_delta) tuples for each timestep.
    """
    def __init__(
        self,
        x: np.ndarray,
        obs_mask: np.ndarray,
        dt: float = 1.0
    ):
        """
        Args:
            x: (N, D) trajectory data
            obs_mask: (N,) boolean mask
            dt: time step between samples
        """
        self.x = torch.tensor(x, dtype=torch.float32)  # (N, D)
        self.m = torch.tensor(obs_mask[:, None].astype(np.float32))  # (N, 1)
        self.dt = dt

        # Compute time deltas (time since last observation)
        N = len(x)
        time_deltas = np.zeros(N)
        last_obs_time = 0.0

        for i in range(N):
            if obs_mask[i]:
                time_deltas[i] = (i - last_obs_time) * dt
                last_obs_time = i
            else:
                time_deltas[i] = (i - last_obs_time) * dt

        self.delta = torch.tensor(time_deltas[:, None], dtype=torch.float32)  # (N, 1)

    def __len__(self):
        return 1  # Full sequence

    def __getitem__(self, idx):
        """
        Returns:
            x: (N, D) trajectory
            m: (N, 1) mask
            delta: (N, 1) time deltas
        """
        return self.x, self.m, self.delta


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    MSE loss only on missing positions.

    Args:
        pred: (B, N, D) predictions
        target: (B, N, D) ground truth
        mask: (B, N, 1) observation mask (1 = observed, 0 = missing)

    Returns:
        loss: scalar tensor
    """
    missing_mask = (1.0 - mask)  # 1 where missing
    squared_errors = (pred - target) ** 2
    masked_errors = squared_errors * missing_mask
    denom = missing_mask.sum().clamp_min(1.0)
    return masked_errors.sum() / denom


# ============================================================================
# GRU-D ARCHITECTURE
# ============================================================================

class GRUDCell(nn.Module):
    """
    GRU-D cell with decay mechanism for missing data.

    Extends standard GRU with:
    - Input decay based on time since last observation
    - Hidden state decay
    - Mask-aware processing
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUDCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Decay parameters
        self.decay_input = nn.Linear(1, input_dim)  # γ_x for input decay
        self.decay_hidden = nn.Linear(1, hidden_dim)  # γ_h for hidden decay

        # Standard GRU gates
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Reset gate
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Update gate
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)  # Candidate hidden

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
        h: torch.Tensor,
        x_mean: torch.Tensor,
        x_last: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, D) current input
            m: (B, 1) current mask
            delta: (B, 1) time since last observation
            h: (B, H) previous hidden state
            x_mean: (B, D) empirical mean
            x_last: (B, D) last observed value

        Returns:
            h_new: (B, H) new hidden state
            x_last_new: (B, D) updated last observed value
        """
        # Input decay
        gamma_x = torch.exp(-torch.relu(self.decay_input(delta)))  # (B, D)
        x_decayed = m * x + (1 - m) * (gamma_x * x_last + (1 - gamma_x) * x_mean)

        # Hidden decay
        gamma_h = torch.exp(-torch.relu(self.decay_hidden(delta)))  # (B, H)
        h_decayed = gamma_h * h

        # Concatenate input and hidden
        combined = torch.cat([x_decayed, h_decayed], dim=1)

        # GRU gates
        r = torch.sigmoid(self.W_r(combined))  # Reset gate
        z = torch.sigmoid(self.W_z(combined))  # Update gate

        # Candidate hidden state
        combined_r = torch.cat([x_decayed, r * h_decayed], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_r))

        # New hidden state
        h_new = (1 - z) * h_decayed + z * h_tilde

        # Update last observed value
        x_last_new = m * x + (1 - m) * x_last

        return h_new, x_last_new


class GRUD(nn.Module):
    """
    GRU-D: Gated Recurrent Unit with Decay.

    Processes sequences with missing data using decay mechanisms.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(GRUD, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Stack of GRU-D cells
        self.cells = nn.ModuleList([
            GRUDCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout) if num_layers > 1 else None

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
        x_mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input sequence
            m: (B, N, 1) mask sequence
            delta: (B, N, 1) time delta sequence
            x_mean: (B, D) empirical mean

        Returns:
            output: (B, N, D) reconstructed sequence
        """
        B, N, D = x.shape

        # Initialize hidden states and last observed values
        h = [torch.zeros(B, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        x_last = [x_mean.clone() for _ in range(self.num_layers)]

        outputs = []

        # Process sequence
        for t in range(N):
            x_t = x[:, t, :]  # (B, D)
            m_t = m[:, t, :]  # (B, 1)
            delta_t = delta[:, t, :]  # (B, 1)

            # Pass through layers
            h_input = x_t
            for layer_idx in range(self.num_layers):
                h[layer_idx], x_last[layer_idx] = self.cells[layer_idx](
                    h_input if layer_idx == 0 else h[layer_idx - 1],
                    m_t if layer_idx == 0 else torch.ones_like(m_t),  # Mask only at input layer
                    delta_t,
                    h[layer_idx],
                    x_mean if layer_idx == 0 else torch.zeros(B, self.hidden_dim, device=x.device),
                    x_last[layer_idx]
                )

                # Dropout between layers
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    h[layer_idx] = self.dropout(h[layer_idx])

            # Project to output
            out_t = self.output_layer(h[-1])  # (B, D)
            outputs.append(out_t)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (B, N, D)

        return output


class BidirectionalGRUD(nn.Module):
    """
    Bidirectional GRU-D: processes sequence in both directions and averages.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(BidirectionalGRUD, self).__init__()

        self.forward_grud = GRUD(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.backward_grud = GRUD(input_dim, hidden_dim, output_dim, num_layers, dropout)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        delta: torch.Tensor,
        x_mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input sequence
            m: (B, N, 1) mask sequence
            delta: (B, N, 1) time delta sequence
            x_mean: (B, D) empirical mean

        Returns:
            output: (B, N, D) reconstructed sequence (average of forward and backward)
        """
        # Forward pass
        out_forward = self.forward_grud(x, m, delta, x_mean)

        # Backward pass (reverse sequence)
        x_rev = torch.flip(x, dims=[1])
        m_rev = torch.flip(m, dims=[1])
        delta_rev = torch.flip(delta, dims=[1])

        out_backward = self.backward_grud(x_rev, m_rev, delta_rev, x_mean)
        out_backward = torch.flip(out_backward, dims=[1])  # Reverse back

        # Average forward and backward
        output = (out_forward + out_backward) / 2.0

        return output


# ============================================================================
# TRAINING
# ============================================================================

def train_grud_model(
    model: nn.Module,
    dataloader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = DEVICE,
    verbose: bool = False
) -> Dict[str, list]:
    """Train GRU-D model with masked reconstruction loss."""
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

        for x_batch, m_batch, delta_batch in dataloader:
            x_batch = x_batch.to(device)  # (B, N, D)
            m_batch = m_batch.to(device)  # (B, N, 1)
            delta_batch = delta_batch.to(device)  # (B, N, 1)

            # Compute empirical mean from observed data
            x_mean = (x_batch * m_batch).sum(dim=1) / m_batch.sum(dim=1).clamp_min(1.0)  # (B, D)

            # Forward pass
            x_pred = model(x_batch, m_batch, delta_batch, x_mean)  # (B, N, D)

            # Compute loss
            loss = masked_mse_loss(x_pred, x_batch, m_batch)

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

def impute_grud(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 1,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using bidirectional GRU-D.

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds
        obs_mask: (N,) boolean mask
        n_epochs: number of training epochs
        batch_size: training batch size
        lr: learning rate
        hidden_dim: hidden dimension
        num_layers: number of GRU layers
        dropout: dropout probability
        verbose: print training progress

    Returns:
        dict with:
            - output_300Hz: (N, D) reconstructed trajectory
            - output_1000Hz: (M, D) interpolated trajectory
            - mean: None
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
    dt = np.median(np.diff(t))

    # Normalize
    X_mean = X[obs_mask].mean(axis=0)
    X_std = X[obs_mask].std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"\n  Training Bidirectional GRU-D:")
        print(f"    Data shape: {X.shape}")
        print(f"    Observation rate: {obs_mask.mean()*100:.1f}%")
        print(f"    Hidden dim: {hidden_dim}, Layers: {num_layers}")

    # Create dataset
    dataset = TimeSeriesImputeDataset(X_norm, obs_mask, dt=dt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = BidirectionalGRUD(
        input_dim=D,
        hidden_dim=hidden_dim,
        output_dim=D,
        num_layers=num_layers,
        dropout=dropout
    )

    # Train
    history = train_grud_model(
        model, dataloader,
        n_epochs=n_epochs,
        lr=lr,
        device=DEVICE,
        verbose=verbose
    )

    # Inference
    model.eval()
    with torch.no_grad():
        x_full = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, N, D)
        m_full = torch.tensor(obs_mask[:, None].astype(np.float32), device=DEVICE).unsqueeze(0)  # (1, N, 1)

        # Compute time deltas
        time_deltas = np.zeros(N)
        last_obs_idx = 0
        for i in range(N):
            if obs_mask[i]:
                time_deltas[i] = (i - last_obs_idx) * dt
                last_obs_idx = i
            else:
                time_deltas[i] = (i - last_obs_idx) * dt

        delta_full = torch.tensor(time_deltas[:, None], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, N, 1)
        x_mean_tensor = torch.tensor(X_norm[obs_mask].mean(axis=0), dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, D)

        x_pred = model(x_full, m_full, delta_full, x_mean_tensor).squeeze(0).cpu().numpy()  # (N, D)

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

    extras = {
        'final_loss': history['loss'][-1] if history['loss'] else np.nan,
        'n_epochs': n_epochs,
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

def evaluate_grud(
    segments: list,
    n_epochs: int = 50,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate GRU-D on all segments."""

    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available. Skipping evaluation.")
        return {}

    print("\n" + "="*80)
    print(f"EVALUATING: Bidirectional GRU-D")
    print("="*80)
    print(f"Parameters: epochs={n_epochs}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_grud(
            X_true, t_seg, t_out, obs_mask,
            n_epochs=n_epochs,
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

    result = impute_grud(
        X_true, t_seg, t_out, obs_mask,
        n_epochs=n_epochs,
        verbose=True
    )

    X_pred = result['output_300Hz']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=None,
        obs_mask=obs_mask,
        method_name="Bidirectional GRU-D",
        filename="09_gru_d_result.png"
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
                method_name="GRU-D (Zoom)",
                filename="09_gru_d_zoom.png"
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
    metrics = evaluate_grud(
        segments,
        n_epochs=50,
        config=CONFIG,
        verbose=False
    )

    # Visualize
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], n_epochs=50, config=CONFIG)

    print("\n✓ GRU-D evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
