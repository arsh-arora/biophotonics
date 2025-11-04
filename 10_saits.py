#!/usr/bin/env python3
"""
SAITS: Self-Attention Imputation for Time Series
Transformer encoder with value+mask embeddings, sinusoidal time encoding,
and [MASK] tokens on missing points.
"""

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_trajectory_data(mat_path, pixel_to_nm=35.0):
    """Load trajectory data and convert to nanometers."""
    data = scipy.io.loadmat(mat_path)

    # Find largest numeric array (automatic detection)
    candidates = {}
    for key, value in data.items():
        if key.startswith('__'):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            value = np.squeeze(value)
            if value.ndim == 1 and value.shape[0] >= 100:
                candidates[key] = value
            elif value.ndim == 2 and value.shape[0] >= 100 and value.shape[1] <= 3:
                candidates[key] = value

    if not candidates:
        raise ValueError("No suitable trajectory data found in MAT file")

    # Pick the largest array
    key, pos_pixels = max(candidates.items(), key=lambda x: x[1].size)
    pos_pixels = np.squeeze(pos_pixels)

    if pos_pixels.ndim == 2:
        # Take first dimension
        pos_pixels = pos_pixels[:, 0]

    pos_nm = pos_pixels * pixel_to_nm
    n = len(pos_nm)
    fs = 300.0
    t = np.arange(n) / fs

    return pos_nm, t, fs

def estimate_ou_parameters_psd(x, dt):
    """Estimate OU parameters from power spectral density."""
    from scipy.signal import welch
    from scipy.optimize import curve_fit

    x_centered = x - np.mean(x)
    fs = 1.0 / dt
    freqs, psd = welch(x_centered, fs=fs, nperseg=min(4096, len(x)//4),
                       scaling='density', detrend='constant')

    mask = (freqs > 0.1) & (freqs < fs/2)
    f_fit = freqs[mask]
    psd_fit = psd[mask]

    def lorentzian(f, S0, fc):
        return S0 / (1.0 + (f/fc)**2)

    fc_guess = f_fit[np.argmin(np.abs(psd_fit - psd_fit[0]/2))]
    S0_guess = psd_fit[0]

    try:
        popt, _ = curve_fit(lorentzian, f_fit, psd_fit, p0=[S0_guess, fc_guess],
                           bounds=([0, 0.1], [np.inf, fs/2]), maxfev=5000)
        S0_fit, fc_fit = popt
    except:
        S0_fit, fc_fit = S0_guess, fc_guess

    tau = 1.0 / (2.0 * np.pi * fc_fit)
    D = S0_fit * (2.0 * np.pi * fc_fit)

    return tau, D, fc_fit

def generate_realistic_gaps(n, fs, gap_prob=0.18, mean_gap_ms=25.0, max_gap_ms=200.0, seed=42):
    """Generate realistic missing data patterns."""
    rng = np.random.RandomState(seed)
    obs_mask = np.ones(n, dtype=bool)

    dt_ms = 1000.0 / fs
    mean_gap_samples = int(mean_gap_ms / dt_ms)
    max_gap_samples = int(max_gap_ms / dt_ms)

    target_missing = int(n * gap_prob)
    total_missing = 0

    while total_missing < target_missing:
        start_idx = rng.randint(0, n)
        if not obs_mask[start_idx]:
            continue

        gap_length = int(rng.exponential(mean_gap_samples))
        gap_length = min(gap_length, max_gap_samples)
        gap_length = max(gap_length, 1)

        end_idx = min(start_idx + gap_length, n)
        obs_mask[start_idx:end_idx] = False
        total_missing += (end_idx - start_idx)

    return obs_mask

def compute_metrics(y_true, y_pred, y_var=None, mask=None):
    """Compute comprehensive evaluation metrics."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_var is not None:
            y_var = y_var[mask]

    residual = y_true - y_pred
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

    if y_var is not None:
        y_std = np.sqrt(y_var)
        nll = 0.5 * (np.log(2 * np.pi * y_var) + residual**2 / y_var)
        metrics['nll'] = np.mean(nll)

        z_scores = np.abs(residual / (y_std + 1e-9))
        metrics['calib_1sigma'] = np.mean(z_scores <= 1.0)
        metrics['calib_2sigma'] = np.mean(z_scores <= 2.0)

    return metrics

# ============================================================================
# SAITS Model Architecture
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class SAITSImputer(nn.Module):
    """
    Self-Attention Imputation for Time Series (SAITS).

    Architecture:
    - Input embedding: value + mask concatenation
    - Positional encoding: sinusoidal
    - Transformer encoder: 2-4 layers, d_model=128
    - Output projection: reconstruct values
    """

    def __init__(self, d_input=1, d_model=128, n_heads=4, n_layers=3,
                 d_ff=256, dropout=0.1):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model

        # Input embedding: value + mask indicator
        self.value_embedding = nn.Linear(d_input, d_model // 2)
        self.mask_embedding = nn.Linear(d_input, d_model // 2)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model // 2))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_input)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: (batch, seq_len, d_input) - input values
        mask: (batch, seq_len, d_input) - 1 for observed, 0 for missing

        Returns:
        - reconstructed values: (batch, seq_len, d_input)
        """
        batch_size, seq_len, _ = x.shape

        # Embed values (only for observed points)
        value_emb = self.value_embedding(x * mask)  # (batch, seq_len, d_model//2)

        # Embed masks
        mask_emb = self.mask_embedding(mask)  # (batch, seq_len, d_model//2)

        # Replace missing values with mask token
        mask_token_expanded = self.mask_token.expand(batch_size, seq_len, -1)
        value_emb = torch.where(
            mask.bool(),
            value_emb,
            mask_token_expanded
        )

        # Concatenate value and mask embeddings
        combined_emb = torch.cat([value_emb, mask_emb], dim=-1)  # (batch, seq_len, d_model)

        # Add positional encoding
        combined_emb = self.pos_encoding(combined_emb)
        combined_emb = self.dropout(combined_emb)

        # Transformer encoder
        encoded = self.transformer_encoder(combined_emb)  # (batch, seq_len, d_model)

        # Project to output
        output = self.output_projection(encoded)  # (batch, seq_len, d_input)

        return output


class TrajectoryDataset(Dataset):
    """Dataset for trajectory imputation."""

    def __init__(self, values, obs_mask, sequence_length=512):
        self.values = values
        self.obs_mask = obs_mask
        self.sequence_length = sequence_length
        self.n_samples = len(values)

    def __len__(self):
        # Number of non-overlapping sequences
        return max(1, self.n_samples // self.sequence_length)

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        end_idx = min(start_idx + self.sequence_length, self.n_samples)

        # Pad if necessary
        if end_idx - start_idx < self.sequence_length:
            pad_length = self.sequence_length - (end_idx - start_idx)
            values = np.pad(self.values[start_idx:end_idx], (0, pad_length), mode='edge')
            obs_mask = np.pad(self.obs_mask[start_idx:end_idx], (0, pad_length), constant_values=0)
        else:
            values = self.values[start_idx:end_idx]
            obs_mask = self.obs_mask[start_idx:end_idx]

        values = values.reshape(-1, 1).astype(np.float32)
        obs_mask = obs_mask.reshape(-1, 1).astype(np.float32)

        return torch.from_numpy(values), torch.from_numpy(obs_mask)


def train_saits(model, train_loader, n_epochs, lr, device, data_mean, data_std, verbose=True):
    """Train SAITS model with normalized data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)

            # Normalize
            x_norm = (x_batch - data_mean) / data_std

            # Forward pass
            x_recon = model(x_norm, mask_batch)

            # Masked MSE loss (only on observed points during training)
            loss = F.mse_loss(x_recon * mask_batch, x_norm * mask_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    return model


def impute_with_saits(model, values, obs_mask, device, data_mean, data_std, sequence_length=512):
    """Use trained SAITS model for imputation."""
    model.eval()
    n_samples = len(values)
    reconstructed = np.zeros_like(values)

    with torch.no_grad():
        # Process in sequences
        for start_idx in range(0, n_samples, sequence_length):
            end_idx = min(start_idx + sequence_length, n_samples)
            seq_len = end_idx - start_idx

            # Prepare input
            x_seq = values[start_idx:end_idx].reshape(1, -1, 1).astype(np.float32)
            mask_seq = obs_mask[start_idx:end_idx].reshape(1, -1, 1).astype(np.float32)

            # Pad if necessary
            if seq_len < sequence_length:
                pad_length = sequence_length - seq_len
                x_seq = np.pad(x_seq, ((0, 0), (0, pad_length), (0, 0)), mode='edge')
                mask_seq = np.pad(mask_seq, ((0, 0), (0, pad_length), (0, 0)), constant_values=0)

            x_tensor = torch.from_numpy(x_seq).to(device)
            mask_tensor = torch.from_numpy(mask_seq).to(device)

            # Normalize
            x_norm = (x_tensor - data_mean) / data_std

            # Reconstruct
            x_recon = model(x_norm, mask_tensor)

            # Denormalize
            x_recon = x_recon * data_std + data_mean

            x_recon_np = x_recon.cpu().numpy()[0, :seq_len, 0]

            reconstructed[start_idx:end_idx] = x_recon_np

    return reconstructed


def upsample_trajectory(values_300hz, obs_mask_300hz, model, device,
                       data_mean, data_std, fs_in=300.0, fs_out=1000.0):
    """Upsample trajectory from 300 Hz to 1000 Hz."""
    n_in = len(values_300hz)

    # Create time arrays - ensure t_out doesn't exceed t_in range
    t_in = np.arange(n_in) / fs_in
    t_max = t_in[-1]
    t_out = np.arange(0, t_max, 1.0 / fs_out)

    # Ensure last point doesn't exceed
    if t_out[-1] > t_max:
        t_out = t_out[:-1]

    # Interpolate to output rate
    from scipy.interpolate import interp1d
    values_1000hz = interp1d(t_in, values_300hz, kind='linear',
                             fill_value='extrapolate')(t_out)
    obs_mask_1000hz = interp1d(t_in, obs_mask_300hz.astype(float),
                               kind='nearest', fill_value=0.0)(t_out).astype(bool)

    # Apply SAITS imputation
    imputed_1000hz = impute_with_saits(model, values_1000hz, obs_mask_1000hz, device,
                                       data_mean, data_std)

    return t_out, imputed_1000hz, obs_mask_1000hz


def plot_results(t, y_true, y_pred, obs_mask, eval_mask, method_name, filename=None):
    """Plot imputation results."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    # Plot ground truth
    ax.plot(t, y_true, 'o', color='gray', markersize=2, alpha=0.3, label='Ground truth')

    # Plot predictions
    ax.plot(t, y_pred, '-', color='#e74c3c', linewidth=1.5, alpha=0.8, label='Prediction')

    # Highlight observed points
    t_obs = t[obs_mask]
    y_obs = y_true[obs_mask]
    ax.plot(t_obs, y_obs, 'o', color='#3498db', markersize=3, alpha=0.6, label='Observed')

    # Highlight evaluation (missing) points
    if eval_mask is not None:
        t_miss = t[eval_mask]
        y_miss = y_true[eval_mask]
        ax.scatter(t_miss, y_miss, s=15, color='red', marker='x',
                  alpha=0.4, label='Missing (ground truth)', zorder=5)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (nm)', fontsize=11)
    ax.set_title(f'{method_name} - Imputation Results', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_zoom(t, y_true, y_pred, obs_mask, eval_mask, method_name,
              zoom_start=1.0, zoom_end=1.6, filename=None):
    """Plot zoomed-in view of imputation."""
    mask_time = (t >= zoom_start) & (t <= zoom_end)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    t_zoom = t[mask_time]
    y_true_zoom = y_true[mask_time]
    y_pred_zoom = y_pred[mask_time]
    obs_zoom = obs_mask[mask_time]

    # Ground truth
    ax.plot(t_zoom, y_true_zoom, 'o', color='gray', markersize=4, alpha=0.5, label='Ground truth')

    # Predictions
    ax.plot(t_zoom, y_pred_zoom, '-', color='#e74c3c', linewidth=2, alpha=0.9, label='Prediction')

    # Observed points
    ax.plot(t_zoom[obs_zoom], y_true_zoom[obs_zoom], 'o', color='#3498db',
           markersize=5, alpha=0.8, label='Observed', zorder=3)

    # Missing points
    if eval_mask is not None:
        eval_zoom = eval_mask[mask_time]
        if np.any(eval_zoom):
            ax.scatter(t_zoom[eval_zoom], y_true_zoom[eval_zoom], s=40,
                      color='red', marker='x', alpha=0.7,
                      label='Missing (ground truth)', zorder=5)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position (nm)', fontsize=11)
    ax.set_title(f'{method_name} (Zoom) - Imputation Results', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("=" * 80)
    print("SAITS: Self-Attention Imputation for Time Series")
    print("=" * 80)

    # Load data
    mat_path = '300fps_15k.mat'
    pos_nm, t_300hz, fs_300 = load_trajectory_data(mat_path, pixel_to_nm=35.0)
    print(f"\nLoaded trajectory: {len(pos_nm)} samples @ {fs_300} Hz")

    # Estimate OU parameters
    dt_300 = 1.0 / fs_300
    tau, D, fc = estimate_ou_parameters_psd(pos_nm, dt_300)
    print(f"OU parameters: τ={tau*1e3:.2f} ms, D={D:.1f} nm²/s, f_c={fc:.2f} Hz")

    # Generate realistic gaps for evaluation
    obs_mask_300 = generate_realistic_gaps(len(pos_nm), fs_300, gap_prob=0.35,
                                          mean_gap_ms=25.0, seed=42)
    obs_fraction = np.mean(obs_mask_300)
    print(f"Evaluation mask: {obs_fraction*100:.1f}% observed, {(1-obs_fraction)*100:.1f}% missing")

    # Compute normalization statistics from observed data
    data_mean = pos_nm[obs_mask_300].mean()
    data_std = pos_nm[obs_mask_300].std() + 1e-8
    print(f"Data normalization: mean={data_mean:.2f} nm, std={data_std:.2f} nm")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Create dataset and dataloader
    print("\n" + "=" * 80)
    print("Training SAITS Model")
    print("=" * 80)

    sequence_length = 512
    dataset = TrajectoryDataset(pos_nm, obs_mask_300, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Initialize model
    model = SAITSImputer(
        d_input=1,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Convert normalization params to tensors
    data_mean_tensor = torch.tensor(data_mean, dtype=torch.float32).reshape(1, 1, 1).to(device)
    data_std_tensor = torch.tensor(data_std, dtype=torch.float32).reshape(1, 1, 1).to(device)

    # Train model
    n_epochs = 100
    lr = 1e-3
    print(f"\nTraining for {n_epochs} epochs with lr={lr}...")
    model = train_saits(model, dataloader, n_epochs, lr, device,
                       data_mean_tensor, data_std_tensor, verbose=True)

    # Upsample and impute
    print("\n" + "=" * 80)
    print("Upsampling and Imputation")
    print("=" * 80)

    t_1000hz, imputed_1000hz, obs_mask_1000 = upsample_trajectory(
        pos_nm, obs_mask_300, model, device,
        data_mean_tensor, data_std_tensor, fs_in=300.0, fs_out=1000.0
    )
    print(f"Upsampled to {len(imputed_1000hz)} samples @ 1000 Hz")

    # Ground truth at 1000 Hz (interpolated for comparison)
    from scipy.interpolate import interp1d
    y_true_1000hz = interp1d(t_300hz, pos_nm, kind='cubic', fill_value='extrapolate')(t_1000hz)

    # Evaluate on missing points
    eval_mask = ~obs_mask_1000
    metrics = compute_metrics(y_true_1000hz, imputed_1000hz, mask=eval_mask)

    print("\n" + "=" * 80)
    print("Results on Missing Data Points")
    print("=" * 80)
    print(f"MAE:  {metrics['mae']:.4f} nm")
    print(f"RMSE: {metrics['rmse']:.4f} nm")
    print(f"R²:   {metrics['r2']:.4f}")

    # Plot results
    print("\nGenerating visualizations...")
    plot_results(t_1000hz, y_true_1000hz, imputed_1000hz, obs_mask_1000, eval_mask,
                'SAITS (Self-Attention Imputation)', filename='10_saits_result.png')
    plot_zoom(t_1000hz, y_true_1000hz, imputed_1000hz, obs_mask_1000, eval_mask,
             'SAITS (Self-Attention Imputation)', filename='10_saits_zoom.png')
    print("Saved: 10_saits_result.png, 10_saits_zoom.png")

    print("\n" + "=" * 80)
    print("SAITS Evaluation Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
