#!/usr/bin/env python3
"""
CSDI: Conditional Score-based Diffusion for Imputation
========================================================
Diffusion model conditioned on observed data for trajectory imputation.

Architecture:
- 1D U-Net backbone with temporal convolutions
- Conditioning on (value, mask, time embeddings)
- Denoising score matching objective
- Multiple sampling (K=20) for uncertainty quantification

Key Features:
- Probabilistic imputation with calibrated uncertainty
- Handles irregular sampling patterns
- Multiple imputations → mean + variance

References:
- Tashiro et al. (2021). CSDI: Conditional Score-based Diffusion Models for
  Probabilistic Time Series Imputation. NeurIPS 2021.
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
# Diffusion Process
# ============================================================================

def get_beta_schedule(schedule='linear', n_timesteps=50, beta_start=1e-4, beta_end=0.02):
    """Get variance schedule for diffusion process."""
    if schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, n_timesteps)
    elif schedule == 'quadratic':
        betas = np.linspace(beta_start**0.5, beta_end**0.5, n_timesteps) ** 2
    elif schedule == 'cosine':
        # Cosine schedule from Improved DDPM
        s = 0.008
        steps = n_timesteps + 1
        x = np.linspace(0, n_timesteps, steps)
        alphas_cumprod = np.cos(((x / n_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': np.sqrt(1.0 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': np.sqrt(1.0 / alphas_cumprod - 1),
    }


# ============================================================================
# 1D U-Net Architecture for CSDI
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """Residual block with time and condition embeddings."""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        """
        x: (B, C, L)
        t_emb: (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample1D(nn.Module):
    """Downsampling layer."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsampling layer."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net for conditional diffusion model.

    Inputs:
    - Noisy trajectory: (B, 1, L)
    - Observed mask: (B, 1, L)
    - Observed values: (B, 1, L)
    - Diffusion timestep: (B,)

    Output:
    - Predicted noise: (B, 1, L)
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=32,
                 time_emb_dim=32, channel_mults=(1, 2, 4, 8), dropout=0.1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )

        # Initial projection
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (downsampling)
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels_block = base_channels * mult

            self.downs.append(nn.ModuleList([
                ResidualBlock1D(now_channels, out_channels_block, time_emb_dim * 4, dropout),
                ResidualBlock1D(out_channels_block, out_channels_block, time_emb_dim * 4, dropout),
                Downsample1D(out_channels_block) if i < len(channel_mults) - 1 else nn.Identity()
            ]))

            now_channels = out_channels_block
            channels.append(now_channels)

        # Bottleneck
        self.mid1 = ResidualBlock1D(now_channels, now_channels, time_emb_dim * 4, dropout)
        self.mid2 = ResidualBlock1D(now_channels, now_channels, time_emb_dim * 4, dropout)

        # Decoder (upsampling)
        self.ups = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_channels_block = base_channels * mult

            self.ups.append(nn.ModuleList([
                ResidualBlock1D(now_channels + channels.pop(), out_channels_block,
                               time_emb_dim * 4, dropout),
                ResidualBlock1D(out_channels_block, out_channels_block,
                               time_emb_dim * 4, dropout),
                Upsample1D(out_channels_block) if i < len(channel_mults) - 1 else nn.Identity()
            ]))

            now_channels = out_channels_block

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv1d(now_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_noisy, mask, x_obs, timestep):
        """
        x_noisy: (B, 1, L) - noisy trajectory
        mask: (B, 1, L) - observation mask
        x_obs: (B, 1, L) - observed values
        timestep: (B,) - diffusion timestep
        """
        # Concatenate inputs as conditioning
        x = torch.cat([x_noisy, mask, x_obs], dim=1)  # (B, 3, L)

        # Time embedding
        t_emb = self.time_mlp(timestep)  # (B, time_emb_dim * 4)

        # Initial conv
        x = self.conv_in(x)

        # Encoder
        skip_connections = []
        for res1, res2, downsample in self.downs:
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Decoder
        for res1, res2, upsample in self.ups:
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = upsample(x)

        # Output
        x = self.conv_out(x)

        return x


# ============================================================================
# CSDI Model and Training
# ============================================================================

class CSDIImputer(nn.Module):
    """CSDI: Conditional Score-based Diffusion Imputation."""

    def __init__(self, n_timesteps=50, base_channels=32, device='cpu'):
        super().__init__()

        self.device = device
        self.n_timesteps = n_timesteps

        # Diffusion schedule
        schedule = get_beta_schedule('cosine', n_timesteps)
        for key, val in schedule.items():
            self.register_buffer(key, torch.tensor(val, dtype=torch.float32))

        # U-Net denoising model
        self.model = UNet1D(
            in_channels=3,  # noisy + mask + observed
            out_channels=1,
            base_channels=base_channels,
            time_emb_dim=32,
            channel_mults=(1, 2, 4, 8),
            dropout=0.1
        )

    def q_sample(self, x_start, t, noise):
        """Forward diffusion: add noise to x_start at timestep t."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, mask, t, noise=None):
        """Compute denoising loss."""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)

        # Condition on observed values
        x_obs = x_start * mask

        # Predict noise
        noise_pred = self.model(x_noisy, mask, x_obs, t)

        # Loss only on missing points
        loss = F.mse_loss(noise_pred * (1 - mask), noise * (1 - mask))

        return loss

    @torch.no_grad()
    def p_sample(self, x, mask, x_obs, t, t_index):
        """Single reverse diffusion step."""
        betas_t = self.betas[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1)

        # Predict noise
        noise_pred = self.model(x, mask, x_obs, t)

        # Mean of reverse distribution
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            # No noise at final step
            return model_mean
        else:
            # Add noise
            noise = torch.randn_like(x)
            # Use posterior variance
            alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].reshape(-1, 1, 1)
            posterior_variance = betas_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - self.alphas_cumprod[t].reshape(-1, 1, 1))
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, mask, x_obs):
        """Full reverse diffusion loop."""
        device = self.device
        b = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Replace observed values
        x = x * (1 - mask) + x_obs * mask

        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, mask, x_obs, t, i)

            # Keep observed values fixed
            x = x * (1 - mask) + x_obs * mask

        return x

    @torch.no_grad()
    def sample(self, mask, x_obs, n_samples=20):
        """Generate multiple imputation samples."""
        samples = []
        shape = x_obs.shape

        for _ in range(n_samples):
            sample = self.p_sample_loop(shape, mask, x_obs)
            samples.append(sample)

        samples = torch.stack(samples, dim=0)  # (n_samples, B, 1, L)

        # Compute mean and variance
        mean = samples.mean(dim=0)
        var = samples.var(dim=0)

        return mean, var, samples


# ============================================================================
# Dataset and Training
# ============================================================================

class TrajectoryDataset(Dataset):
    """Dataset for diffusion model training."""

    def __init__(self, values, obs_mask, sequence_length=256):
        self.values = values
        self.obs_mask = obs_mask
        self.sequence_length = sequence_length
        self.n_samples = len(values)

    def __len__(self):
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

        values = values.reshape(1, -1).astype(np.float32)
        obs_mask = obs_mask.reshape(1, -1).astype(np.float32)

        return torch.from_numpy(values), torch.from_numpy(obs_mask)


def train_csdi(model, train_loader, n_epochs, lr, device, verbose=True):
    """Train CSDI model."""
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

            # Random timestep
            t = torch.randint(0, model.n_timesteps, (x_batch.shape[0],), device=device).long()

            # Compute loss
            loss = model.p_losses(x_batch, mask_batch, t)

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


def impute_with_csdi(model, values, obs_mask, device, data_mean, data_std, n_samples=20):
    """Use CSDI for imputation with uncertainty quantification."""
    model.eval()

    # Normalize
    values_norm = (values - data_mean) / data_std

    # Pad to multiple of 16 (for 4 levels of 2x downsampling)
    orig_len = len(values_norm)
    pad_to = ((orig_len + 15) // 16) * 16
    pad_len = pad_to - orig_len

    if pad_len > 0:
        values_norm = np.pad(values_norm, (0, pad_len), mode='edge')
        obs_mask = np.pad(obs_mask, (0, pad_len), constant_values=0)

    # Prepare tensors
    x_obs = torch.from_numpy(values_norm.reshape(1, 1, -1).astype(np.float32)).to(device)
    mask = torch.from_numpy(obs_mask.reshape(1, 1, -1).astype(np.float32)).to(device)

    # Sample multiple imputations
    mean, var, samples = model.sample(mask, x_obs, n_samples=n_samples)

    # Denormalize
    mean = mean * data_std + data_mean
    var = var * (data_std ** 2)

    # Convert to numpy and remove padding
    mean_np = mean.cpu().numpy()[0, 0, :orig_len]
    var_np = var.cpu().numpy()[0, 0, :orig_len]

    return mean_np, var_np


def upsample_trajectory(values_300hz, obs_mask_300hz, model, device,
                       data_mean, data_std, fs_in=300.0, fs_out=1000.0, n_samples=20):
    """Upsample trajectory from 300 Hz to 1000 Hz."""
    n_in = len(values_300hz)

    # Create time arrays
    t_in = np.arange(n_in) / fs_in
    t_max = t_in[-1]
    t_out = np.arange(0, t_max, 1.0 / fs_out)

    if t_out[-1] > t_max:
        t_out = t_out[:-1]

    # Interpolate to output rate
    from scipy.interpolate import interp1d
    values_1000hz = interp1d(t_in, values_300hz, kind='linear',
                             fill_value='extrapolate')(t_out)
    obs_mask_1000hz = interp1d(t_in, obs_mask_300hz.astype(float),
                               kind='nearest', fill_value=0.0)(t_out).astype(bool)

    # Apply CSDI imputation
    imputed_1000hz, var_1000hz = impute_with_csdi(
        model, values_1000hz, obs_mask_1000hz, device, data_mean, data_std, n_samples
    )

    return t_out, imputed_1000hz, var_1000hz, obs_mask_1000hz


# ============================================================================
# Visualization
# ============================================================================

def plot_results(t, y_true, y_pred, y_std, obs_mask, eval_mask, method_name, filename=None):
    """Plot imputation results with uncertainty."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    # Plot ground truth
    ax.plot(t, y_true, 'o', color='gray', markersize=2, alpha=0.3, label='Ground truth')

    # Plot predictions with uncertainty
    ax.plot(t, y_pred, '-', color='#e74c3c', linewidth=1.5, alpha=0.8, label='Prediction')

    if y_std is not None:
        ax.fill_between(t, y_pred - 2*y_std, y_pred + 2*y_std,
                        color='#e74c3c', alpha=0.2, label='±2σ uncertainty')

    # Highlight observed points
    t_obs = t[obs_mask]
    y_obs = y_true[obs_mask]
    ax.plot(t_obs, y_obs, 'o', color='#3498db', markersize=3, alpha=0.6, label='Observed')

    # Highlight missing points
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


def plot_zoom(t, y_true, y_pred, y_std, obs_mask, eval_mask, method_name,
              zoom_start=1.0, zoom_end=1.6, filename=None):
    """Plot zoomed view with uncertainty."""
    mask_time = (t >= zoom_start) & (t <= zoom_end)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    t_zoom = t[mask_time]
    y_true_zoom = y_true[mask_time]
    y_pred_zoom = y_pred[mask_time]
    y_std_zoom = y_std[mask_time] if y_std is not None else None
    obs_zoom = obs_mask[mask_time]

    # Ground truth
    ax.plot(t_zoom, y_true_zoom, 'o', color='gray', markersize=4, alpha=0.5, label='Ground truth')

    # Predictions
    ax.plot(t_zoom, y_pred_zoom, '-', color='#e74c3c', linewidth=2, alpha=0.9, label='Prediction')

    if y_std_zoom is not None:
        ax.fill_between(t_zoom, y_pred_zoom - 2*y_std_zoom, y_pred_zoom + 2*y_std_zoom,
                       color='#e74c3c', alpha=0.2, label='±2σ uncertainty')

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
    print("CSDI: Conditional Score-based Diffusion for Imputation")
    print("=" * 80)

    # Load data
    mat_path = '300fps_15k.mat'
    pos_nm, t_300hz, fs_300 = load_trajectory_data(mat_path, pixel_to_nm=35.0)
    print(f"\nLoaded trajectory: {len(pos_nm)} samples @ {fs_300} Hz")

    # Estimate OU parameters
    dt_300 = 1.0 / fs_300
    tau, D, fc = estimate_ou_parameters_psd(pos_nm, dt_300)
    print(f"OU parameters: τ={tau*1e3:.2f} ms, D={D:.1f} nm²/s, f_c={fc:.2f} Hz")

    # Generate realistic gaps
    obs_mask_300 = generate_realistic_gaps(len(pos_nm), fs_300, gap_prob=0.35,
                                          mean_gap_ms=25.0, seed=42)
    obs_fraction = np.mean(obs_mask_300)
    print(f"Evaluation mask: {obs_fraction*100:.1f}% observed, {(1-obs_fraction)*100:.1f}% missing")

    # Normalization
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

    # Create dataset
    print("\n" + "=" * 80)
    print("Training CSDI Model")
    print("=" * 80)

    sequence_length = 256
    dataset = TrajectoryDataset(
        (pos_nm - data_mean) / data_std,  # Normalize
        obs_mask_300,
        sequence_length=sequence_length
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize model
    model = CSDIImputer(
        n_timesteps=50,
        base_channels=32,
        device=device
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    n_epochs = 100
    lr = 1e-3
    print(f"\nTraining for {n_epochs} epochs with lr={lr}...")
    model = train_csdi(model, dataloader, n_epochs, lr, device, verbose=True)

    # Upsample and impute
    print("\n" + "=" * 80)
    print("Upsampling and Imputation (K=20 samples)")
    print("=" * 80)

    t_1000hz, imputed_1000hz, var_1000hz, obs_mask_1000 = upsample_trajectory(
        pos_nm, obs_mask_300, model, device, data_mean, data_std,
        fs_in=300.0, fs_out=1000.0, n_samples=20
    )
    print(f"Upsampled to {len(imputed_1000hz)} samples @ 1000 Hz")

    # Ground truth at 1000 Hz
    from scipy.interpolate import interp1d
    y_true_1000hz = interp1d(t_300hz, pos_nm, kind='cubic', fill_value='extrapolate')(t_1000hz)

    # Evaluate on missing points
    eval_mask = ~obs_mask_1000
    metrics = compute_metrics(y_true_1000hz, imputed_1000hz, var_1000hz, mask=eval_mask)

    print("\n" + "=" * 80)
    print("Results on Missing Data Points")
    print("=" * 80)
    print(f"MAE:  {metrics['mae']:.4f} nm")
    print(f"RMSE: {metrics['rmse']:.4f} nm")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"NLL:  {metrics['nll']:.4f}")
    print(f"Calibration @ 1σ: {metrics['calib_1sigma']*100:.1f}%")
    print(f"Calibration @ 2σ: {metrics['calib_2sigma']*100:.1f}%")

    # Plot results
    print("\nGenerating visualizations...")
    y_std_1000hz = np.sqrt(var_1000hz)
    plot_results(t_1000hz, y_true_1000hz, imputed_1000hz, y_std_1000hz,
                obs_mask_1000, eval_mask, 'CSDI (Diffusion)', filename='11_csdi_result.png')
    plot_zoom(t_1000hz, y_true_1000hz, imputed_1000hz, y_std_1000hz,
             obs_mask_1000, eval_mask, 'CSDI (Diffusion)', filename='11_csdi_zoom.png')
    print("Saved: 11_csdi_result.png, 11_csdi_zoom.png")

    print("\n" + "=" * 80)
    print("CSDI Evaluation Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
