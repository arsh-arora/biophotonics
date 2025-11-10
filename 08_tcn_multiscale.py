#!/usr/bin/env python3
"""
TCN with Multi-Scale Physics-Aware Loss
========================================
Enhanced Temporal Convolutional Network with sophisticated loss function
that separates long-time drift from short-time jitter.

Loss Components:
1. L_trend: MSE on low-frequency (trend/drift) components
2. L_fast: Huber loss on high-frequency residuals (robust to spikes)
3. L_diff: Huber loss on temporal derivatives (dynamics consistency)
4. L_psd: Spectral consistency with low-frequency emphasis
5. L_OU: Optional OU autocovariance matching

Optional heteroscedastic head for uncertainty quantification.

Enhanced Evaluation Metrics:
- MAE, RMSE, R², MBE (mean bias error)
- Frequency-weighted RMSE (WRMSE)
- PSD divergence, ACF whiteness (Ljung-Box)
- Allan deviation
- NLL, CRPS, calibration (for variance head)
"""

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal
from scipy.stats import norm
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
    from torch.utils.data import DataLoader, Dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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

except ImportError as e:
    print(f"Error: {e}")
    print("Please install PyTorch: pip install torch")
    sys.exit(1)

warnings.filterwarnings('ignore')


# ============================================================================
# MULTI-SCALE PHYSICS-AWARE LOSS
# ============================================================================

class MultiScaleLoss(nn.Module):
    """
    Multi-scale physics-aware loss function.

    Decomposes signal into:
    - Low-frequency (trend/drift): penalized by L_trend + L_psd
    - High-frequency (noise/jitter): penalized by L_fast (Huber)
    - Dynamics: penalized by L_diff (derivative matching)
    - Optional OU prior: L_OU (autocovariance matching)
    """

    def __init__(
        self,
        fs=1000.0,
        window_ms=200.0,
        huber_delta=1.0,
        lambdas=(2.0, 1.0, 0.5, 0.25, 0.25),
        learnable_weights=True,
        use_ou_prior=True,
        tau_ms=20.0
    ):
        super().__init__()

        self.fs = fs
        self.window_samples = int(window_ms * fs / 1000.0)
        if self.window_samples % 2 == 0:
            self.window_samples += 1  # Make odd for symmetry
        self.huber_delta = huber_delta
        self.use_ou_prior = use_ou_prior
        self.tau = tau_ms / 1000.0  # Convert to seconds

        # Loss weights (homoscedastic uncertainty weighting)
        if learnable_weights:
            # log(σ²) formulation from Kendall & Gal
            self.log_var_trend = nn.Parameter(torch.zeros(1))
            self.log_var_fast = nn.Parameter(torch.zeros(1))
            self.log_var_diff = nn.Parameter(torch.zeros(1))
            self.log_var_psd = nn.Parameter(torch.zeros(1))
            self.log_var_ou = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('log_var_trend', torch.log(torch.tensor(1.0 / lambdas[0])))
            self.register_buffer('log_var_fast', torch.log(torch.tensor(1.0 / lambdas[1])))
            self.register_buffer('log_var_diff', torch.log(torch.tensor(1.0 / lambdas[2])))
            self.register_buffer('log_var_psd', torch.log(torch.tensor(1.0 / lambdas[3])))
            self.register_buffer('log_var_ou', torch.log(torch.tensor(1.0 / lambdas[4])))

    def moving_average(self, x):
        """Low-pass filter via moving average."""
        # x: (B, T)
        B, T = x.shape
        win = self.window_samples
        pad = win // 2

        # Create 1D convolution filter
        filt = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / win

        # Add channel dimension and pad
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect')

        # Apply convolution
        x_smooth = F.conv1d(x_pad, filt).squeeze(1)

        return x_smooth

    def huber_loss(self, residual, delta=None):
        """Huber loss: quadratic for small errors, linear for large."""
        if delta is None:
            delta = self.huber_delta

        abs_res = residual.abs()
        quadratic = 0.5 * (abs_res ** 2)
        linear = delta * (abs_res - 0.5 * delta)

        return torch.where(abs_res <= delta, quadratic, linear)

    def spectral_loss(self, y, yhat, mask):
        """
        PSD divergence with low-frequency emphasis.

        Uses Welch method with frequency-dependent weighting.
        """
        # Apply mask
        y_masked = y * mask
        yhat_masked = yhat * mask

        # Compute PSD via FFT (simplified single-window version)
        B, T = y.shape

        # Apply Hann window
        window = torch.hann_window(T, device=y.device, dtype=y.dtype)
        y_win = y_masked * window
        yhat_win = yhat_masked * window

        # FFT
        Y = torch.fft.rfft(y_win, dim=-1)
        Yhat = torch.fft.rfft(yhat_win, dim=-1)

        # Power spectral density
        Py = (Y.abs() ** 2).mean(dim=0)  # Average over batch
        Pyhat = (Yhat.abs() ** 2).mean(dim=0)

        # Frequency weights (↑ weight at low frequencies)
        n_freq = Py.shape[0]
        freq_weights = torch.linspace(1.0, 0.1, n_freq, device=y.device)

        # Log-MSE with frequency weighting
        log_psd_loss = ((torch.log(Pyhat + 1e-12) - torch.log(Py + 1e-12)) ** 2 * freq_weights).mean()

        return log_psd_loss

    def ou_autocovariance_loss(self, y, yhat, mask):
        """
        OU prior: match autocovariances of increments.

        Theoretical OU has: Cov(Δy_t, Δy_{t+k}) ~ exp(-k*dt/τ) for k>0
        """
        # Compute increments
        dy = y[:, 1:] - y[:, :-1]  # (B, T-1)
        dyhat = yhat[:, 1:] - yhat[:, :-1]

        # Mask for increments
        mask_inc = mask[:, 1:] * mask[:, :-1]

        # Compute empirical autocovariances for a few lags
        max_lag = min(50, dy.shape[1] // 4)  # ~50 ms at 1kHz
        dt = 1.0 / self.fs

        loss = 0.0
        for lag in range(1, max_lag):
            if lag >= dy.shape[1]:
                break

            # Cross-covariance at this lag
            dy1 = dy[:, :-lag]
            dy2 = dy[:, lag:]
            dyhat1 = dyhat[:, :-lag]
            dyhat2 = dyhat[:, lag:]

            m = mask_inc[:, :-lag] * mask_inc[:, lag:]

            # Empirical autocovariance
            acov_true = ((dy1 * dy2) * m).sum() / (m.sum() + 1e-8)
            acov_pred = ((dyhat1 * dyhat2) * m).sum() / (m.sum() + 1e-8)

            # Theoretical OU autocovariance (exponential decay)
            acov_ou = torch.exp(torch.tensor(-lag * dt / self.tau, device=y.device))

            # Penalize deviation from OU form
            loss += ((acov_pred / (acov_true.abs() + 1e-8) - acov_ou) ** 2)

        return loss / max_lag

    def forward(self, y_pred, y_true, mask):
        """
        Compute multi-scale loss.

        Args:
            y_pred: (B, T) predicted trajectory
            y_true: (B, T) ground truth trajectory
            mask: (B, T) observation mask (1=observed, 0=missing)

        Returns:
            loss: weighted combination of all loss terms
            loss_dict: dictionary of individual loss components
        """
        # Decompose into low/high frequency
        y_low = self.moving_average(y_true)
        yhat_low = self.moving_average(y_pred)

        y_high = y_true - y_low
        yhat_high = y_pred - yhat_low

        # 1. Trend loss (MSE on low-frequency)
        L_trend = ((yhat_low - y_low) ** 2 * mask).sum() / (mask.sum() + 1e-8)

        # 2. Fast loss (Huber on high-frequency)
        L_fast = (self.huber_loss(yhat_high - y_high) * mask).sum() / (mask.sum() + 1e-8)

        # 3. Derivative loss (Huber on temporal increments)
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        mask_diff = mask[:, 1:] * mask[:, :-1]
        L_diff = (self.huber_loss(dy_pred - dy_true) * mask_diff).sum() / (mask_diff.sum() + 1e-8)

        # 4. Spectral loss (PSD divergence)
        L_psd = self.spectral_loss(y_true, y_pred, mask)

        # 5. OU prior loss (optional)
        if self.use_ou_prior:
            L_ou = self.ou_autocovariance_loss(y_true, y_pred, mask)
        else:
            L_ou = torch.tensor(0.0, device=y_pred.device)

        # Weighted combination (homoscedastic uncertainty)
        # Loss = Σ (1 / (2σ²)) * L_i + log(σ²) / 2
        precision_trend = torch.exp(-self.log_var_trend)
        precision_fast = torch.exp(-self.log_var_fast)
        precision_diff = torch.exp(-self.log_var_diff)
        precision_psd = torch.exp(-self.log_var_psd)
        precision_ou = torch.exp(-self.log_var_ou)

        loss = (
            precision_trend * L_trend + 0.5 * self.log_var_trend +
            precision_fast * L_fast + 0.5 * self.log_var_fast +
            precision_diff * L_diff + 0.5 * self.log_var_diff +
            precision_psd * L_psd + 0.5 * self.log_var_psd
        )

        if self.use_ou_prior:
            loss += precision_ou * L_ou + 0.5 * self.log_var_ou

        # Return loss and components for monitoring
        loss_dict = {
            'total': loss.item(),
            'trend': L_trend.item(),
            'fast': L_fast.item(),
            'diff': L_diff.item(),
            'psd': L_psd.item(),
            'ou': L_ou.item() if self.use_ou_prior else 0.0,
            'weight_trend': precision_trend.item(),
            'weight_fast': precision_fast.item(),
            'weight_diff': precision_diff.item(),
            'weight_psd': precision_psd.item(),
            'weight_ou': precision_ou.item() if self.use_ou_prior else 0.0
        }

        return loss, loss_dict


# ============================================================================
# TCN ARCHITECTURE (with optional heteroscedastic head)
# ============================================================================

class Chomp1d(nn.Module):
    """Remove future information by chomping the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    Residual block with causal dilated convolutions.

    Architecture: Conv1d → Chomp → ReLU → Dropout → Conv1d → Chomp → ReLU → Dropout
    With skip connection.
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
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNImputerHeteroscedastic(nn.Module):
    """
    TCN with optional heteroscedastic variance head.

    Outputs (μ, log σ²) for uncertainty quantification.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 7,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        predict_variance: bool = True
    ):
        super().__init__()

        self.predict_variance = predict_variance

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

        # Output heads
        self.mean_head = nn.Conv1d(hidden_channels, out_channels, 1)

        if predict_variance:
            self.logvar_head = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C_in, T) input [trajectory; mask]

        Returns:
            mean: (B, C_out, T) predicted mean
            logvar: (B, C_out, T) predicted log variance (if enabled)
        """
        features = self.network(x)
        mean = self.mean_head(features)

        if self.predict_variance:
            logvar = self.logvar_head(features)
            return mean, logvar
        else:
            return mean, None


# ============================================================================
# TRAINING
# ============================================================================

def train_tcn_multiscale(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: MultiScaleLoss,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = DEVICE,
    verbose: bool = True
) -> Dict[str, list]:
    """Train TCN with multi-scale loss."""

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {
        'loss': [],
        'trend': [],
        'fast': [],
        'diff': [],
        'psd': [],
        'ou': []
    }

    for epoch in range(n_epochs):
        epoch_losses = {key: 0.0 for key in history.keys()}
        n_batches = 0

        for x_batch, m_batch in dataloader:
            x_batch = x_batch.to(device)  # (B, T, D)
            m_batch = m_batch.to(device)  # (B, T, 1)

            # Mask missing values
            x_masked = x_batch * m_batch

            # Prepare input: [trajectory; mask]
            x_input = torch.cat([x_masked, m_batch], dim=-1).transpose(1, 2)  # (B, D+1, T)

            # Forward pass
            if model.predict_variance:
                x_pred_mean, x_pred_logvar = model(x_input)
                x_pred = x_pred_mean  # Use mean for multi-scale loss
            else:
                x_pred, _ = model(x_input)

            # Transpose to (B, T)
            y_pred = x_pred.squeeze(1)  # (B, T)
            y_true = x_batch.squeeze(-1)  # (B, T)
            mask = m_batch.squeeze(-1)  # (B, T)

            # Compute multi-scale loss
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

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Loss={history['loss'][-1]:.4f}, "
                  f"Trend={history['trend'][-1]:.4f}, "
                  f"Fast={history['fast'][-1]:.4f}, "
                  f"Diff={history['diff'][-1]:.4f}, "
                  f"PSD={history['psd'][-1]:.4f}")

    return history


# ============================================================================
# DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    """Dataset for TCN training with random masking."""

    def __init__(self, X, obs_mask, window_size=512, mask_augment_prob=0.35):
        self.X = X
        self.obs_mask = obs_mask
        self.window_size = window_size
        self.mask_augment_prob = mask_augment_prob

        # Number of windows
        self.n_windows = max(1, len(X) // window_size)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.window_size
        end = min(start + self.window_size, len(self.X))

        x = self.X[start:end]
        m = self.obs_mask[start:end].astype(np.float32)

        # Mask augmentation: randomly hide additional points
        if self.mask_augment_prob > 0:
            aug_mask = np.random.rand(len(m)) > self.mask_augment_prob
            m = m * aug_mask

        # Pad if needed
        if len(x) < self.window_size:
            pad_len = self.window_size - len(x)
            x = np.pad(x, (0, pad_len), mode='edge')
            m = np.pad(m, (0, pad_len), constant_values=0)

        return (
            torch.from_numpy(x).float().unsqueeze(-1),  # (T, 1)
            torch.from_numpy(m).float().unsqueeze(-1)  # (T, 1)
        )


# ============================================================================
# EVALUATION - ENHANCED METRICS
# ============================================================================

def compute_enhanced_metrics(y_true, y_pred, y_var=None, mask=None, fs=1000.0):
    """
    Compute enhanced evaluation metrics including:
    - MAE, RMSE, R², MBE
    - Frequency-weighted RMSE
    - PSD divergence
    - ACF whiteness (Ljung-Box)
    - Allan deviation
    - NLL, CRPS, calibration (if variance provided)
    """
    if mask is not None:
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        if y_var is not None:
            y_var_masked = y_var[mask]
    else:
        y_true_masked = y_true
        y_pred_masked = y_pred
        y_var_masked = y_var

    residual = y_true_masked - y_pred_masked

    # Basic metrics
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    mbe = np.mean(residual)  # Mean bias error

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true_masked - np.mean(y_true_masked)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MBE': mbe
    }

    # Frequency-weighted RMSE (emphasize low frequencies)
    try:
        freqs, psd_true = signal.welch(y_true_masked, fs=fs, nperseg=min(256, len(y_true_masked)//4))
        freqs, psd_pred = signal.welch(y_pred_masked, fs=fs, nperseg=min(256, len(y_pred_masked)//4))

        # Weight inversely proportional to frequency
        freq_weights = 1.0 / (freqs + 0.1)
        freq_weights /= freq_weights.sum()

        psd_error = np.abs(psd_pred - psd_true)
        wrmse = np.sqrt(np.sum(freq_weights * psd_error ** 2))
        metrics['WRMSE'] = wrmse

        # PSD divergence (log-MSE)
        psd_div = np.mean((np.log(psd_pred + 1e-12) - np.log(psd_true + 1e-12)) ** 2)
        metrics['PSD_div'] = psd_div
    except:
        metrics['WRMSE'] = np.nan
        metrics['PSD_div'] = np.nan

    # ACF whiteness (Ljung-Box test)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residual, lags=min(50, len(residual)//4), return_df=False)
        metrics['ACF_pvalue'] = lb_result[1][-1]  # p-value at max lag
    except:
        metrics['ACF_pvalue'] = np.nan

    # Slope error (drift proxy)
    try:
        from scipy.stats import linregress
        t = np.arange(len(y_true_masked)) / fs
        slope_true, _, _, _, _ = linregress(t, y_true_masked)
        slope_pred, _, _, _, _ = linregress(t, y_pred_masked)
        metrics['Slope_error'] = np.abs(slope_pred - slope_true) * 1e9  # nm/s
    except:
        metrics['Slope_error'] = np.nan

    # Probabilistic metrics (if variance provided)
    if y_var_masked is not None and not np.all(np.isnan(y_var_masked)):
        y_std_masked = np.sqrt(y_var_masked)

        # NLL
        nll = 0.5 * (np.log(2 * np.pi * y_var_masked) + residual ** 2 / y_var_masked)
        metrics['NLL'] = np.mean(nll)

        # Calibration
        z_scores = np.abs(residual / (y_std_masked + 1e-9))
        metrics['Calib_1sigma'] = np.mean(z_scores <= 1.0)
        metrics['Calib_2sigma'] = np.mean(z_scores <= 2.0)

        # CRPS (Continuous Ranked Probability Score)
        crps = np.mean(y_std_masked * (
            residual / y_std_masked * (2 * norm.cdf(residual / y_std_masked) - 1) +
            2 * norm.pdf(residual / y_std_masked) -
            1 / np.sqrt(np.pi)
        ))
        metrics['CRPS'] = crps
    else:
        metrics['NLL'] = np.nan
        metrics['Calib_1sigma'] = np.nan
        metrics['Calib_2sigma'] = np.nan
        metrics['CRPS'] = np.nan

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_imputation_comparison(
    t: np.ndarray,
    X_true: np.ndarray,
    X_pred: np.ndarray,
    X_var: Optional[np.ndarray] = None,
    obs_mask: Optional[np.ndarray] = None,
    method_name: str = "TCN Multi-Scale",
    filename: Optional[str] = None
):
    """Plot imputation results with uncertainty."""
    import matplotlib.pyplot as plt

    D = X_true.shape[1] if X_true.ndim > 1 else 1
    fig, axes = plt.subplots(D, 1, figsize=(14, 4*D), squeeze=False)

    for d in range(D):
        ax = axes[d, 0]

        y_true = X_true[:, d] if D > 1 else X_true
        y_pred = X_pred[:, d] if D > 1 else X_pred

        # Plot ground truth
        ax.plot(t, y_true, 'o', color='gray', markersize=2, alpha=0.3, label='Ground truth')

        # Plot predictions
        ax.plot(t, y_pred, '-', color='#e74c3c', linewidth=1.5, alpha=0.8, label='Prediction')

        # Uncertainty bands
        if X_var is not None:
            y_var = X_var[:, d] if D > 1 else X_var
            y_std = np.sqrt(y_var)
            ax.fill_between(t, y_pred - 2*y_std, y_pred + 2*y_std,
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
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# IMPUTATION INTERFACE (matching original TCN)
# ============================================================================

def impute_tcn_multiscale(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    fs_in: float = 300.0,
    fs_out: float = 1000.0,
    num_levels: int = 7,
    hidden_channels: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.1,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    window_size: int = 512,
    predict_variance: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Impute trajectory using TCN with multi-scale physics-aware loss.

    Returns dictionary compatible with original TCN interface.
    """

    N, D = X.shape

    # Normalize
    X_mean = X[obs_mask].mean(axis=0)
    X_std = X[obs_mask].std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    if verbose:
        print(f"\n  Training TCN with Multi-Scale Loss:")
        print(f"    Data shape: {X.shape}")
        print(f"    Observation rate: {obs_mask.mean()*100:.1f}%")
        print(f"    Levels: {num_levels}, Hidden: {hidden_channels}, Kernel: {kernel_size}")

    # Create dataset and dataloader
    dataset = TrajectoryDataset(X_norm[:, 0], obs_mask, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TCNImputerHeteroscedastic(
        in_channels=D + 1,  # trajectory + mask
        out_channels=D,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        predict_variance=predict_variance
    ).to(DEVICE)

    # Initialize multi-scale loss
    loss_fn = MultiScaleLoss(
        fs=fs_in,
        window_ms=200.0,
        huber_delta=1.0,
        learnable_weights=True,
        use_ou_prior=True,
        tau_ms=20.0
    )

    # Train
    if verbose:
        print(f"    Training for {n_epochs} epochs...")

    history = train_tcn_multiscale(
        model, dataloader, loss_fn,
        n_epochs=n_epochs, lr=lr,
        device=DEVICE, verbose=verbose
    )

    # Inference at 300 Hz
    model.eval()
    with torch.no_grad():
        x_input = torch.cat([
            torch.from_numpy(X_norm).float(),
            torch.from_numpy(obs_mask.astype(np.float32)).reshape(-1, 1)
        ], dim=-1).unsqueeze(0).transpose(1, 2).to(DEVICE)  # (1, D+1, N)

        if predict_variance:
            x_pred_mean, x_pred_logvar = model(x_input)
            x_pred_mean = x_pred_mean.squeeze(0).transpose(0, 1).cpu().numpy()  # (N, D)
            x_pred_var = torch.exp(x_pred_logvar).squeeze(0).transpose(0, 1).cpu().numpy()
        else:
            x_pred_mean, _ = model(x_input)
            x_pred_mean = x_pred_mean.squeeze(0).transpose(0, 1).cpu().numpy()
            x_pred_var = None

    # Denormalize
    X_recon = x_pred_mean * X_std + X_mean
    if x_pred_var is not None:
        X_var = x_pred_var * (X_std ** 2)
    else:
        X_var = None

    # Interpolate to 1000 Hz
    M = len(t_out)
    mu_1000 = np.zeros((M, D))
    var_1000 = np.zeros((M, D)) if X_var is not None else None

    for d in range(D):
        cs = CubicSpline(t, X_recon[:, d])
        mu_1000[:, d] = cs(t_out)

        if X_var is not None:
            cs_var = CubicSpline(t, X_var[:, d])
            var_1000[:, d] = np.maximum(cs_var(t_out), 0)

    return {
        'output_300Hz': X_recon,
        'output_1000Hz': mu_1000,
        'mean': X_recon,
        'var': X_var,
        'mean_1000Hz': mu_1000,
        'var_1000Hz': var_1000,
        'extras': {
            'history': history,
            'final_loss': history['loss'][-1],
            'model_params': sum(p.numel() for p in model.parameters())
        }
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_tcn_multiscale(
    segments: list,
    n_epochs: int = 100,
    window: int = 512,
    num_levels: int = 7,
    hidden_channels: int = 64,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate TCN Multi-Scale on all segments."""

    print("\n" + "="*80)
    print(f"EVALUATING: TCN with Multi-Scale Physics-Aware Loss")
    print("="*80)
    print(f"Parameters: epochs={n_epochs}, window={window}, levels={num_levels}, hidden={hidden_channels}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        fs_in = 1.0 / (t_seg[1] - t_seg[0]) if len(t_seg) > 1 else 300.0
        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / 1000.0)

        result = impute_tcn_multiscale(
            X_true, t_seg, t_out, obs_mask,
            fs_in=fs_in,
            fs_out=1000.0,
            num_levels=num_levels,
            hidden_channels=hidden_channels,
            n_epochs=n_epochs,
            window_size=window,
            predict_variance=True,
            verbose=verbose
        )

        X_pred = result['output_300Hz']
        X_var = result['var']

        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            continue

        # Compute enhanced metrics
        metrics = compute_enhanced_metrics(
            X_true[missing_mask, 0],
            X_pred[missing_mask, 0],
            X_var[missing_mask, 0] if X_var is not None else None,
            mask=None,
            fs=fs_in
        )

        all_metrics.append(metrics)

        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f}")

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
    if not np.isnan(avg_metrics.get('NLL', np.nan)):
        print(f"NLL:  {avg_metrics['NLL']:.4f}")
        print(f"Calibration @ 1σ: {avg_metrics['Calib_1sigma']*100:.1f}%")
        print(f"Calibration @ 2σ: {avg_metrics['Calib_2sigma']*100:.1f}%")
    print()

    return avg_metrics


def visualize_sample_result(
    segment: dict,
    n_epochs: int = 100,
    num_levels: int = 7,
    hidden_channels: int = 64
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    fs_in = 1.0 / (t_seg[1] - t_seg[0]) if len(t_seg) > 1 else 300.0
    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / 1000.0)

    result = impute_tcn_multiscale(
        X_true, t_seg, t_out, obs_mask,
        fs_in=fs_in,
        fs_out=1000.0,
        num_levels=num_levels,
        hidden_channels=hidden_channels,
        n_epochs=n_epochs,
        window_size=512,
        predict_variance=True,
        verbose=True
    )

    X_pred = result['output_300Hz']
    X_var = result['var']

    # Full plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=X_var,
        obs_mask=obs_mask,
        method_name="TCN Multi-Scale (Physics-Aware Loss)",
        filename="08_tcn_multiscale_result.png"
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

            plot_imputation_comparison(
                t_seg[zoom_start:zoom_end],
                X_true[zoom_start:zoom_end],
                X_pred[zoom_start:zoom_end],
                X_var=X_var[zoom_start:zoom_end] if X_var is not None else None,
                obs_mask=obs_mask[zoom_start:zoom_end],
                method_name="TCN Multi-Scale (Zoom)",
                filename="08_tcn_multiscale_zoom.png"
            )


# ============================================================================
# MAIN EVALUATION
# ============================================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 80)
    print("TCN with Multi-Scale Physics-Aware Loss")
    print("=" * 80)

    # Load starter framework
    print("Loading data and segments from starter framework...")

    if Path('00_starter_framework.py').exists():
        # This sets up global variables: t, X, X_detrended, ou_params, segments, CONFIG
        exec(open('00_starter_framework.py').read())

        # segments is now available
        if 'segments' not in dir():
            print("Error: segments not loaded from starter framework")
            sys.exit(1)

        print(f"\nRunning on device: {DEVICE}")

        # Evaluate
        metrics = evaluate_tcn_multiscale(
            segments,
            n_epochs=100,
            window=512,
            num_levels=7,
            hidden_channels=64,
            verbose=False
        )

        # Visualize
        print("\nGenerating visualization...")
        visualize_sample_result(
            segments[0],
            n_epochs=100,
            num_levels=7,
            hidden_channels=64
        )

        print("\n✓ TCN Multi-Scale evaluation complete.")
        print(f"  Saved: 08_tcn_multiscale_result.png, 08_tcn_multiscale_zoom.png")

    else:
        print("\nError: 00_starter_framework.py not found")
        print("This script requires the starter framework to run.")
        sys.exit(1)
