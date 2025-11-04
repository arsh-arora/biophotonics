#!/usr/bin/env python3
"""
Neural SDE with OU Drift Prior
================================
Hybrid physics-informed approach combining Ornstein-Uhlenbeck process
with learned neural corrections.

SDE form:
    dX = (-(X/τ) + g_θ(X,t)) dt + sqrt(2D) dW

Where:
- -(X/τ): Physics-based OU drift (trap restoring force)
- g_θ(X,t): Learned neural correction for model discrepancies
- sqrt(2D): Diffusion coefficient from thermal fluctuations

Training:
- Initialize τ, D from OU parameter estimation
- Train g_θ via masked reconstruction
- Regularize ||g_θ|| to keep corrections small
- Generate K=10 posterior samples for uncertainty

References:
- Kidger et al. (2020). Neural Controlled Differential Equations for
  Irregular Time Series. NeurIPS 2020.
- Raissi (2018). Deep Hidden Physics Models. arXiv:1801.06637.
"""

import numpy as np
import scipy.io
from scipy.interpolate import CubicSpline, interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchsde
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_trajectory_data(mat_path, pixel_to_nm=35.0):
    """Load trajectory data and convert to nanometers."""
    data = scipy.io.loadmat(mat_path)

    # Find largest numeric array
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

    key, pos_pixels = max(candidates.items(), key=lambda x: x[1].size)
    pos_pixels = np.squeeze(pos_pixels)

    if pos_pixels.ndim == 2:
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
# Neural SDE Architecture
# ============================================================================

class DriftNet(nn.Module):
    """
    Drift function combining OU physics with neural correction.

    dX/dt = -(X/τ) + g_θ(X,t)

    Where:
    - -(X/τ): OU restoring force
    - g_θ(X,t): Small MLP correction
    """

    def __init__(self, state_dim, hidden_dim=64, tau_init=0.021, trainable_tau=False):
        super().__init__()

        self.state_dim = state_dim
        self.trainable_tau = trainable_tau

        # OU time constant (τ)
        if trainable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init, dtype=torch.float32)))
        else:
            self.register_buffer('log_tau', torch.log(torch.tensor(tau_init, dtype=torch.float32)))

        # Neural correction network g_θ(X,t)
        self.correction_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Initialize correction network with small weights
        for layer in self.correction_net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, t, x):
        """
        Compute drift: dX/dt = -(X/τ) + g_θ(X,t)

        Args:
            t: scalar time
            x: (batch_size, state_dim) state

        Returns:
            drift: (batch_size, state_dim)
        """
        tau = torch.exp(self.log_tau).clamp(min=1e-4)

        # OU restoring force
        ou_drift = -x / tau

        # Neural correction
        t_expanded = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        x_t = torch.cat([x, t_expanded], dim=1)
        correction = self.correction_net(x_t)

        return ou_drift + correction

    def get_correction(self, t, x):
        """Get neural correction term separately."""
        t_expanded = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        x_t = torch.cat([x, t_expanded], dim=1)
        return self.correction_net(x_t)


class DiffusionNet(nn.Module):
    """
    Diagonal diffusion: σ(t,X) = sqrt(2D) * I

    Where D is the thermal diffusion coefficient.
    """

    def __init__(self, state_dim, D_init=18000.0, trainable_D=False):
        super().__init__()

        self.state_dim = state_dim
        self.trainable_D = trainable_D

        # Diffusion coefficient (D)
        if trainable_D:
            self.log_D = nn.Parameter(torch.log(torch.tensor(D_init, dtype=torch.float32)))
        else:
            self.register_buffer('log_D', torch.log(torch.tensor(D_init, dtype=torch.float32)))

    def forward(self, t, x):
        """
        Compute diffusion: σ = sqrt(2D) * I

        Args:
            t: scalar time
            x: (batch_size, state_dim) state

        Returns:
            diffusion: (batch_size, state_dim, brownian_dim)
        """
        D = torch.exp(self.log_D).clamp(min=1e-4)
        sigma = torch.sqrt(2.0 * D)

        # Diagonal diffusion matrix
        batch_size = x.size(0)
        diffusion = torch.diag_embed(sigma.expand(batch_size, self.state_dim))

        return diffusion


class NeuralSDE(nn.Module):
    """
    Neural SDE with OU prior.

    dX = [-(X/τ) + g_θ(X,t)] dt + sqrt(2D) dW
    """

    def __init__(self, state_dim, hidden_dim=64, tau_init=0.021, D_init=18000.0,
                 trainable_tau=False, trainable_D=False, sde_type='ito', noise_type='diagonal'):
        super().__init__()

        self.state_dim = state_dim
        self.sde_type = sde_type
        self.noise_type = noise_type

        # Drift and diffusion
        self.drift = DriftNet(state_dim, hidden_dim, tau_init, trainable_tau)
        self.diffusion = DiffusionNet(state_dim, D_init, trainable_D)

    def f(self, t, x):
        """Drift term."""
        return self.drift(t, x)

    def g(self, t, x):
        """Diffusion term."""
        return self.diffusion(t, x)


# ============================================================================
# Training
# ============================================================================

class TrajectoryDataset(Dataset):
    """Dataset for Neural SDE training."""

    def __init__(self, values, times, obs_mask, segment_length=1.0, fs=300.0):
        self.values = values
        self.times = times
        self.obs_mask = obs_mask
        self.segment_length = segment_length
        self.fs = fs

        # Create segments
        n_samples = int(segment_length * fs)
        self.n_segments = len(values) // n_samples
        self.n_samples = n_samples

    def __len__(self):
        return self.n_segments

    def __getitem__(self, idx):
        start_idx = idx * self.n_samples
        end_idx = start_idx + self.n_samples

        values = self.values[start_idx:end_idx].astype(np.float32)
        times = self.times[start_idx:end_idx].astype(np.float32)
        obs_mask = self.obs_mask[start_idx:end_idx].astype(np.float32)

        return (
            torch.from_numpy(values).unsqueeze(-1),  # (seq_len, 1)
            torch.from_numpy(times),  # (seq_len,)
            torch.from_numpy(obs_mask)  # (seq_len,)
        )


def train_neural_sde(model, train_loader, n_epochs, lr, device,
                     correction_reg=0.01, verbose=True):
    """Train Neural SDE with masked reconstruction."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_reg_loss = 0.0
        n_batches = 0

        for batch_values, batch_times, batch_obs_mask in train_loader:
            batch_values = batch_values.to(device)  # (batch, seq_len, 1)
            batch_times = batch_times.to(device)  # (batch, seq_len)
            batch_obs_mask = batch_obs_mask.to(device)  # (batch, seq_len)

            batch_size = batch_values.size(0)

            # Initial condition (first observed point)
            x0 = batch_values[:, 0, :]  # (batch, 1)

            # Simulate SDE
            ts = batch_times[0, :]  # Use first batch's times (all same)

            # Use Euler-Maruyama method for efficiency
            dt = (ts[1] - ts[0]).item()
            xs = [x0]

            for i in range(len(ts) - 1):
                t = ts[i]
                x = xs[-1]

                # Drift
                drift = model.f(t, x)

                # Diffusion
                diffusion = model.g(t, x).squeeze(-1)  # (batch, 1)

                # Brownian increment
                dW = torch.randn_like(x) * np.sqrt(dt)

                # Euler-Maruyama step
                x_next = x + drift * dt + diffusion * dW
                xs.append(x_next)

            # Stack trajectory
            xs_simulated = torch.stack(xs, dim=1)  # (batch, seq_len, 1)

            # Masked reconstruction loss (only on observed points)
            obs_mask_expanded = batch_obs_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            recon_loss = F.mse_loss(
                xs_simulated * obs_mask_expanded,
                batch_values * obs_mask_expanded
            )

            # Regularization: penalize correction network
            correction_samples = []
            for i in range(0, len(ts), max(1, len(ts) // 10)):  # Sample 10 time points
                t = ts[i]
                x = xs[i]
                correction = model.drift.get_correction(t, x)
                correction_samples.append(correction)

            correction_samples = torch.stack(correction_samples, dim=0)
            reg_loss = torch.mean(correction_samples ** 2)

            # Total loss
            loss = recon_loss + correction_reg * reg_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_reg_loss += reg_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_reg = epoch_reg_loss / n_batches

        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f} "
                  f"(Recon: {avg_recon:.6f}, Reg: {avg_reg:.6f})")

    return model


# ============================================================================
# Inference and Imputation
# ============================================================================

@torch.no_grad()
def simulate_sde_trajectory(model, x0, times, device, n_samples=10):
    """
    Simulate multiple SDE trajectories for uncertainty quantification.

    Args:
        model: Neural SDE model
        x0: (1,) initial condition
        times: (seq_len,) time points
        device: torch device
        n_samples: number of posterior samples

    Returns:
        samples: (n_samples, seq_len, 1) trajectories
    """
    model.eval()

    samples = []

    for _ in range(n_samples):
        x0_batch = x0.unsqueeze(0).unsqueeze(-1).to(device)  # (1, 1)
        ts = times.to(device)
        dt = (ts[1] - ts[0]).item()

        xs = [x0_batch]

        for i in range(len(ts) - 1):
            t = ts[i]
            x = xs[-1]

            # Drift
            drift = model.f(t, x)

            # Diffusion
            diffusion = model.g(t, x).squeeze(-1)

            # Brownian increment
            dW = torch.randn_like(x) * np.sqrt(dt)

            # Euler-Maruyama step
            x_next = x + drift * dt + diffusion * dW
            xs.append(x_next)

        trajectory = torch.stack(xs, dim=1).squeeze(0)  # (seq_len, 1)
        samples.append(trajectory)

    samples = torch.stack(samples, dim=0)  # (n_samples, seq_len, 1)

    return samples


def impute_with_neural_sde(model, values, obs_mask, times, device,
                           data_mean, data_std, n_samples=10):
    """Use Neural SDE for imputation with uncertainty."""

    # Normalize
    values_norm = (values - data_mean) / data_std

    # Initial condition (first observed point)
    obs_indices = np.where(obs_mask)[0]
    if len(obs_indices) == 0:
        raise ValueError("No observed points in sequence")

    x0 = torch.tensor(values_norm[obs_indices[0]], dtype=torch.float32)
    times_tensor = torch.from_numpy(times.astype(np.float32))

    # Simulate multiple trajectories
    samples = simulate_sde_trajectory(model, x0, times_tensor, device, n_samples)

    # Compute mean and variance
    samples_np = samples.cpu().numpy()  # (n_samples, seq_len, 1)
    mean = samples_np.mean(axis=0).squeeze(-1)  # (seq_len,)
    var = samples_np.var(axis=0).squeeze(-1)  # (seq_len,)

    # Denormalize
    mean = mean * data_std + data_mean
    var = var * (data_std ** 2)

    # Replace observed values with ground truth
    mean[obs_mask] = values[obs_mask]
    var[obs_mask] = 0.0

    return mean, var


def upsample_trajectory(values_300hz, obs_mask_300hz, times_300hz, model, device,
                       data_mean, data_std, fs_in=300.0, fs_out=1000.0, n_samples=10):
    """Upsample trajectory from 300 Hz to 1000 Hz."""

    # Impute at 300 Hz
    mean_300hz, var_300hz = impute_with_neural_sde(
        model, values_300hz, obs_mask_300hz, times_300hz,
        device, data_mean, data_std, n_samples
    )

    # Upsample to 1000 Hz using cubic spline
    t_in = times_300hz
    t_max = t_in[-1]
    t_out = np.arange(0, t_max, 1.0 / fs_out)

    if t_out[-1] > t_max:
        t_out = t_out[:-1]

    # Spline interpolation for mean
    spline = CubicSpline(t_in, mean_300hz)
    mean_1000hz = spline(t_out)

    # Linear interpolation for variance
    var_1000hz = interp1d(t_in, var_300hz, kind='linear',
                          fill_value='extrapolate')(t_out)
    var_1000hz = np.maximum(var_1000hz, 0.0)

    # Interpolate observation mask
    obs_mask_1000hz = interp1d(t_in, obs_mask_300hz.astype(float),
                               kind='nearest', fill_value=0.0)(t_out).astype(bool)

    return t_out, mean_1000hz, var_1000hz, obs_mask_1000hz


# ============================================================================
# Visualization
# ============================================================================

def plot_results(t, y_true, y_pred, y_std, obs_mask, eval_mask, method_name, filename=None):
    """Plot imputation results with uncertainty."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    # Ground truth
    ax.plot(t, y_true, 'o', color='gray', markersize=2, alpha=0.3, label='Ground truth')

    # Predictions with uncertainty
    ax.plot(t, y_pred, '-', color='#e74c3c', linewidth=1.5, alpha=0.8, label='Prediction')

    if y_std is not None:
        ax.fill_between(t, y_pred - 2*y_std, y_pred + 2*y_std,
                        color='#e74c3c', alpha=0.2, label='±2σ uncertainty')

    # Observed points
    t_obs = t[obs_mask]
    y_obs = y_true[obs_mask]
    ax.plot(t_obs, y_obs, 'o', color='#3498db', markersize=3, alpha=0.6, label='Observed')

    # Missing points
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
    print("Neural SDE with OU Drift Prior")
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
    print("Training Neural SDE Model")
    print("=" * 80)

    # Normalize data for training
    pos_nm_norm = (pos_nm - data_mean) / data_std

    dataset = TrajectoryDataset(
        pos_nm_norm,
        t_300hz,
        obs_mask_300,
        segment_length=1.0,  # 1 second segments
        fs=fs_300
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Initialize model with OU parameters
    model = NeuralSDE(
        state_dim=1,
        hidden_dim=64,
        tau_init=tau,
        D_init=D,
        trainable_tau=False,  # Keep τ fixed
        trainable_D=False,    # Keep D fixed
        sde_type='ito',
        noise_type='diagonal'
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initialized with τ={tau*1e3:.2f} ms, D={D:.1f} nm²/s")

    # Train
    n_epochs = 80
    lr = 1e-3
    correction_reg = 0.01
    print(f"\nTraining for {n_epochs} epochs with lr={lr}, correction_reg={correction_reg}...")
    model = train_neural_sde(model, dataloader, n_epochs, lr, device,
                            correction_reg=correction_reg, verbose=True)

    # Upsample and impute
    print("\n" + "=" * 80)
    print("Upsampling and Imputation (K=10 samples)")
    print("=" * 80)

    t_1000hz, imputed_1000hz, var_1000hz, obs_mask_1000 = upsample_trajectory(
        pos_nm, obs_mask_300, t_300hz, model, device, data_mean, data_std,
        fs_in=300.0, fs_out=1000.0, n_samples=10
    )
    print(f"Upsampled to {len(imputed_1000hz)} samples @ 1000 Hz")

    # Ground truth at 1000 Hz
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
                obs_mask_1000, eval_mask, 'Neural SDE (OU + MLP)',
                filename='12_neural_sde_result.png')
    plot_zoom(t_1000hz, y_true_1000hz, imputed_1000hz, y_std_1000hz,
             obs_mask_1000, eval_mask, 'Neural SDE (OU + MLP)',
             filename='12_neural_sde_zoom.png')
    print("Saved: 12_neural_sde_result.png, 12_neural_sde_zoom.png")

    print("\n" + "=" * 80)
    print("Neural SDE Evaluation Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
