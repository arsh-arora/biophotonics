#!/usr/bin/env python3
"""
OU-Prior GP + Learned Residual (OU ⊕ RBF-short)
================================================
Hybrid Gaussian Process combining physics-based and data-driven kernels.

Kernel composition:
    k_total(t,t') = k_OU(t,t') + k_RBF(t,t')

Where:
- k_OU: Matérn-1/2 kernel capturing long-range OU physics
- k_RBF: RBF kernel with short lengthscale for residual corrections

This separates:
- Long-scale dynamics: trap relaxation, thermal fluctuations
- Short-scale quirks: drift, model discrepancies, high-frequency features

Benefits:
- Interpretable decomposition of physics vs residuals
- Better uncertainty quantification
- Prevents overfitting by constraining long-range behavior

References:
- Duvenaud et al. (2013). Structure Discovery in Nonparametric Regression
  through Compositional Kernel Search. ICML 2013.
- Rasmussen & Williams (2006). Gaussian Processes for Machine Learning.
"""

import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
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
# Hybrid GP Model (OU + RBF)
# ============================================================================

class HybridOURBFGP(ExactGP):
    """
    Gaussian Process with hybrid kernel: k_total = k_OU + k_RBF(short)

    Components:
    - k_OU: Matérn-1/2 kernel for long-range OU physics
    - k_RBF: RBF kernel with short lengthscale for residuals
    """

    def __init__(self, train_x, train_y, likelihood, tau_init=0.021):
        super(HybridOURBFGP, self).__init__(train_x, train_y, likelihood)

        # Mean function (constant)
        self.mean_module = ConstantMean()

        # OU kernel (Matérn-1/2) for long-range physics
        # Matérn-1/2: k(t,t') = σ² exp(-|t-t'|/λ)
        # For OU process: λ = τ (relaxation time)
        self.covar_module_ou = ScaleKernel(
            MaternKernel(nu=0.5, ard_num_dims=1)
        )

        # Initialize OU lengthscale to τ
        self.covar_module_ou.base_kernel.lengthscale = tau_init

        # Set prior favoring long lengthscales (physics)
        lengthscale_prior_ou = gpytorch.priors.NormalPrior(tau_init, tau_init * 0.5)
        self.covar_module_ou.base_kernel.register_prior(
            'lengthscale_prior', lengthscale_prior_ou, 'lengthscale'
        )

        # RBF kernel for short-range residuals
        self.covar_module_rbf = ScaleKernel(
            RBFKernel(ard_num_dims=1)
        )

        # Initialize RBF lengthscale to be short (e.g., 5-10 samples at 300 Hz)
        short_lengthscale = 5.0 / 300.0  # ~5 samples
        self.covar_module_rbf.base_kernel.lengthscale = short_lengthscale

        # Set prior favoring short lengthscales (residuals)
        lengthscale_prior_rbf = gpytorch.priors.NormalPrior(short_lengthscale, short_lengthscale * 0.5)
        self.covar_module_rbf.base_kernel.register_prior(
            'lengthscale_prior', lengthscale_prior_rbf, 'lengthscale'
        )

    def forward(self, x):
        """Forward pass: compute mean and covariance."""
        mean_x = self.mean_module(x)

        # Additive kernel: OU + RBF
        covar_ou = self.covar_module_ou(x)
        covar_rbf = self.covar_module_rbf(x)
        covar_x = covar_ou + covar_rbf

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_decomposed_covariance(self, x1, x2=None):
        """Get OU and RBF covariance components separately."""
        if x2 is None:
            x2 = x1

        covar_ou = self.covar_module_ou(x1, x2).evaluate()
        covar_rbf = self.covar_module_rbf(x1, x2).evaluate()

        return covar_ou, covar_rbf


# ============================================================================
# Training
# ============================================================================

def train_gp(model, likelihood, train_x, train_y, n_iter=100, lr=0.1, verbose=True):
    """Train GP model using Adam optimizer."""

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if verbose and (i + 1) % 10 == 0:
            with torch.no_grad():
                ou_lengthscale = model.covar_module_ou.base_kernel.lengthscale.item()
                rbf_lengthscale = model.covar_module_rbf.base_kernel.lengthscale.item()
                ou_scale = model.covar_module_ou.outputscale.item()
                rbf_scale = model.covar_module_rbf.outputscale.item()
                noise = likelihood.noise.item()

            print(f"Iter {i+1}/{n_iter}, Loss: {loss.item():.4f}, "
                  f"OU_λ: {ou_lengthscale*1e3:.2f}ms, RBF_λ: {rbf_lengthscale*1e3:.2f}ms, "
                  f"OU_σ²: {ou_scale:.2f}, RBF_σ²: {rbf_scale:.2f}, noise: {noise:.2f}")

    return model, likelihood


# ============================================================================
# Inference and Imputation
# ============================================================================

@torch.no_grad()
def predict_gp(model, likelihood, train_x, test_x):
    """Make predictions with trained GP."""
    model.eval()
    likelihood.eval()

    # Predict
    pred_dist = likelihood(model(test_x))

    mean = pred_dist.mean.cpu().numpy()
    var = pred_dist.variance.cpu().numpy()

    return mean, var


def impute_trajectory(model, likelihood, values, obs_mask, times, device):
    """Impute missing values using trained GP."""

    # Observed data
    obs_indices = np.where(obs_mask)[0]
    train_x = torch.tensor(times[obs_indices], dtype=torch.float32).unsqueeze(-1).to(device)
    train_y = torch.tensor(values[obs_indices], dtype=torch.float32).to(device)

    # All timepoints
    test_x = torch.tensor(times, dtype=torch.float32).unsqueeze(-1).to(device)

    # Predict
    mean, var = predict_gp(model, likelihood, train_x, test_x)

    return mean, var


def upsample_trajectory(values_300hz, obs_mask_300hz, times_300hz, model, likelihood,
                       device, fs_in=300.0, fs_out=1000.0):
    """Upsample trajectory from 300 Hz to 1000 Hz."""

    # Predict at 300 Hz
    mean_300hz, var_300hz = impute_trajectory(
        model, likelihood, values_300hz, obs_mask_300hz, times_300hz, device
    )

    # Create 1000 Hz timepoints
    t_max = times_300hz[-1]
    t_1000hz = np.arange(0, t_max, 1.0 / fs_out)
    if t_1000hz[-1] > t_max:
        t_1000hz = t_1000hz[:-1]

    # Predict at 1000 Hz
    # Use observed data from 300 Hz
    obs_indices = np.where(obs_mask_300hz)[0]
    train_x = torch.tensor(times_300hz[obs_indices], dtype=torch.float32).unsqueeze(-1).to(device)
    test_x = torch.tensor(t_1000hz, dtype=torch.float32).unsqueeze(-1).to(device)

    mean_1000hz, var_1000hz = predict_gp(model, likelihood, train_x, test_x)

    # Interpolate observation mask
    obs_mask_1000hz = interp1d(times_300hz, obs_mask_300hz.astype(float),
                               kind='nearest', fill_value=0.0)(t_1000hz).astype(bool)

    return t_1000hz, mean_1000hz, var_1000hz, obs_mask_1000hz


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


def plot_kernel_decomposition(model, times_sample, filename=None):
    """Plot OU and RBF kernel components."""
    model.eval()

    # Sample timepoints for visualization
    n_sample = min(500, len(times_sample))
    indices = np.linspace(0, len(times_sample) - 1, n_sample, dtype=int)
    t_sample = times_sample[indices]

    # Convert to tensor
    t_tensor = torch.tensor(t_sample, dtype=torch.float32).unsqueeze(-1)

    with torch.no_grad():
        # Get decomposed covariance at a reference point
        ref_idx = n_sample // 2
        t_ref = t_tensor[ref_idx:ref_idx+1]

        covar_ou, covar_rbf = model.get_decomposed_covariance(t_ref, t_tensor)
        covar_ou = covar_ou.squeeze().cpu().numpy()
        covar_rbf = covar_rbf.squeeze().cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # OU kernel
    axes[0].plot(t_sample - t_sample[ref_idx], covar_ou, 'b-', linewidth=2)
    axes[0].set_xlabel('Time lag (s)', fontsize=11)
    axes[0].set_ylabel('Covariance', fontsize=11)
    axes[0].set_title('OU Kernel (Long-range Physics)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # RBF kernel
    axes[1].plot(t_sample - t_sample[ref_idx], covar_rbf, 'r-', linewidth=2)
    axes[1].set_xlabel('Time lag (s)', fontsize=11)
    axes[1].set_ylabel('Covariance', fontsize=11)
    axes[1].set_title('RBF Kernel (Short-range Residuals)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Total kernel
    axes[2].plot(t_sample - t_sample[ref_idx], covar_ou + covar_rbf, 'g-', linewidth=2, label='Total')
    axes[2].plot(t_sample - t_sample[ref_idx], covar_ou, 'b--', linewidth=1, alpha=0.6, label='OU')
    axes[2].plot(t_sample - t_sample[ref_idx], covar_rbf, 'r--', linewidth=1, alpha=0.6, label='RBF')
    axes[2].set_xlabel('Time lag (s)', fontsize=11)
    axes[2].set_ylabel('Covariance', fontsize=11)
    axes[2].set_title('Total Kernel (OU + RBF)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

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
    print("OU-Prior GP + Learned Residual (OU ⊕ RBF-short)")
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

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device('cpu')  # GPyTorch has issues with MPS
        print("Using CPU (GPyTorch MPS compatibility)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Prepare training data (observed points only)
    obs_indices = np.where(obs_mask_300)[0]
    train_x = torch.tensor(t_300hz[obs_indices], dtype=torch.float32).unsqueeze(-1).to(device)
    train_y = torch.tensor(pos_nm[obs_indices], dtype=torch.float32).to(device)

    print(f"\nTraining data: {len(train_x)} observed points")

    # Initialize model
    print("\n" + "=" * 80)
    print("Training Hybrid GP Model (OU + RBF)")
    print("=" * 80)

    likelihood = GaussianLikelihood().to(device)
    model = HybridOURBFGP(train_x, train_y, likelihood, tau_init=tau).to(device)

    print(f"Initialized OU lengthscale: {tau*1e3:.2f} ms")
    print(f"Initialized RBF lengthscale: {5.0/fs_300*1e3:.2f} ms (~5 samples)")

    # Train
    n_iter = 100
    lr = 0.1
    print(f"\nTraining for {n_iter} iterations with lr={lr}...")
    model, likelihood = train_gp(model, likelihood, train_x, train_y,
                                 n_iter=n_iter, lr=lr, verbose=True)

    # Print final hyperparameters
    print("\n" + "=" * 80)
    print("Learned Hyperparameters")
    print("=" * 80)
    with torch.no_grad():
        ou_lengthscale = model.covar_module_ou.base_kernel.lengthscale.item()
        rbf_lengthscale = model.covar_module_rbf.base_kernel.lengthscale.item()
        ou_scale = model.covar_module_ou.outputscale.item()
        rbf_scale = model.covar_module_rbf.outputscale.item()
        noise = likelihood.noise.item()
        mean = model.mean_module.constant.item()

    print(f"OU kernel:")
    print(f"  Lengthscale (λ_OU): {ou_lengthscale*1e3:.2f} ms")
    print(f"  Output scale (σ²_OU): {ou_scale:.2f} nm²")
    print(f"RBF kernel:")
    print(f"  Lengthscale (λ_RBF): {rbf_lengthscale*1e3:.2f} ms")
    print(f"  Output scale (σ²_RBF): {rbf_scale:.2f} nm²")
    print(f"Likelihood noise: {noise:.2f} nm²")
    print(f"Mean: {mean:.2f} nm")

    # Visualize kernel decomposition
    print("\nGenerating kernel decomposition plot...")
    plot_kernel_decomposition(model, t_300hz, filename='13_gp_ou_rbf_kernels.png')
    print("Saved: 13_gp_ou_rbf_kernels.png")

    # Upsample and impute
    print("\n" + "=" * 80)
    print("Upsampling and Imputation")
    print("=" * 80)

    t_1000hz, imputed_1000hz, var_1000hz, obs_mask_1000 = upsample_trajectory(
        pos_nm, obs_mask_300, t_300hz, model, likelihood, device,
        fs_in=300.0, fs_out=1000.0
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
                obs_mask_1000, eval_mask, 'GP (OU + RBF)',
                filename='13_gp_ou_rbf_result.png')
    plot_zoom(t_1000hz, y_true_1000hz, imputed_1000hz, y_std_1000hz,
             obs_mask_1000, eval_mask, 'GP (OU + RBF)',
             filename='13_gp_ou_rbf_zoom.png')
    print("Saved: 13_gp_ou_rbf_result.png, 13_gp_ou_rbf_zoom.png")

    print("\n" + "=" * 80)
    print("Hybrid GP Evaluation Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
