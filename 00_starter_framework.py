#!/usr/bin/env python3
"""
Research-Grade Framework for Optical Tweezers Trajectory Imputation
====================================================================
300 Hz → 1000 Hz upsampling with physically-informed and data-driven methods.

Bead: 3μm diameter, 532nm laser, 20mW power
Conversion: 1 pixel = 35 nm
Sampling: 300 fps → 1000 fps target
"""

import os
import sys
import math
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
import numpy as np
from scipy.io import loadmat
from scipy import signal, interpolate, stats
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
sns.set_style("whitegrid")

# Try importing PyTorch (optional for neural methods)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠ PyTorch not available. Neural methods will be disabled.")

# Reproducibility
RNG = np.random.default_rng(42)
np.set_printoptions(suppress=True, precision=6, linewidth=120)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Physical and experimental parameters."""
    # Physical constants
    pixel_to_nm: float = 35.0  # nm per pixel
    bead_diameter_um: float = 3.0
    laser_wavelength_nm: float = 532.0
    laser_power_mw: float = 20.0
    temperature_K: float = 298.0  # Room temperature
    viscosity_water_pas: float = 8.9e-4  # Pa·s at 25°C

    # Sampling parameters
    fs_in: float = 300.0  # Hz - native sampling rate
    fs_out: float = 1000.0  # Hz - target sampling rate

    # Data path
    data_path: str = "300fps_15k.mat"

    # Evaluation parameters
    n_test_segments: int = 5  # Number of independent test segments
    test_segment_length_s: float = 2.0  # Length of each test segment
    gap_prob: float = 0.18  # Probability of gap start
    mean_gap_duration_ms: float = 25.0  # Mean gap duration
    max_gap_duration_ms: float = 200.0  # Maximum gap duration

    # Derived properties
    def __post_init__(self):
        self.dt_in = 1.0 / self.fs_in
        self.dt_out = 1.0 / self.fs_out
        self.bead_radius_m = (self.bead_diameter_um * 1e-6) / 2.0
        # Stokes drag coefficient: γ = 6πηr
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.gamma = 6 * np.pi * self.viscosity_water_pas * self.bead_radius_m

    def get_thermal_diffusion_coeff(self) -> float:
        """Einstein relation: D = k_B T / γ"""
        return self.k_B * self.temperature_K / self.gamma

    def get_corner_frequency_estimate(self, trap_stiffness_pN_per_nm: float = 0.1) -> float:
        """Estimate trap corner frequency: f_c = κ / (2πγ)"""
        kappa = trap_stiffness_pN_per_nm * 1e-3  # Convert to N/m
        return kappa / (2 * np.pi * self.gamma)

CONFIG = ExperimentConfig()

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_trajectory_data(path: str = None, config: ExperimentConfig = CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess trajectory data from MAT file.

    Returns:
        t: (N,) time array in seconds
        X: (N, D) position array in nanometers (D=1 or 2 for x or x,y)
    """
    if path is None:
        path = config.data_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"Loading trajectory from: {path}")
    mat = loadmat(path)

    # Identify the trajectory data
    # Strategy: find largest numeric array with reasonable shape
    candidates = {}
    for key, value in mat.items():
        if key.startswith('__'):  # Skip MATLAB metadata
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
    key, raw_data = max(candidates.items(), key=lambda x: x[1].size)
    print(f"  → Using variable '{key}' with shape {raw_data.shape}")

    # Ensure shape (N, D)
    X_pixels = np.atleast_2d(raw_data)
    if X_pixels.shape[0] < X_pixels.shape[1]:
        X_pixels = X_pixels.T

    N, D = X_pixels.shape

    # Convert to nanometers
    X_nm = X_pixels * config.pixel_to_nm

    # Create time array
    t = np.arange(N) * config.dt_in

    # Basic data validation
    if np.any(~np.isfinite(X_nm)):
        print(f"  ⚠ Warning: {np.sum(~np.isfinite(X_nm))} non-finite values detected, replacing with linear interpolation")
        for d in range(D):
            mask = np.isfinite(X_nm[:, d])
            if not np.all(mask):
                X_nm[~mask, d] = np.interp(t[~mask], t[mask], X_nm[mask, d])

    print(f"  ✓ Loaded {N} samples ({t[-1]:.2f}s) with {D} dimension(s)")
    print(f"  ✓ Position range: [{X_nm.min():.1f}, {X_nm.max():.1f}] nm")

    mean_pos = X_nm.mean(axis=0)
    std_pos = X_nm.std(axis=0)
    if D == 1:
        print(f"  ✓ Mean position: {mean_pos[0]:.2f} nm")
        print(f"  ✓ Std deviation: {std_pos[0]:.2f} nm")
    else:
        print(f"  ✓ Mean position: {mean_pos} nm")
        print(f"  ✓ Std deviation: {std_pos} nm")

    return t, X_nm


def detrend_trajectory(X: np.ndarray, method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove slow drift from trajectory.

    Args:
        X: (N, D) trajectory
        method: 'linear', 'constant', or 'polynomial'

    Returns:
        X_detrended: (N, D) detrended trajectory
        trend: (N, D) removed trend
    """
    N, D = X.shape
    X_detrended = np.zeros_like(X)
    trend = np.zeros_like(X)

    for d in range(D):
        if method == 'linear':
            X_detrended[:, d] = signal.detrend(X[:, d], type='linear')
            trend[:, d] = X[:, d] - X_detrended[:, d]
        elif method == 'constant':
            X_detrended[:, d] = X[:, d] - np.mean(X[:, d])
            trend[:, d] = np.mean(X[:, d])
        else:
            raise ValueError(f"Unknown detrend method: {method}")

    return X_detrended, trend


# ============================================================================
# POWER SPECTRAL DENSITY ANALYSIS
# ============================================================================

def compute_psd(x: np.ndarray, fs: float, nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Returns:
        freqs: (M,) frequency array in Hz
        psd: (M,) power spectral density in nm²/Hz
    """
    if nperseg is None:
        nperseg = min(len(x) // 8, 2048)

    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg,
                              window='hann', scaling='density',
                              detrend='linear')
    return freqs, psd


def fit_lorentzian_psd(freqs: np.ndarray, psd: np.ndarray,
                        f_min: float = 1.0, f_max: float = 100.0) -> Dict[str, float]:
    """
    Fit Lorentzian PSD model for trapped Brownian particle:
    S(f) = D / (π² * (f² + f_c²))

    where D is diffusion coefficient and f_c is corner frequency.

    Returns dict with: D (nm²/s), f_c (Hz), tau (ms), and fit quality R²
    """
    # Restrict to fitting range
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_fit = freqs[mask]
    S_fit = psd[mask]

    # Log-space linear fit: log(S) ≈ log(D/π²) - 2*log(f) for f >> f_c
    # For full fit, use nonlinear least squares
    from scipy.optimize import curve_fit

    def lorentzian(f, D, f_c):
        return D / (np.pi**2 * (f**2 + f_c**2))

    # Initial guess: f_c ~ 10 Hz, D from low-freq plateau
    p0 = [S_fit[0] * np.pi**2 * f_fit[0]**2, 10.0]

    try:
        popt, pcov = curve_fit(lorentzian, f_fit, S_fit, p0=p0,
                               bounds=([0, 0.1], [np.inf, 200.0]),
                               maxfev=5000)
        D, f_c = popt

        # Compute R²
        S_pred = lorentzian(f_fit, D, f_c)
        ss_res = np.sum((S_fit - S_pred)**2)
        ss_tot = np.sum((S_fit - S_fit.mean())**2)
        r2 = 1 - ss_res / ss_tot

        tau_ms = 1000.0 / (2 * np.pi * f_c)  # Relaxation time in ms

        return {
            'D': D,  # nm²/s
            'f_c': f_c,  # Hz
            'tau': tau_ms,  # ms
            'R2': r2,
            'stderr_D': np.sqrt(pcov[0, 0]) if pcov is not None else np.nan,
            'stderr_fc': np.sqrt(pcov[1, 1]) if pcov is not None else np.nan
        }
    except Exception as e:
        print(f"  ⚠ PSD fit failed: {e}")
        return {'D': np.nan, 'f_c': np.nan, 'tau': np.nan, 'R2': 0.0}


def estimate_ou_parameters(x: np.ndarray, dt: float, method: str = 'psd') -> Dict[str, float]:
    """
    Estimate Ornstein-Uhlenbeck process parameters from trajectory.

    Model: dx = -x/τ dt + sqrt(2D) dW

    Args:
        x: (N,) trajectory (should be mean-centered)
        dt: sampling interval
        method: 'psd' (PSD fitting), 'acf' (autocorrelation), or 'mle' (maximum likelihood)

    Returns:
        dict with: D (diffusion), tau (relaxation time), sigma_m (measurement noise)
    """
    fs = 1.0 / dt
    x = x - np.mean(x)  # Ensure zero mean

    if method == 'psd':
        freqs, psd = compute_psd(x, fs)
        params = fit_lorentzian_psd(freqs, psd)
        D = params['D']
        tau = params['tau'] * 1e-3  # Convert ms to s

        # Estimate measurement noise from high-frequency plateau
        high_freq_mask = freqs > 100.0
        if np.any(high_freq_mask):
            noise_floor = np.median(psd[high_freq_mask])
            sigma_m = np.sqrt(noise_floor * fs / 2)  # Convert PSD to variance
        else:
            sigma_m = 0.0

        return {'D': D, 'tau': tau, 'sigma_m': sigma_m, 'f_c': params.get('f_c', np.nan)}

    elif method == 'acf':
        # Autocorrelation function fitting
        max_lag = min(len(x) // 4, int(0.5 * fs))  # Up to 0.5s lag
        acf = np.correlate(x, x, mode='full')[len(x)-1:len(x)+max_lag] / len(x)
        acf = acf / acf[0]  # Normalize

        lags = np.arange(max_lag) * dt

        # Fit exponential: ACF(t) = exp(-t/τ)
        def exp_decay(t, tau):
            return np.exp(-t / tau)

        from scipy.optimize import curve_fit
        try:
            popt, _ = curve_fit(exp_decay, lags[1:], acf[1:], p0=[0.01],
                                bounds=([1e-4], [1.0]))
            tau = popt[0]

            # Estimate D from variance: var = D * τ
            D = np.var(x) / tau

            return {'D': D, 'tau': tau, 'sigma_m': 0.0, 'f_c': 1/(2*np.pi*tau)}
        except:
            return {'D': np.nan, 'tau': np.nan, 'sigma_m': np.nan, 'f_c': np.nan}

    elif method == 'mle':
        # Maximum likelihood estimation for discretized OU process
        # x_{t+dt} = α * x_t + ε, where α = exp(-dt/τ)
        # var(ε) = D*τ*(1 - exp(-2*dt/τ))

        x_cur = x[:-1]
        x_next = x[1:]

        # Linear regression to get α
        alpha = np.sum(x_cur * x_next) / np.sum(x_cur**2)
        alpha = np.clip(alpha, 0, 1)  # Physical constraint

        if alpha > 0 and alpha < 1:
            tau = -dt / np.log(alpha)
        else:
            tau = np.inf

        # Estimate D from residual variance
        residuals = x_next - alpha * x_cur
        var_residual = np.var(residuals)

        if tau < np.inf:
            D = var_residual / (tau * (1 - np.exp(-2*dt/tau)))
        else:
            D = var_residual / (2 * dt)

        return {'D': D, 'tau': tau, 'sigma_m': 0.0, 'f_c': 1/(2*np.pi*tau) if tau < np.inf else 0.0}

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# MASKING & EVALUATION FRAMEWORK
# ============================================================================

def generate_realistic_gaps(n: int, fs: float,
                            gap_prob: float = 0.18,
                            mean_gap_ms: float = 25.0,
                            max_gap_ms: float = 200.0,
                            rng: np.random.Generator = RNG) -> np.ndarray:
    """
    Generate realistic missing data pattern for optical tweezers.

    Models realistic dropout patterns:
    - Random isolated gaps (detector saturation, dust particles)
    - Longer gaps (temporary loss of lock, vibrations)

    Returns:
        mask: (n,) boolean array, True = observed, False = missing
    """
    mask = np.ones(n, dtype=bool)

    mean_gap_samples = max(1, int(mean_gap_ms * 1e-3 * fs))
    max_gap_samples = int(max_gap_ms * 1e-3 * fs)

    i = 0
    while i < n:
        if rng.random() < gap_prob:
            # Sample gap length from exponential distribution, clipped
            gap_len = int(rng.exponential(mean_gap_samples))
            gap_len = min(gap_len, max_gap_samples)
            gap_len = max(1, gap_len)

            # Create gap
            end = min(i + gap_len, n)
            mask[i:end] = False
            i = end
        else:
            i += 1

    return mask


def create_evaluation_splits(t: np.ndarray, X: np.ndarray,
                             config: ExperimentConfig = CONFIG) -> List[Dict[str, Any]]:
    """
    Create multiple independent test segments for robust evaluation.

    Returns list of dicts with:
        - t: time array for segment
        - X_true: ground truth
        - obs_mask: observation mask
        - segment_idx: (start, end) indices in original data
    """
    N = len(t)
    segment_length = int(config.test_segment_length_s * config.fs_in)
    n_segments = config.n_test_segments

    # Ensure segments don't overlap
    available_length = N - segment_length
    if available_length < n_segments * segment_length:
        print(f"⚠ Warning: Not enough data for {n_segments} non-overlapping segments")
        n_segments = max(1, available_length // segment_length)

    segment_starts = RNG.choice(available_length, size=n_segments, replace=False)
    segment_starts = np.sort(segment_starts)

    segments = []
    for i, start in enumerate(segment_starts):
        end = start + segment_length

        t_seg = t[start:end] - t[start]  # Reset time to start at 0
        X_seg = X[start:end]

        # Generate observation mask
        obs_mask = generate_realistic_gaps(
            segment_length,
            config.fs_in,
            config.gap_prob,
            config.mean_gap_duration_ms,
            config.max_gap_duration_ms,
            rng=RNG
        )

        segments.append({
            'segment_id': i,
            't': t_seg,
            'X_true': X_seg,
            'obs_mask': obs_mask,
            'segment_idx': (start, end),
            'obs_fraction': obs_mask.mean()
        })

        print(f"  Segment {i}: t=[{t[start]:.2f}, {t[end]:.2f}]s, "
              f"{obs_mask.sum()}/{len(obs_mask)} observed ({obs_mask.mean()*100:.1f}%)")

    return segments


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_var: np.ndarray = None, mask: np.ndarray = None,
                   name: str = "Metric") -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: (N,) or (N, D) ground truth
        y_pred: (N,) or (N, D) predictions
        y_var: (N,) or (N, D) predictive variance (optional)
        mask: (N,) boolean mask of which points to evaluate
        name: metric name for display

    Returns:
        dict of metrics
    """
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle multi-dimensional case
    if y_true.ndim > 1:
        metrics_per_dim = []
        for d in range(y_true.shape[1]):
            m = compute_metrics(y_true[:, d], y_pred[:, d],
                              y_var[:, d] if y_var is not None else None,
                              mask, f"{name}_dim{d}")
            metrics_per_dim.append(m)

        # Average across dimensions
        avg_metrics = {}
        for key in metrics_per_dim[0].keys():
            if not key.endswith('_dim0'):  # Skip dimension-specific keys
                values = [m.get(key, np.nan) for m in metrics_per_dim]
                avg_metrics[key] = np.nanmean(values)

        return avg_metrics

    # 1D case
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    errors = y_pred_masked - y_true_masked

    metrics = {
        'MAE': float(np.mean(np.abs(errors))),
        'MSE': float(np.mean(errors**2)),
        'RMSE': float(np.sqrt(np.mean(errors**2))),
        'MaxAE': float(np.max(np.abs(errors))),
        'MedAE': float(np.median(np.abs(errors))),
        'R2': float(1 - np.sum(errors**2) / np.sum((y_true_masked - y_true_masked.mean())**2)),
        'Pearson_r': float(np.corrcoef(y_true_masked, y_pred_masked)[0, 1])
    }

    # Probabilistic metrics
    if y_var is not None:
        y_var_masked = y_var[mask]
        y_var_masked = np.clip(y_var_masked, 1e-10, None)  # Avoid numerical issues

        # Negative log-likelihood
        nll = 0.5 * np.mean(np.log(2 * np.pi * y_var_masked) + errors**2 / y_var_masked)
        metrics['NLL'] = float(nll)

        # Calibration: empirical coverage
        for n_std in [1, 2, 3]:
            std = np.sqrt(y_var_masked)
            in_interval = np.abs(errors) <= n_std * std
            empirical_coverage = in_interval.mean()
            expected_coverage = stats.norm.cdf(n_std) - stats.norm.cdf(-n_std)
            metrics[f'Coverage_{n_std}std'] = float(empirical_coverage)
            metrics[f'Coverage_error_{n_std}std'] = float(abs(empirical_coverage - expected_coverage))

        # Mean standardized log loss (MSLL)
        baseline_var = np.var(y_true_masked)
        msll = nll - 0.5 * np.log(2 * np.pi * baseline_var) - 0.5
        metrics['MSLL'] = float(msll)

    return metrics


def print_metrics_table(results: Dict[str, Dict[str, float]],
                       metrics_to_show: List[str] = None):
    """Pretty print metrics comparison table."""
    if metrics_to_show is None:
        metrics_to_show = ['MAE', 'RMSE', 'R2', 'NLL', 'Coverage_1std', 'Coverage_2std']

    # Filter available metrics
    all_metrics = set()
    for r in results.values():
        all_metrics.update(r.keys())
    metrics_to_show = [m for m in metrics_to_show if m in all_metrics]

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Header
    header = f"{'Method':<25}"
    for m in metrics_to_show:
        header += f"{m:>12}"
    print(header)
    print("-"*80)

    # Rows
    for method_name, metrics in results.items():
        row = f"{method_name:<25}"
        for m in metrics_to_show:
            val = metrics.get(m, np.nan)
            if np.isnan(val):
                row += f"{'N/A':>12}"
            elif abs(val) < 0.01 or abs(val) > 1000:
                row += f"{val:>12.2e}"
            else:
                row += f"{val:>12.4f}"
        print(row)

    print("="*80 + "\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_trajectory_overview(t: np.ndarray, X: np.ndarray,
                             obs_mask: np.ndarray = None,
                             title: str = "Trajectory Overview",
                             filename: str = None):
    """Plot trajectory with observation mask."""
    D = X.shape[1]

    fig, axes = plt.subplots(D, 1, figsize=(14, 3*D), squeeze=False)

    for d in range(D):
        ax = axes[d, 0]

        if obs_mask is not None:
            # Plot observed points
            ax.plot(t[obs_mask], X[obs_mask, d], 'o', ms=2, alpha=0.6,
                   label='Observed', color='C0')
            # Plot missing segments
            if not np.all(obs_mask):
                ax.plot(t[~obs_mask], X[~obs_mask, d], 'x', ms=3, alpha=0.4,
                       label='Missing (ground truth)', color='red')
        else:
            ax.plot(t, X[:, d], '-', lw=0.8, alpha=0.8)

        ax.set_ylabel(f'Position {"XY"[d] if D > 1 else "X"} (nm)')
        if obs_mask is not None:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close(fig)


def plot_psd_analysis(t: np.ndarray, X: np.ndarray,
                      fs: float = 300.0,
                      fit_params: Dict[str, float] = None,
                      filename: str = None):
    """Plot power spectral density with Lorentzian fit."""
    D = X.shape[1]

    fig, axes = plt.subplots(1, D, figsize=(7*D, 5), squeeze=False)

    for d in range(D):
        ax = axes[0, d]

        # Compute PSD
        x_detrended = signal.detrend(X[:, d], type='linear')
        freqs, psd = compute_psd(x_detrended, fs)

        # Plot
        ax.loglog(freqs, psd, 'o-', ms=3, alpha=0.6, label='Data')

        # Fit and overlay if params provided
        if fit_params is not None and d < len(fit_params):
            params = fit_params[d] if isinstance(fit_params, list) else fit_params
            if 'D' in params and 'f_c' in params:
                D_fit = params['D']
                f_c = params['f_c']
                psd_fit = D_fit / (np.pi**2 * (freqs**2 + f_c**2))
                ax.loglog(freqs, psd_fit, 'r--', lw=2,
                         label=f'Lorentzian fit\n$f_c$={f_c:.1f} Hz\n$D$={D_fit:.1f} nm²/s')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (nm²/Hz)')
        ax.set_title(f'Dimension {"XY"[d] if D > 1 else "X"}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Power Spectral Density Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close(fig)


def plot_imputation_comparison(t: np.ndarray, X_true: np.ndarray,
                               X_pred: np.ndarray, X_var: np.ndarray = None,
                               obs_mask: np.ndarray = None,
                               method_name: str = "Method",
                               t_zoom: Tuple[float, float] = None,
                               filename: str = None):
    """Plot imputation results with uncertainty."""
    D = X_true.shape[1]

    fig, axes = plt.subplots(D, 1, figsize=(14, 3*D), squeeze=False)

    for d in range(D):
        ax = axes[d, 0]

        # Ground truth
        ax.plot(t, X_true[:, d], '-', lw=1.5, alpha=0.5,
               label='Ground truth', color='black')

        # Predictions
        ax.plot(t, X_pred[:, d], '-', lw=1.2, alpha=0.8,
               label='Prediction', color='C1')

        # Uncertainty bands
        if X_var is not None:
            std = np.sqrt(X_var[:, d])
            ax.fill_between(t, X_pred[:, d] - 2*std, X_pred[:, d] + 2*std,
                           alpha=0.2, color='C1', label='±2σ')

        # Observation markers
        if obs_mask is not None:
            ax.plot(t[obs_mask], X_true[obs_mask, d], 'o', ms=2, alpha=0.5,
                   color='C0', label='Observed')

        ax.set_ylabel(f'Position {"XY"[d] if D > 1 else "X"} (nm)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        if t_zoom is not None:
            ax.set_xlim(t_zoom)

    axes[-1, 0].set_xlabel('Time (s)')
    fig.suptitle(f'{method_name} - Imputation Results',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  → Saved: {filename}")
    plt.close(fig)


# ============================================================================
# MAIN SETUP
# ============================================================================

def main_setup():
    """Run main setup and diagnostics."""
    print("="*80)
    print("OPTICAL TWEEZERS TRAJECTORY IMPUTATION FRAMEWORK")
    print("="*80)
    print(f"Device: {DEVICE if TORCH_AVAILABLE else 'CPU (PyTorch N/A)'}")
    print(f"Data path: {CONFIG.data_path}")
    print()

    # Load data
    t, X = load_trajectory_data(CONFIG.data_path, CONFIG)
    print()

    # Detrend
    X_detrended, trend = detrend_trajectory(X, method='linear')
    trend_slope = (trend[-1] - trend[0]) / t[-1]
    if X.shape[1] == 1:
        print(f"Removed linear trend: slope = {trend_slope[0]:.3f} nm/s")
    else:
        print(f"Removed linear trend: slope = {trend_slope} nm/s")
    print()

    # Estimate OU parameters per dimension
    print("Estimating Ornstein-Uhlenbeck parameters from PSD...")
    ou_params = []
    for d in range(X.shape[1]):
        params = estimate_ou_parameters(X_detrended[:, d], CONFIG.dt_in, method='psd')
        ou_params.append(params)
        print(f"  Dimension {d}: D={params['D']:.2f} nm²/s, "
              f"τ={params['tau']*1000:.2f} ms, f_c={params['f_c']:.2f} Hz")
    print()

    # Create evaluation splits
    print("Creating evaluation segments...")
    segments = create_evaluation_splits(t, X, CONFIG)
    print()

    # Visualizations
    print("Generating diagnostic plots...")
    plot_trajectory_overview(t[:6000], X[:6000],
                            title="Full Trajectory (first 20s)",
                            filename="00_trajectory_overview.png")
    plot_psd_analysis(t, X_detrended, CONFIG.fs_in, ou_params,
                     filename="00_psd_analysis.png")

    # Show sample segment
    seg = segments[0]
    plot_trajectory_overview(seg['t'], seg['X_true'], seg['obs_mask'],
                            title=f"Sample Evaluation Segment 0 "
                                  f"({seg['obs_fraction']*100:.1f}% observed)",
                            filename="00_sample_segment.png")

    print("✓ Setup complete. Ready for method implementation.")
    print("="*80)

    return t, X, X_detrended, ou_params, segments


# ============================================================================
# RUN SETUP
# ============================================================================

if __name__ == "__main__":
    t, X, X_detrended, ou_params, segments = main_setup()

    # Export for use in method-specific notebooks
    print("\nExported variables:")
    print("  - t: (N,) time array (seconds)")
    print("  - X: (N, D) trajectory (nanometers)")
    print("  - X_detrended: (N, D) detrended trajectory")
    print("  - ou_params: List of OU parameters per dimension")
    print("  - segments: List of evaluation segments with masks")
    print("  - CONFIG: ExperimentConfig object")
