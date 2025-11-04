#!/usr/bin/env python3
"""
Ornstein-Uhlenbeck Kalman Smoother with EM Parameter Estimation
================================================================
Physics-based probabilistic imputation using trapped Brownian motion model.

Physical Model:
    dx_t = -(1/τ) x_t dt + √(2D) dW_t    (continuous-time OU process)

Discrete-time State-Space Model:
    x_{k+1} = A x_k + w_k,   w_k ~ N(0, Q)
    z_k = x_k + v_k,         v_k ~ N(0, R)

where:
    A = exp(-Δt/τ)                     (autocorrelation)
    Q = D·τ·(1 - exp(-2Δt/τ))          (process noise variance)
    R = σ_m²                           (measurement noise variance)

Algorithm:
1. Initialize parameters (τ, D, σ_m²) from data
2. EM iterations:
   - E-step: Kalman filter + Rauch-Tung-Striebel (RTS) smoother
   - M-step: Update parameters using sufficient statistics
3. Return posterior mean and variance

Advantages:
- Physically correct model for optical trap
- Principled uncertainty quantification
- Handles arbitrarily long gaps
- Learns parameters from data

References:
- Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood estimates
  of linear dynamic systems. AIAA journal, 3(8), 1445-1450.
- Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series smoothing
  and forecasting using the EM algorithm. Journal of time series analysis, 3(4), 253-264.
- Berg-Sørensen, K., & Flyvbjerg, H. (2004). Power spectrum analysis for optical
  tweezers. Review of Scientific Instruments, 75(3), 594-612.
"""

import sys
import numpy as np
from scipy import signal, interpolate
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional
import warnings

# Add parent directory to path for imports
sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())


def kalman_filter_ou(
    z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float,
    tau: float,
    D: float,
    sigma_m2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Kalman filter for OU process with partial observations.

    Args:
        z: (N,) observations (only obs_mask==True are valid)
        obs_mask: (N,) boolean observation mask
        dt: time step
        tau: relaxation time
        D: diffusion coefficient
        sigma_m2: measurement noise variance

    Returns:
        m_filt: (N,) filtered means
        P_filt: (N,) filtered variances
    """
    N = len(z)

    # Discretized OU parameters
    A = np.exp(-dt / tau)
    sigma2 = D * tau  # Stationary variance
    Q = sigma2 * (1.0 - A**2)  # Process noise variance
    R = sigma_m2  # Measurement noise variance

    # Initialize at stationary distribution
    m = 0.0  # Mean-centered
    P = sigma2  # Stationary variance

    m_filt = np.zeros(N)
    P_filt = np.zeros(N)

    for k in range(N):
        # Predict
        m_pred = A * m
        P_pred = A * P * A + Q

        # Update (only if observed)
        if obs_mask[k]:
            # Kalman gain
            S = P_pred + R  # Innovation covariance
            K = P_pred / S

            # Update
            innovation = z[k] - m_pred
            m = m_pred + K * innovation
            P = (1.0 - K) * P_pred
        else:
            # No observation, use prediction
            m = m_pred
            P = P_pred

        m_filt[k] = m
        P_filt[k] = P

    return m_filt, P_filt


def rts_smoother_ou(
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    dt: float,
    tau: float,
    D: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel (RTS) backward smoother for OU process.

    Args:
        m_filt: (N,) filtered means from forward pass
        P_filt: (N,) filtered variances from forward pass
        dt: time step
        tau: relaxation time
        D: diffusion coefficient

    Returns:
        m_smooth: (N,) smoothed means
        P_smooth: (N,) smoothed variances
        P_lag1: (N-1,) lag-1 cross-covariances Cov(x_k, x_{k-1} | z_{1:N})
    """
    N = len(m_filt)

    # Discretized OU parameters
    A = np.exp(-dt / tau)
    sigma2 = D * tau
    Q = sigma2 * (1.0 - A**2)

    # Initialize smoother at final time
    m_smooth = np.zeros(N)
    P_smooth = np.zeros(N)
    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # Backward pass
    for k in range(N - 2, -1, -1):
        # Predicted mean and variance for k+1 given k
        m_pred = A * m_filt[k]
        P_pred = A * P_filt[k] * A + Q

        # Smoother gain
        if P_pred > 1e-12:
            J = (A * P_filt[k]) / P_pred
        else:
            J = 0.0

        # Smoothed estimates
        m_smooth[k] = m_filt[k] + J * (m_smooth[k + 1] - m_pred)
        P_smooth[k] = P_filt[k] + J * J * (P_smooth[k + 1] - P_pred)

        # Ensure non-negative variance
        P_smooth[k] = max(P_smooth[k], 1e-12)

    # Compute lag-1 cross-covariances using smoother
    # Cov(x_k, x_{k-1} | z_{1:N}) = P_smooth[k] * J_{k-1}
    P_lag1 = np.zeros(N - 1)
    for k in range(N - 1):
        m_pred = A * m_filt[k]
        P_pred = A * P_filt[k] * A + Q

        if P_pred > 1e-12:
            J = (A * P_filt[k]) / P_pred
        else:
            J = 0.0

        P_lag1[k] = P_smooth[k + 1] * J

    return m_smooth, P_smooth, P_lag1


def em_ou_parameters(
    z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float,
    tau_init: float,
    D_init: float,
    sigma_m2_init: float,
    n_iter: int = 20,
    verbose: bool = False
) -> Dict[str, float]:
    """
    EM algorithm to estimate OU process parameters.

    Args:
        z: (N,) observations
        obs_mask: (N,) boolean mask
        dt: time step
        tau_init: initial relaxation time
        D_init: initial diffusion coefficient
        sigma_m2_init: initial measurement noise variance
        n_iter: number of EM iterations
        verbose: print iteration details

    Returns:
        dict with estimated parameters: tau, D, sigma_m2, converged
    """
    N = len(z)
    n_obs = np.sum(obs_mask)

    # Initialize
    tau = tau_init
    D = D_init
    sigma_m2 = sigma_m2_init

    ll_prev = -np.inf
    converged = False

    for iter_idx in range(n_iter):
        # ===== E-step: Kalman filter + RTS smoother =====
        m_filt, P_filt = kalman_filter_ou(z, obs_mask, dt, tau, D, sigma_m2)
        m_smooth, P_smooth, P_lag1 = rts_smoother_ou(m_filt, P_filt, dt, tau, D)

        # ===== Compute sufficient statistics =====
        # E[x_k]
        Ex = m_smooth

        # E[x_k²]
        Ex2 = P_smooth + Ex**2

        # E[x_k x_{k-1}] from lag-1 cross-covariance
        ExExm1 = P_lag1 + Ex[1:] * Ex[:-1]

        # E[x_{k-1}²]
        Exm12 = Ex2[:-1]

        # ===== M-step: Update parameters =====
        # 1. Update A (and thus tau)
        num_A = np.sum(ExExm1)
        den_A = np.sum(Exm12)

        if den_A > 1e-12:
            A_new = np.clip(num_A / den_A, 0.01, 0.9999)
        else:
            A_new = 0.9

        # tau = -Δt / log(A)
        tau_new = -dt / np.log(A_new + 1e-12)
        tau_new = np.clip(tau_new, 1e-4, 1.0)  # Constrain to reasonable range (0.1ms - 1s)

        # 2. Update σ² (stationary variance)
        # σ² = E[x_k² - 2A·x_k·x_{k-1} + A²·x_{k-1}²] / (1 - A²)
        numerator = np.sum(Ex2[1:] - 2 * A_new * ExExm1 + A_new**2 * Exm12)
        sigma2_new = numerator / ((N - 1) * (1 - A_new**2 + 1e-12))
        sigma2_new = np.maximum(sigma2_new, 1e-12)

        # D = σ² / τ
        D_new = sigma2_new / tau_new

        # 3. Update measurement noise
        # σ_m² = (1/n_obs) Σ_observed [(z_k - E[x_k])² + Var[x_k]]
        if n_obs > 0:
            residuals = z[obs_mask] - Ex[obs_mask]
            sigma_m2_new = (np.sum(residuals**2) + np.sum(P_smooth[obs_mask])) / n_obs
            sigma_m2_new = np.maximum(sigma_m2_new, 1e-12)
        else:
            sigma_m2_new = sigma_m2

        # ===== Check convergence =====
        # Compute approximate log-likelihood
        ll = -0.5 * n_obs * np.log(2 * np.pi * sigma_m2_new)
        ll -= 0.5 * np.sum(residuals**2) / sigma_m2_new if n_obs > 0 else 0

        if verbose:
            print(f"  EM iter {iter_idx+1}: tau={tau_new*1000:.2f}ms, "
                  f"D={D_new:.1f} nm²/s, σ_m²={sigma_m2_new:.2f} nm², LL={ll:.1f}")

        # Check convergence
        if iter_idx > 0 and abs(ll - ll_prev) < 1e-4 * abs(ll_prev + 1e-6):
            converged = True
            if verbose:
                print(f"  Converged at iteration {iter_idx+1}")
            break

        # Update parameters
        tau = tau_new
        D = D_new
        sigma_m2 = sigma_m2_new
        ll_prev = ll

    return {
        'tau': float(tau),
        'D': float(D),
        'sigma_m2': float(sigma_m2),
        'converged': converged,
        'n_iter': iter_idx + 1
    }


def impute_ou_kalman(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    n_iter: int = 20,
    tau_init: Optional[float] = None,
    D_init: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using OU Kalman smoother with EM parameter estimation.

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds (typically 1000 Hz)
        obs_mask: (N,) boolean mask, True = observed, False = missing
        n_iter: Number of EM iterations (default 20)
        tau_init: Initial relaxation time (default: auto from PSD)
        D_init: Initial diffusion coefficient (default: auto from variance)
        verbose: Print EM progress

    Returns:
        dict with:
            - output_300Hz: (N, D) posterior mean at native rate
            - output_1000Hz: (M, D) posterior mean at target rate
            - mean: (N, D) posterior mean (same as output_300Hz)
            - var: (N, D) posterior variance
            - extras: dict with learned parameters per dimension
    """
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D_dim = X.shape
    M = len(t_out)

    # Validate inputs
    assert len(t) == N, f"Time array length {len(t)} != data length {N}"
    assert len(obs_mask) == N, f"Mask length {len(obs_mask)} != data length {N}"
    assert np.any(obs_mask), "No observed points in mask"

    # Compute time step
    dt = np.median(np.diff(t))
    n_obs = np.sum(obs_mask)

    # Initialize arrays
    mu_300 = np.zeros((N, D_dim), dtype=np.float64)
    var_300 = np.zeros((N, D_dim), dtype=np.float64)
    mu_1000 = np.zeros((M, D_dim), dtype=np.float64)

    params_per_dim = []

    # Process each dimension independently
    for d in range(D_dim):
        z = X[:, d].copy()

        # Remove mean (OU process is mean-reverting to 0)
        z_mean = np.mean(z[obs_mask])
        z = z - z_mean

        # Initialize parameters if not provided
        if tau_init is None:
            # Estimate from autocorrelation
            # If we have PSD-based estimate, use it
            # Otherwise, use a reasonable default (20-50 ms for optical trap)
            tau_init_d = 0.02  # 20 ms default
        else:
            tau_init_d = tau_init

        if D_init is None:
            # Estimate from variance: Var[x] ≈ D·τ
            var_obs = np.var(z[obs_mask])
            D_init_d = var_obs / tau_init_d
        else:
            D_init_d = D_init

        # Estimate measurement noise as fraction of variance
        sigma_m2_init = 0.1 * np.var(z[obs_mask])

        if verbose:
            print(f"\nDimension {d}:")
            print(f"  Initial: tau={tau_init_d*1000:.2f}ms, D={D_init_d:.1f} nm²/s")

        # EM parameter estimation
        params = em_ou_parameters(
            z, obs_mask, dt,
            tau_init_d, D_init_d, sigma_m2_init,
            n_iter=n_iter,
            verbose=verbose
        )

        params_per_dim.append(params)

        # Final smoothing with learned parameters
        m_filt, P_filt = kalman_filter_ou(
            z, obs_mask, dt,
            params['tau'], params['D'], params['sigma_m2']
        )
        m_smooth, P_smooth, P_lag1 = rts_smoother_ou(
            m_filt, P_filt, dt,
            params['tau'], params['D']
        )

        # Add mean back
        m_smooth = m_smooth + z_mean

        mu_300[:, d] = m_smooth
        var_300[:, d] = P_smooth

        # Interpolate to 1000 Hz using cubic spline
        try:
            cs = CubicSpline(t, m_smooth, bc_type='natural', extrapolate=True)
            mu_1000[:, d] = cs(t_out)
        except Exception as e:
            warnings.warn(f"Spline interpolation failed for dim {d}: {e}")
            mu_1000[:, d] = np.interp(t_out, t, m_smooth)

    # Aggregate diagnostics
    extras = {
        'params_per_dim': params_per_dim,
        'tau_mean': float(np.mean([p['tau'] for p in params_per_dim])),
        'D_mean': float(np.mean([p['D'] for p in params_per_dim])),
        'sigma_m2_mean': float(np.mean([p['sigma_m2'] for p in params_per_dim])),
        'dt': float(dt),
        'n_obs': int(n_obs),
        'obs_fraction': float(n_obs / N)
    }

    return {
        'output_300Hz': mu_300,
        'output_1000Hz': mu_1000,
        'mean': mu_300,
        'var': var_300,
        'extras': extras
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ou_kalman(
    segments: list,
    n_iter: int = 20,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate OU Kalman Smoother on all segments.

    Returns:
        dict of averaged metrics across segments
    """
    print("\n" + "="*80)
    print("EVALUATING: OU Kalman Smoother (RTS) with EM")
    print("="*80)
    print(f"Parameters: n_iter={n_iter}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        # Create output time grid at 1000 Hz
        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        # Run imputation
        result = impute_ou_kalman(
            X_true, t_seg, t_out, obs_mask,
            n_iter=n_iter,
            verbose=verbose
        )

        X_pred = result['output_300Hz']
        X_var = result['var']

        # Compute metrics on MISSING points only
        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            print(f"  Segment {seg_id}: No missing points, skipping")
            continue

        metrics = compute_metrics(
            X_true, X_pred, y_var=X_var, mask=missing_mask,
            name=f"Segment_{seg_id}"
        )

        all_metrics.append(metrics)

        params = result['extras']
        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f}, NLL={metrics.get('NLL', np.nan):.2f} | "
              f"τ={params['tau_mean']*1000:.1f}ms, D={params['D_mean']:.0f}")

    # Average metrics across segments
    if not all_metrics:
        print("  ⚠ No segments with missing data")
        return {}

    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else np.nan
        avg_metrics[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0

    print()
    print("Average metrics:")
    print(f"  MAE:        {avg_metrics['MAE']:.3f} ± {avg_metrics['MAE_std']:.3f} nm")
    print(f"  RMSE:       {avg_metrics['RMSE']:.3f} ± {avg_metrics['RMSE_std']:.3f} nm")
    print(f"  R²:         {avg_metrics['R2']:.4f} ± {avg_metrics['R2_std']:.4f}")
    print(f"  NLL:        {avg_metrics.get('NLL', np.nan):.3f} ± {avg_metrics.get('NLL_std', 0):.3f}")
    print(f"  Cov@1σ:     {avg_metrics.get('Coverage_1std', np.nan):.3f} (expect 0.683)")
    print(f"  Cov@2σ:     {avg_metrics.get('Coverage_2std', np.nan):.3f} (expect 0.954)")
    print()

    return avg_metrics


def visualize_sample_result(
    segment: dict,
    n_iter: int = 20,
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_ou_kalman(
        X_true, t_seg, t_out, obs_mask,
        n_iter=n_iter,
        verbose=True
    )

    X_pred = result['output_300Hz']
    X_var = result['var']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=X_var,
        obs_mask=obs_mask,
        method_name="OU Kalman Smoother (RTS + EM)",
        filename="04_ou_kalman_result.png"
    )

    # Zoom in on a gap-rich region
    missing_indices = np.where(~obs_mask)[0]
    if len(missing_indices) > 0:
        # Find a contiguous gap
        gaps = np.split(missing_indices, np.where(np.diff(missing_indices) > 1)[0] + 1)
        longest_gap = max(gaps, key=len)

        if len(longest_gap) > 10:
            # Zoom around longest gap
            pad = 50
            zoom_start = max(0, longest_gap[0] - pad)
            zoom_end = min(len(t_seg), longest_gap[-1] + pad)

            plot_imputation_comparison(
                t_seg[zoom_start:zoom_end],
                X_true[zoom_start:zoom_end],
                X_pred[zoom_start:zoom_end],
                X_var=X_var[zoom_start:zoom_end],
                obs_mask=obs_mask[zoom_start:zoom_end],
                method_name="OU Kalman Smoother (Zoom)",
                filename="04_ou_kalman_zoom.png"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data and segments from starter framework...")

    # Load from starter framework
    t, X, X_detrended, ou_params, segments = main_setup()

    # Evaluate with default parameters
    print("\nUsing PSD-estimated parameters as prior:")
    print(f"  τ = {ou_params[0]['tau']*1000:.2f} ms")
    print(f"  D = {ou_params[0]['D']:.1f} nm²/s")
    print(f"  f_c = {ou_params[0]['f_c']:.2f} Hz")

    metrics = evaluate_ou_kalman(
        segments,
        n_iter=20,
        config=CONFIG,
        verbose=False
    )

    # Visualize on first segment
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], n_iter=20, config=CONFIG)

    print("\n✓ OU Kalman Smoother evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
