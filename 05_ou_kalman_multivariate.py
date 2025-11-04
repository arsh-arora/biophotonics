#!/usr/bin/env python3
"""
Multi-variate Ornstein-Uhlenbeck Kalman Smoother
=================================================
Extension to handle multi-dimensional trajectories (e.g., x,y jointly).

Model:
    x_{k+1} = A x_k + w_k,   w_k ~ N(0, Q)
    z_k = x_k + v_k,         v_k ~ N(0, R)

where:
    A = diag(A_1, ..., A_D) with A_d = exp(-Δt/τ_d)   (diagonal, per-axis)
    Q = diag(Q_1, ..., Q_D) with Q_d = σ²_d(1 - A_d²) (diagonal process noise)
    R = diag(R_1, ..., R_D) with R_d = σ_{m,d}²       (diagonal measurement noise)

Options:
1. Independent per axis: separate (τ_d, D_d, σ_{m,d}²) for each dimension
2. Shared τ: same relaxation time across axes (physically coupled trap)
3. Full diagonal: most general, separate parameters per dimension

This implementation uses diagonal covariance (no cross-axis correlation) for efficiency.
For D=1 (current data), reduces to standard 1D OU Kalman.
For D=2 (x,y trajectories), handles both axes jointly while maintaining independence.
"""

import sys
import numpy as np
from scipy import signal, interpolate
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional, Literal
import warnings

# Add parent directory to path for imports
sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())


def kalman_filter_ou_multivariate(
    Z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float,
    tau: np.ndarray,
    D: np.ndarray,
    sigma_m2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-variate Kalman filter for OU process with diagonal covariance.

    Args:
        Z: (N, D_dim) observations
        obs_mask: (N, D_dim) boolean observation mask (can be per-element)
        dt: time step
        tau: (D_dim,) relaxation times per dimension
        D: (D_dim,) diffusion coefficients per dimension
        sigma_m2: (D_dim,) measurement noise variances per dimension

    Returns:
        M_filt: (N, D_dim) filtered means
        P_filt: (N, D_dim) filtered variances (diagonal elements only)
    """
    N, D_dim = Z.shape

    # Handle scalar obs_mask (same for all dimensions)
    if obs_mask.ndim == 1:
        obs_mask = np.tile(obs_mask[:, None], (1, D_dim))

    # Discretized OU parameters (per dimension)
    A = np.exp(-dt / tau)  # (D_dim,)
    sigma2 = D * tau  # Stationary variance (D_dim,)
    Q = sigma2 * (1.0 - A**2)  # Process noise variance (D_dim,)
    R = sigma_m2  # Measurement noise variance (D_dim,)

    # Initialize at stationary distribution
    m = np.zeros(D_dim)  # (D_dim,)
    P = sigma2.copy()  # (D_dim,) diagonal covariance

    M_filt = np.zeros((N, D_dim))
    P_filt = np.zeros((N, D_dim))

    for k in range(N):
        # Predict (per dimension, vectorized)
        m_pred = A * m
        P_pred = A * P * A + Q

        # Update (per dimension, only if observed)
        for d in range(D_dim):
            if obs_mask[k, d]:
                # Kalman gain for dimension d
                S = P_pred[d] + R[d]  # Innovation covariance
                K = P_pred[d] / S

                # Update
                innovation = Z[k, d] - m_pred[d]
                m[d] = m_pred[d] + K * innovation
                P[d] = (1.0 - K) * P_pred[d]
            else:
                # No observation, use prediction
                m[d] = m_pred[d]
                P[d] = P_pred[d]

        M_filt[k] = m
        P_filt[k] = P

    return M_filt, P_filt


def rts_smoother_ou_multivariate(
    M_filt: np.ndarray,
    P_filt: np.ndarray,
    dt: float,
    tau: np.ndarray,
    D: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-variate RTS smoother for OU process with diagonal covariance.

    Args:
        M_filt: (N, D_dim) filtered means
        P_filt: (N, D_dim) filtered variances (diagonal)
        dt: time step
        tau: (D_dim,) relaxation times
        D: (D_dim,) diffusion coefficients

    Returns:
        M_smooth: (N, D_dim) smoothed means
        P_smooth: (N, D_dim) smoothed variances (diagonal)
        P_lag1: (N-1, D_dim) lag-1 cross-covariances (diagonal)
    """
    N, D_dim = M_filt.shape

    # Discretized OU parameters
    A = np.exp(-dt / tau)  # (D_dim,)
    sigma2 = D * tau
    Q = sigma2 * (1.0 - A**2)  # (D_dim,)

    # Initialize smoother at final time
    M_smooth = np.zeros((N, D_dim))
    P_smooth = np.zeros((N, D_dim))
    M_smooth[-1] = M_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # Backward pass (vectorized per dimension)
    for k in range(N - 2, -1, -1):
        # Predicted mean and variance for k+1 given k
        M_pred = A * M_filt[k]
        P_pred = A * P_filt[k] * A + Q

        # Smoother gain (per dimension)
        J = np.where(P_pred > 1e-12, (A * P_filt[k]) / P_pred, 0.0)

        # Smoothed estimates
        M_smooth[k] = M_filt[k] + J * (M_smooth[k + 1] - M_pred)
        P_smooth[k] = P_filt[k] + J * J * (P_smooth[k + 1] - P_pred)

        # Ensure non-negative variance
        P_smooth[k] = np.maximum(P_smooth[k], 1e-12)

    # Compute lag-1 cross-covariances (diagonal)
    P_lag1 = np.zeros((N - 1, D_dim))
    for k in range(N - 1):
        M_pred = A * M_filt[k]
        P_pred = A * P_filt[k] * A + Q

        J = np.where(P_pred > 1e-12, (A * P_filt[k]) / P_pred, 0.0)
        P_lag1[k] = P_smooth[k + 1] * J

    return M_smooth, P_smooth, P_lag1


def em_ou_parameters_multivariate(
    Z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float,
    tau_init: np.ndarray,
    D_init: np.ndarray,
    sigma_m2_init: np.ndarray,
    n_iter: int = 20,
    share_tau: bool = False,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    EM algorithm for multi-variate OU process with diagonal covariance.

    Args:
        Z: (N, D_dim) observations
        obs_mask: (N, D_dim) or (N,) boolean mask
        dt: time step
        tau_init: (D_dim,) initial relaxation times
        D_init: (D_dim,) initial diffusion coefficients
        sigma_m2_init: (D_dim,) initial measurement noise variances
        n_iter: number of EM iterations
        share_tau: if True, use same τ across all dimensions
        verbose: print iteration details

    Returns:
        dict with estimated parameters per dimension
    """
    N, D_dim = Z.shape

    # Handle scalar obs_mask
    if obs_mask.ndim == 1:
        obs_mask = np.tile(obs_mask[:, None], (1, D_dim))

    n_obs = np.sum(obs_mask, axis=0)  # Per dimension

    # Initialize
    tau = tau_init.copy()
    D_coeff = D_init.copy()
    sigma_m2 = sigma_m2_init.copy()

    for iter_idx in range(n_iter):
        # ===== E-step: Kalman filter + RTS smoother =====
        M_filt, P_filt = kalman_filter_ou_multivariate(Z, obs_mask, dt, tau, D_coeff, sigma_m2)
        M_smooth, P_smooth, P_lag1 = rts_smoother_ou_multivariate(M_filt, P_filt, dt, tau, D_coeff)

        # ===== Compute sufficient statistics (per dimension) =====
        Ex = M_smooth  # (N, D_dim)
        Ex2 = P_smooth + Ex**2  # (N, D_dim)
        ExExm1 = P_lag1 + Ex[1:] * Ex[:-1]  # (N-1, D_dim)
        Exm12 = Ex2[:-1]  # (N-1, D_dim)

        # ===== M-step: Update parameters (per dimension) =====
        tau_new = np.zeros(D_dim)
        D_new = np.zeros(D_dim)
        sigma_m2_new = np.zeros(D_dim)

        for d in range(D_dim):
            # 1. Update A_d (and thus τ_d)
            num_A = np.sum(ExExm1[:, d])
            den_A = np.sum(Exm12[:, d])

            if den_A > 1e-12:
                A_new = np.clip(num_A / den_A, 0.01, 0.9999)
            else:
                A_new = 0.9

            tau_d = -dt / np.log(A_new + 1e-12)
            tau_d = np.clip(tau_d, 1e-4, 1.0)
            tau_new[d] = tau_d

            # 2. Update σ²_d
            numerator = np.sum(Ex2[1:, d] - 2 * A_new * ExExm1[:, d] + A_new**2 * Exm12[:, d])
            sigma2 = numerator / ((N - 1) * (1 - A_new**2 + 1e-12))
            sigma2 = np.maximum(sigma2, 1e-12)

            D_new[d] = sigma2 / tau_d

            # 3. Update measurement noise
            if n_obs[d] > 0:
                obs_d = obs_mask[:, d]
                residuals = Z[obs_d, d] - Ex[obs_d, d]
                sigma_m2_new[d] = (np.sum(residuals**2) + np.sum(P_smooth[obs_d, d])) / n_obs[d]
                sigma_m2_new[d] = np.maximum(sigma_m2_new[d], 1e-12)
            else:
                sigma_m2_new[d] = sigma_m2[d]

        # Optional: Share τ across dimensions
        if share_tau:
            tau_new[:] = np.mean(tau_new)

        if verbose:
            print(f"  EM iter {iter_idx+1}:")
            for d in range(D_dim):
                print(f"    Dim {d}: τ={tau_new[d]*1000:.2f}ms, "
                      f"D={D_new[d]:.1f} nm²/s, σ_m²={sigma_m2_new[d]:.2f} nm²")

        # Update parameters
        tau = tau_new
        D_coeff = D_new
        sigma_m2 = sigma_m2_new

    return {
        'tau': tau,
        'D': D_coeff,
        'sigma_m2': sigma_m2,
        'n_iter': n_iter
    }


def impute_ou_kalman_multivariate(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    n_iter: int = 20,
    share_tau: bool = False,
    tau_init: Optional[np.ndarray] = None,
    D_init: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Multi-variate OU Kalman smoother with EM parameter estimation.

    Supports both 1D and 2D trajectories with diagonal covariance structure.
    Dimensions are treated independently (no cross-correlation).

    Args:
        X: (N, D_dim) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds (typically 1000 Hz)
        obs_mask: (N,) or (N, D_dim) boolean mask, True = observed, False = missing
        n_iter: Number of EM iterations (default 20)
        share_tau: If True, use same τ across all dimensions
        tau_init: (D_dim,) initial relaxation times (default: auto)
        D_init: (D_dim,) initial diffusion coefficients (default: auto)
        verbose: Print EM progress

    Returns:
        dict with:
            - output_300Hz: (N, D_dim) posterior mean at native rate
            - output_1000Hz: (M, D_dim) posterior mean at target rate
            - mean: (N, D_dim) posterior mean
            - var: (N, D_dim) posterior variance (diagonal elements)
            - extras: dict with learned parameters
    """
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D_dim = X.shape
    M = len(t_out)

    # Handle scalar obs_mask
    if obs_mask.ndim == 1:
        obs_mask_full = np.tile(obs_mask[:, None], (1, D_dim))
    else:
        obs_mask_full = obs_mask

    # Validate inputs
    assert len(t) == N, f"Time array length {len(t)} != data length {N}"
    assert np.any(obs_mask), "No observed points in mask"

    # Compute time step
    dt = np.median(np.diff(t))

    # Remove mean per dimension
    Z_mean = np.mean(X[obs_mask_full].reshape(-1, D_dim) if obs_mask.ndim > 1
                     else X[obs_mask], axis=0)
    Z = X - Z_mean

    # Initialize parameters per dimension
    if tau_init is None:
        tau_init = np.full(D_dim, 0.02)  # 20 ms default
    else:
        tau_init = np.atleast_1d(tau_init)
        if len(tau_init) == 1:
            tau_init = np.full(D_dim, tau_init[0])

    if D_init is None:
        var_per_dim = np.var(Z, axis=0)
        D_init = var_per_dim / tau_init
    else:
        D_init = np.atleast_1d(D_init)
        if len(D_init) == 1:
            D_init = np.full(D_dim, D_init[0])

    sigma_m2_init = 0.1 * np.var(Z, axis=0)

    if verbose:
        print(f"\nMulti-variate OU Kalman (D={D_dim}, share_tau={share_tau}):")
        for d in range(D_dim):
            print(f"  Dim {d} init: τ={tau_init[d]*1000:.2f}ms, D={D_init[d]:.1f} nm²/s")

    # EM parameter estimation
    params = em_ou_parameters_multivariate(
        Z, obs_mask_full, dt,
        tau_init, D_init, sigma_m2_init,
        n_iter=n_iter,
        share_tau=share_tau,
        verbose=verbose
    )

    # Final smoothing with learned parameters
    M_filt, P_filt = kalman_filter_ou_multivariate(
        Z, obs_mask_full, dt,
        params['tau'], params['D'], params['sigma_m2']
    )
    M_smooth, P_smooth, P_lag1 = rts_smoother_ou_multivariate(
        M_filt, P_filt, dt,
        params['tau'], params['D']
    )

    # Add mean back
    M_smooth = M_smooth + Z_mean

    mu_300 = M_smooth
    var_300 = P_smooth

    # Interpolate to 1000 Hz per dimension
    mu_1000 = np.zeros((M, D_dim))
    for d in range(D_dim):
        try:
            cs = CubicSpline(t, M_smooth[:, d], bc_type='natural', extrapolate=True)
            mu_1000[:, d] = cs(t_out)
        except Exception as e:
            warnings.warn(f"Spline failed for dim {d}: {e}")
            mu_1000[:, d] = np.interp(t_out, t, M_smooth[:, d])

    # Aggregate diagnostics
    extras = {
        'tau': params['tau'],
        'D': params['D'],
        'sigma_m2': params['sigma_m2'],
        'tau_mean': float(np.mean(params['tau'])),
        'D_mean': float(np.mean(params['D'])),
        'sigma_m2_mean': float(np.mean(params['sigma_m2'])),
        'share_tau': share_tau,
        'dt': float(dt),
        'n_obs': int(np.sum(obs_mask)),
        'obs_fraction': float(np.mean(obs_mask))
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

def evaluate_ou_kalman_multivariate(
    segments: list,
    n_iter: int = 20,
    share_tau: bool = False,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate multi-variate OU Kalman Smoother."""
    print("\n" + "="*80)
    print(f"EVALUATING: Multi-variate OU Kalman Smoother (share_tau={share_tau})")
    print("="*80)
    print(f"Parameters: n_iter={n_iter}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_ou_kalman_multivariate(
            X_true, t_seg, t_out, obs_mask,
            n_iter=n_iter,
            share_tau=share_tau,
            verbose=verbose
        )

        X_pred = result['output_300Hz']
        X_var = result['var']

        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            continue

        metrics = compute_metrics(
            X_true, X_pred, y_var=X_var, mask=missing_mask
        )

        all_metrics.append(metrics)

        params = result['extras']
        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f}, NLL={metrics.get('NLL', np.nan):.2f} | "
              f"τ={params['tau_mean']*1000:.1f}ms, D={params['D_mean']:.0f}")

    if not all_metrics:
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
    print(f"  NLL:        {avg_metrics.get('NLL', np.nan):.3f}")
    print(f"  Cov@1σ:     {avg_metrics.get('Coverage_1std', np.nan):.3f}")
    print(f"  Cov@2σ:     {avg_metrics.get('Coverage_2std', np.nan):.3f}")
    print()

    return avg_metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data and segments from starter framework...")
    t, X, X_detrended, ou_params, segments = main_setup()

    # Compare independent vs shared τ
    print("\n" + "="*80)
    print("COMPARISON: Independent τ vs Shared τ")
    print("="*80)

    metrics_indep = evaluate_ou_kalman_multivariate(
        segments, n_iter=20, share_tau=False, config=CONFIG, verbose=False
    )

    metrics_shared = evaluate_ou_kalman_multivariate(
        segments, n_iter=20, share_tau=True, config=CONFIG, verbose=False
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Independent τ: MAE={metrics_indep.get('MAE', np.nan):.3f} nm, "
          f"R²={metrics_indep.get('R2', np.nan):.4f}")
    print(f"Shared τ:      MAE={metrics_shared.get('MAE', np.nan):.3f} nm, "
          f"R²={metrics_shared.get('R2', np.nan):.4f}")
    print("\n✓ Multi-variate OU Kalman evaluation complete.")
