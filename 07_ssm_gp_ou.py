#!/usr/bin/env python3
"""
State-Space GP with OU Prior (O(N) via Kalman Filter)
======================================================
GP regression with OU kernel reformulated as linear state-space model.

Key Insight:
    GP with Matérn-1/2 (exponential/OU) kernel can be represented as:

    State-space model (linear SDE):
        dx_t = -(1/τ) x_t dt + √(2D) dW_t
        y_t = x_t + ε_t,  ε_t ~ N(0, σ_m²)

    Discretized:
        x_{k+1} = A x_k + w_k,  w_k ~ N(0, Q)
        y_k = x_k + v_k,        v_k ~ N(0, R)

This allows O(N) inference via Kalman filter/smoother instead of O(N³) GP.

Algorithm:
1. Parameterize by θ = (log τ, log D, log σ_m²)
2. For each θ, run Kalman filter to compute log p(y|θ) (marginal likelihood)
3. Optimize θ using gradient-based optimizer (Adam or L-BFGS)
4. Run RTS smoother with optimal θ for posterior mean/variance
5. Spline to 1000 Hz

Advantages over EM (B1):
- Direct optimization of marginal likelihood
- Can use modern optimizers (Adam, L-BFGS)
- Same O(N) complexity but potentially faster convergence
- Equivalent to exact GP inference with OU kernel

References:
- Särkkä, S., & Solin, A. (2019). Applied stochastic differential equations.
- Hartikainen, J., & Särkkä, S. (2010). Kalman filtering and smoothing solutions
  to temporal Gaussian process regression models. MLSP 2010.
"""

import sys
import numpy as np
from scipy import signal, interpolate, optimize
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional, Callable
import warnings

# Add parent directory to path for imports
sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())

# Try importing PyTorch for gradient-based optimization
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None


def kalman_filter_marginal_likelihood(
    z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float,
    tau: float,
    D: float,
    sigma_m2: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Kalman filter that also computes marginal log-likelihood.

    Returns:
        log_likelihood: log p(z | τ, D, σ_m²)
        m_filt: (N,) filtered means
        P_filt: (N,) filtered variances
    """
    N = len(z)

    # Discretized OU parameters
    A = np.exp(-dt / tau)
    sigma2 = D * tau
    Q = sigma2 * (1.0 - A**2)
    R = sigma_m2

    # Initialize
    m = 0.0
    P = sigma2

    m_filt = np.zeros(N)
    P_filt = np.zeros(N)

    log_likelihood = 0.0

    for k in range(N):
        # Predict
        m_pred = A * m
        P_pred = A * P * A + Q

        # Update (only if observed)
        if obs_mask[k]:
            # Innovation
            innovation = z[k] - m_pred
            S = P_pred + R  # Innovation covariance

            # Kalman gain
            K = P_pred / S

            # Update
            m = m_pred + K * innovation
            P = (1.0 - K) * P_pred

            # Marginal log-likelihood contribution
            # log N(z_k | m_pred, S)
            log_likelihood += -0.5 * (np.log(2 * np.pi * S) + innovation**2 / S)
        else:
            # No observation
            m = m_pred
            P = P_pred

        m_filt[k] = m
        P_filt[k] = P

    return log_likelihood, m_filt, P_filt


def rts_smoother(
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    dt: float,
    tau: float,
    D: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel backward smoother.

    (Same as in OU Kalman, just returns mean and variance)
    """
    N = len(m_filt)

    A = np.exp(-dt / tau)
    sigma2 = D * tau
    Q = sigma2 * (1.0 - A**2)

    # Initialize at final time
    m_smooth = np.zeros(N)
    P_smooth = np.zeros(N)
    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # Backward pass
    for k in range(N - 2, -1, -1):
        m_pred = A * m_filt[k]
        P_pred = A * P_filt[k] * A + Q

        if P_pred > 1e-12:
            J = (A * P_filt[k]) / P_pred
        else:
            J = 0.0

        m_smooth[k] = m_filt[k] + J * (m_smooth[k + 1] - m_pred)
        P_smooth[k] = P_filt[k] + J * J * (P_smooth[k + 1] - P_pred)
        P_smooth[k] = max(P_smooth[k], 1e-12)

    return m_smooth, P_smooth


def negative_marginal_likelihood(
    log_params: np.ndarray,
    z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float
) -> float:
    """
    Negative marginal log-likelihood for optimization.

    Args:
        log_params: [log(τ), log(D), log(σ_m²)]
        z: observations
        obs_mask: boolean mask
        dt: time step

    Returns:
        -log p(z | θ)
    """
    # Transform from log-space
    tau = np.exp(log_params[0])
    D = np.exp(log_params[1])
    sigma_m2 = np.exp(log_params[2])

    # Constrain to reasonable ranges
    tau = np.clip(tau, 1e-4, 1.0)
    D = np.clip(D, 1.0, 1e6)
    sigma_m2 = np.clip(sigma_m2, 1e-6, 1e4)

    try:
        log_lik, _, _ = kalman_filter_marginal_likelihood(
            z, obs_mask, dt, tau, D, sigma_m2
        )
        return -log_lik
    except:
        return 1e10  # Return large value on numerical errors


def optimize_hyperparameters_scipy(
    z: np.ndarray,
    obs_mask: np.ndarray,
    dt: float,
    tau_init: float,
    D_init: float,
    sigma_m2_init: float,
    method: str = 'L-BFGS-B',
    maxiter: int = 100,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Optimize hyperparameters using scipy.optimize.

    Args:
        z: (N,) observations
        obs_mask: (N,) boolean mask
        dt: time step
        tau_init: initial τ
        D_init: initial D
        sigma_m2_init: initial σ_m²
        method: optimization method ('L-BFGS-B', 'BFGS', 'Nelder-Mead')
        maxiter: maximum iterations
        verbose: print progress

    Returns:
        dict with optimized parameters
    """
    # Initialize in log-space
    log_params_init = np.array([
        np.log(tau_init),
        np.log(D_init),
        np.log(sigma_m2_init)
    ])

    # Bounds in log-space
    bounds = [
        (np.log(1e-4), np.log(1.0)),      # τ: 0.1ms to 1s
        (np.log(1.0), np.log(1e6)),        # D: 1 to 1e6 nm²/s
        (np.log(1e-6), np.log(1e4))        # σ_m²: very small to large
    ]

    if verbose:
        print(f"    Optimizing hyperparameters with {method}...")
        print(f"    Initial: τ={tau_init*1000:.2f}ms, D={D_init:.1f}, σ_m²={sigma_m2_init:.2f}")

    # Optimize
    result = optimize.minimize(
        negative_marginal_likelihood,
        log_params_init,
        args=(z, obs_mask, dt),
        method=method,
        bounds=bounds,
        options={'maxiter': maxiter, 'disp': verbose}
    )

    # Extract optimized parameters
    log_params_opt = result.x
    tau_opt = np.exp(log_params_opt[0])
    D_opt = np.exp(log_params_opt[1])
    sigma_m2_opt = np.exp(log_params_opt[2])

    if verbose:
        print(f"    Optimized: τ={tau_opt*1000:.2f}ms, D={D_opt:.1f}, σ_m²={sigma_m2_opt:.2f}")
        print(f"    Log-likelihood: {-result.fun:.2f}, Iterations: {result.nit}")

    return {
        'tau': float(tau_opt),
        'D': float(D_opt),
        'sigma_m2': float(sigma_m2_opt),
        'log_likelihood': float(-result.fun),
        'success': result.success,
        'n_iter': result.nit
    }


def impute_ssm_gp_ou(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    tau_init: Optional[float] = None,
    D_init: Optional[float] = None,
    method: str = 'L-BFGS-B',
    maxiter: int = 100,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    State-space GP with OU prior using Kalman filter (O(N) complexity).

    Equivalent to exact GP with Matérn-1/2 kernel but O(N) instead of O(N³).

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds
        obs_mask: (N,) boolean mask
        tau_init: initial relaxation time (default: auto)
        D_init: initial diffusion coefficient (default: auto)
        method: scipy optimization method ('L-BFGS-B', 'BFGS', 'Nelder-Mead')
        maxiter: maximum optimization iterations
        verbose: print optimization progress

    Returns:
        dict with:
            - output_300Hz: (N, D) posterior mean at native rate
            - output_1000Hz: (M, D) posterior mean at target rate
            - mean: (N, D) posterior mean
            - var: (N, D) posterior variance
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

    # Validate
    assert len(t) == N
    assert len(obs_mask) == N
    assert np.any(obs_mask)

    dt = np.median(np.diff(t))
    n_obs = np.sum(obs_mask)

    # Initialize arrays
    mu_300 = np.zeros((N, D_dim), dtype=np.float64)
    var_300 = np.zeros((N, D_dim), dtype=np.float64)
    mu_1000 = np.zeros((M, D_dim), dtype=np.float64)

    params_per_dim = []

    # Process each dimension
    for d in range(D_dim):
        z = X[:, d].copy()

        # Remove mean
        z_mean = np.mean(z[obs_mask])
        z = z - z_mean

        # Initialize parameters
        if tau_init is None:
            tau_init_d = 0.02  # 20 ms
        else:
            tau_init_d = tau_init

        if D_init is None:
            var_obs = np.var(z[obs_mask])
            D_init_d = var_obs / tau_init_d
        else:
            D_init_d = D_init

        sigma_m2_init = 0.1 * np.var(z[obs_mask])

        if verbose:
            print(f"\nDimension {d}:")

        # Optimize hyperparameters
        params = optimize_hyperparameters_scipy(
            z, obs_mask, dt,
            tau_init_d, D_init_d, sigma_m2_init,
            method=method,
            maxiter=maxiter,
            verbose=verbose
        )

        params_per_dim.append(params)

        # Final smoothing with optimized parameters
        _, m_filt, P_filt = kalman_filter_marginal_likelihood(
            z, obs_mask, dt,
            params['tau'], params['D'], params['sigma_m2']
        )

        m_smooth, P_smooth = rts_smoother(
            m_filt, P_filt, dt,
            params['tau'], params['D']
        )

        # Add mean back
        m_smooth = m_smooth + z_mean

        mu_300[:, d] = m_smooth
        var_300[:, d] = P_smooth

        # Interpolate to 1000 Hz
        try:
            cs = CubicSpline(t, m_smooth, bc_type='natural', extrapolate=True)
            mu_1000[:, d] = cs(t_out)
        except Exception as e:
            warnings.warn(f"Spline failed for dim {d}: {e}")
            mu_1000[:, d] = np.interp(t_out, t, m_smooth)

    # Aggregate diagnostics
    extras = {
        'params_per_dim': params_per_dim,
        'tau_mean': float(np.mean([p['tau'] for p in params_per_dim])),
        'D_mean': float(np.mean([p['D'] for p in params_per_dim])),
        'sigma_m2_mean': float(np.mean([p['sigma_m2'] for p in params_per_dim])),
        'log_likelihood_mean': float(np.mean([p['log_likelihood'] for p in params_per_dim])),
        'method': method,
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

def evaluate_ssm_gp_ou(
    segments: list,
    method: str = 'L-BFGS-B',
    maxiter: int = 100,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate State-Space GP with OU prior."""

    print("\n" + "="*80)
    print(f"EVALUATING: State-Space GP with OU Prior (method={method})")
    print("="*80)
    print(f"Parameters: maxiter={maxiter}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_ssm_gp_ou(
            X_true, t_seg, t_out, obs_mask,
            method=method,
            maxiter=maxiter,
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
              f"τ={params['tau_mean']*1000:.1f}ms, D={params['D_mean']:.0f}, "
              f"LL={params['log_likelihood_mean']:.1f}")

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


def visualize_sample_result(
    segment: dict,
    method: str = 'L-BFGS-B',
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_ssm_gp_ou(
        X_true, t_seg, t_out, obs_mask,
        method=method,
        verbose=True
    )

    X_pred = result['output_300Hz']
    X_var = result['var']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=X_var,
        obs_mask=obs_mask,
        method_name=f"State-Space GP (OU Prior, {method})",
        filename="07_ssm_gp_ou_result.png"
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
                X_var=X_var[zoom_start:zoom_end],
                obs_mask=obs_mask[zoom_start:zoom_end],
                method_name="State-Space GP (Zoom)",
                filename="07_ssm_gp_ou_zoom.png"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data and segments from starter framework...")
    t, X, X_detrended, ou_params, segments = main_setup()

    # Evaluate
    metrics = evaluate_ssm_gp_ou(
        segments,
        method='L-BFGS-B',
        maxiter=100,
        config=CONFIG,
        verbose=False
    )

    # Visualize
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], method='L-BFGS-B', config=CONFIG)

    print("\n✓ State-Space GP with OU prior evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
