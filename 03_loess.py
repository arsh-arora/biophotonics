#!/usr/bin/env python3
"""
LOESS (Locally Estimated Scatterplot Smoothing)
================================================
Robust non-parametric local polynomial regression.

Method:
1. For each evaluation point, select k nearest observed neighbors
2. Fit weighted local polynomial (degree 0, 1, or 2)
3. Use tricube kernel for distance weighting
4. Iteratively reweight using Tukey bisquare for robustness to outliers
5. Interpolate to target sampling rate using cubic spline

Advantages:
- Robust to outliers and non-uniform gap distributions
- No assumption about global functional form
- Adapts to local data density

Disadvantages:
- Computationally expensive (O(N²) for naive implementation)
- Requires careful bandwidth (frac) tuning
- No uncertainty quantification

References:
- Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots.
  Journal of the American statistical association, 74(368), 829-836.
- Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted regression: an approach to
  regression analysis by local fitting. JASA, 83(403), 596-610.
"""

import sys
import math
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


def tricube_kernel(u: np.ndarray) -> np.ndarray:
    """
    Tricube kernel function for LOESS weighting.

    K(u) = (1 - |u|^3)^3 for |u| < 1, else 0

    Args:
        u: Normalized distance (distance / max_distance)

    Returns:
        Kernel weights in [0, 1]
    """
    u = np.abs(u)
    a = np.clip(1.0 - u**3, 0.0, 1.0)
    return a**3


def loess_1d(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_eval: np.ndarray,
    frac: float = 0.1,
    n_iter: int = 2,
    degree: int = 1
) -> np.ndarray:
    """
    1D LOESS with robust iterative reweighting.

    Args:
        x_obs: (n_obs,) observed x-coordinates
        y_obs: (n_obs,) observed y-values
        x_eval: (n_eval,) evaluation points
        frac: Fraction of data to use for smoothing (bandwidth parameter)
              Larger frac = smoother fit
        n_iter: Number of robust reweighting iterations
                iter=1 is standard LOESS, iter>1 adds robustness
        degree: Local polynomial degree (0=constant, 1=linear, 2=quadratic)

    Returns:
        y_eval: (n_eval,) fitted values at evaluation points
    """
    x_obs = np.asarray(x_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)

    n_obs = len(x_obs)
    n_eval = len(x_eval)

    # Determine neighborhood size
    k_neighbors = max(degree + 2, int(math.ceil(frac * n_obs)))
    k_neighbors = min(k_neighbors, n_obs)  # Don't exceed number of observations

    y_eval = np.zeros(n_eval, dtype=np.float64)
    robust_weights = np.ones(n_obs, dtype=np.float64)

    # Iterative robust fitting
    for iter_idx in range(n_iter):
        # Fit at each evaluation point
        for i, x_target in enumerate(x_eval):
            # Find k nearest neighbors
            distances = np.abs(x_obs - x_target)

            # Use argpartition for O(n) selection instead of O(n log n) sort
            if k_neighbors < n_obs:
                neighbor_indices = np.argpartition(distances, k_neighbors - 1)[:k_neighbors]
            else:
                neighbor_indices = np.arange(n_obs)

            x_neighbors = x_obs[neighbor_indices]
            y_neighbors = y_obs[neighbor_indices]
            d_neighbors = distances[neighbor_indices]

            # Tricube distance weights
            max_dist = np.max(d_neighbors)
            if max_dist < 1e-10:
                # Evaluation point coincides with observed point
                y_eval[i] = np.mean(y_neighbors * robust_weights[neighbor_indices])
                continue

            normalized_dist = d_neighbors / max_dist
            distance_weights = tricube_kernel(normalized_dist)

            # Combined weights: distance × robust
            weights = distance_weights * robust_weights[neighbor_indices]
            weights_sum = np.sum(weights)

            if weights_sum < 1e-10:
                # All weights zero - use unweighted mean
                y_eval[i] = np.mean(y_neighbors)
                continue

            # Normalize weights
            weights = weights / weights_sum

            # Fit local polynomial using weighted least squares
            # Center x at evaluation point for numerical stability
            x_centered = x_neighbors - x_target

            if degree == 0:
                # Local constant (weighted mean)
                y_eval[i] = np.sum(weights * y_neighbors)

            elif degree == 1:
                # Local linear: y = β₀ + β₁(x - x_target)
                # Design matrix: [1, x - x_target]
                A = np.vstack([np.ones(k_neighbors), x_centered]).T

                # Weighted least squares: (A'WA)⁻¹ A'Wy
                W_sqrt = np.sqrt(weights)
                A_weighted = A * W_sqrt[:, None]
                y_weighted = y_neighbors * W_sqrt

                try:
                    beta, residuals, rank, s = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
                    y_eval[i] = beta[0]  # Intercept at x_target
                except np.linalg.LinAlgError:
                    # Fallback to weighted mean
                    y_eval[i] = np.sum(weights * y_neighbors)

            elif degree == 2:
                # Local quadratic: y = β₀ + β₁(x - x_target) + β₂(x - x_target)²
                A = np.vstack([np.ones(k_neighbors), x_centered, x_centered**2]).T

                W_sqrt = np.sqrt(weights)
                A_weighted = A * W_sqrt[:, None]
                y_weighted = y_neighbors * W_sqrt

                try:
                    beta, residuals, rank, s = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
                    y_eval[i] = beta[0]  # Intercept at x_target
                except np.linalg.LinAlgError:
                    # Fallback to weighted mean
                    y_eval[i] = np.sum(weights * y_neighbors)
            else:
                raise ValueError(f"Unsupported polynomial degree: {degree}")

        # Robust reweighting (except on last iteration)
        if iter_idx < n_iter - 1:
            # Compute residuals at observed points
            # Interpolate fitted values back to observed x-coordinates
            y_obs_fitted = np.interp(x_obs, x_eval, y_eval)
            residuals = y_obs - y_obs_fitted

            # Median absolute deviation for robust scale estimate
            mad = np.median(np.abs(residuals))
            if mad < 1e-10:
                # Perfect fit or all residuals zero
                break

            # Tukey's bisquare weighting function
            # c = 6 * MAD is a robust scale estimate
            scale = 6.0 * mad
            u = residuals / scale

            # Bisquare weights: (1 - u²)² for |u| < 1, else 0
            robust_weights = np.where(
                np.abs(u) < 1.0,
                (1.0 - u**2)**2,
                0.0
            )

    return y_eval


def impute_loess(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    frac: float = 0.1,
    n_iter: int = 2,
    degree: int = 1
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using LOESS (Locally Estimated Scatterplot Smoothing).

    Fits robust local polynomial regression on observed points only, then
    interpolates to full time grid and target sampling rate.

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds (typically 1000 Hz)
        obs_mask: (N,) boolean mask, True = observed, False = missing
        frac: Smoothing bandwidth as fraction of data (0.05-0.3 typical)
              Larger = smoother, smaller = more flexible
        n_iter: Number of robust reweighting iterations (1-5 typical)
                1 = standard LOESS, >1 = robust LOESS
        degree: Local polynomial degree (0, 1, or 2)
                0 = local constant (Nadaraya-Watson)
                1 = local linear (most common)
                2 = local quadratic (may overfit)

    Returns:
        dict with:
            - output_300Hz: (N, D) smoothed trajectory at native rate
            - output_1000Hz: (M, D) interpolated trajectory at target rate
            - mean: None (deterministic method)
            - var: None
            - extras: dict with LOESS parameters
    """
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D = X.shape
    M = len(t_out)

    # Validate inputs
    assert len(t) == N, f"Time array length {len(t)} != data length {N}"
    assert len(obs_mask) == N, f"Mask length {len(obs_mask)} != data length {N}"
    assert np.any(obs_mask), "No observed points in mask"
    assert 0 < frac <= 1.0, f"frac must be in (0, 1], got {frac}"
    assert degree in [0, 1, 2], f"degree must be 0, 1, or 2, got {degree}"
    assert n_iter >= 1, f"n_iter must be >= 1, got {n_iter}"

    n_obs = np.sum(obs_mask)
    k_neighbors = max(degree + 2, int(math.ceil(frac * n_obs)))

    # Store diagnostics
    diagnostics = {
        'frac': frac,
        'n_iter': n_iter,
        'degree': degree,
        'n_observed': n_obs,
        'n_total': N,
        'obs_fraction': n_obs / N,
        'k_neighbors': k_neighbors
    }

    out_300 = np.zeros((N, D), dtype=np.float64)
    out_1000 = np.zeros((M, D), dtype=np.float64)

    # Process each dimension independently
    for d in range(D):
        t_obs = t[obs_mask]
        y_obs = X[obs_mask, d]

        if len(t_obs) < degree + 1:
            warnings.warn(f"Insufficient observed points for degree {degree} LOESS in dim {d}")
            # Fallback: use mean
            y_mean = np.mean(y_obs)
            out_300[:, d] = y_mean
            out_1000[:, d] = y_mean
            continue

        # LOESS fit at native time grid
        y_fit = loess_1d(t_obs, y_obs, t, frac=frac, n_iter=n_iter, degree=degree)
        out_300[:, d] = y_fit

        # Interpolate to target grid using cubic spline
        try:
            cs = CubicSpline(t, y_fit, bc_type='natural', extrapolate=True)
            out_1000[:, d] = cs(t_out)
        except Exception as e:
            warnings.warn(f"Cubic spline failed for dim {d}: {e}")
            # Fallback: linear interpolation
            out_1000[:, d] = np.interp(t_out, t, y_fit)

    return {
        'output_300Hz': out_300,
        'output_1000Hz': out_1000,
        'mean': None,  # Deterministic method
        'var': None,
        'extras': diagnostics
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_loess(
    segments: list,
    frac: float = 0.1,
    n_iter: int = 2,
    degree: int = 1,
    config: ExperimentConfig = CONFIG
) -> Dict[str, float]:
    """
    Evaluate LOESS on all segments.

    Returns:
        dict of averaged metrics across segments
    """
    print("\n" + "="*80)
    print("EVALUATING: LOESS (Locally Weighted Regression)")
    print("="*80)
    print(f"Parameters: frac={frac:.3f}, n_iter={n_iter}, degree={degree}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        # Create output time grid at 1000 Hz
        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        # Run imputation
        result = impute_loess(
            X_true, t_seg, t_out, obs_mask,
            frac=frac, n_iter=n_iter, degree=degree
        )

        X_pred = result['output_300Hz']

        # Compute metrics on MISSING points only
        missing_mask = ~obs_mask
        if not np.any(missing_mask):
            print(f"  Segment {seg_id}: No missing points, skipping")
            continue

        metrics = compute_metrics(
            X_true, X_pred, y_var=None, mask=missing_mask,
            name=f"Segment_{seg_id}"
        )

        all_metrics.append(metrics)

        print(f"  Segment {seg_id} ({obs_mask.mean()*100:.1f}% obs): "
              f"MAE={metrics['MAE']:.3f} nm, "
              f"RMSE={metrics['RMSE']:.3f} nm, "
              f"R²={metrics['R2']:.4f}")

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
    print(f"  MAE:  {avg_metrics['MAE']:.3f} ± {avg_metrics['MAE_std']:.3f} nm")
    print(f"  RMSE: {avg_metrics['RMSE']:.3f} ± {avg_metrics['RMSE_std']:.3f} nm")
    print(f"  R²:   {avg_metrics['R2']:.4f} ± {avg_metrics['R2_std']:.4f}")
    print()

    return avg_metrics


def visualize_sample_result(
    segment: dict,
    frac: float = 0.1,
    n_iter: int = 2,
    degree: int = 1,
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_loess(
        X_true, t_seg, t_out, obs_mask,
        frac=frac, n_iter=n_iter, degree=degree
    )

    X_pred = result['output_300Hz']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=None,
        obs_mask=obs_mask,
        method_name=f"LOESS (frac={frac:.2f}, degree={degree})",
        filename="03_loess_result.png"
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
                X_var=None,
                obs_mask=obs_mask[zoom_start:zoom_end],
                method_name=f"LOESS (Zoom)",
                filename="03_loess_zoom.png"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data and segments from starter framework...")

    # Load from starter framework
    t, X, X_detrended, ou_params, segments = main_setup()

    # Test different bandwidth parameters
    print("\n" + "="*80)
    print("TESTING MULTIPLE BANDWIDTH PARAMETERS")
    print("="*80)

    results = {}

    # Test different fractions
    for frac in [0.05, 0.1, 0.15, 0.2, 0.3]:
        print(f"\n--- Bandwidth frac = {frac:.2f} ---")
        metrics = evaluate_loess(
            segments,
            frac=frac,
            n_iter=2,
            degree=1,
            config=CONFIG
        )
        results[f'frac_{frac:.2f}'] = metrics

    # Find best bandwidth
    best_frac = min(results.keys(), key=lambda k: results[k].get('MAE', np.inf))
    print("\n" + "="*80)
    print(f"BEST BANDWIDTH: {best_frac}")
    print("="*80)
    print(f"MAE:  {results[best_frac]['MAE']:.3f} nm")
    print(f"RMSE: {results[best_frac]['RMSE']:.3f} nm")
    print(f"R²:   {results[best_frac]['R2']:.4f}")

    # Test robustness iterations with best frac
    best_frac_val = float(best_frac.split('_')[1])

    print("\n" + "="*80)
    print(f"TESTING ROBUSTNESS ITERATIONS (frac={best_frac_val:.2f})")
    print("="*80)

    for n_iter in [1, 2, 3]:
        print(f"\n--- n_iter = {n_iter} ---")
        metrics = evaluate_loess(
            segments,
            frac=best_frac_val,
            n_iter=n_iter,
            degree=1,
            config=CONFIG
        )
        results[f'iter_{n_iter}'] = metrics

    # Visualize with best parameters
    print("\nGenerating visualization with best parameters...")
    visualize_sample_result(segments[0], frac=best_frac_val, n_iter=2, degree=1, config=CONFIG)

    print("\n✓ LOESS evaluation complete.")
