#!/usr/bin/env python3
"""
Savitzky-Golay + Cubic Spline Interpolation
============================================
Fast deterministic baseline using polynomial smoothing and spline interpolation.

Method:
1. Detrend observed data with polynomial fit
2. Fill gaps temporarily with linear interpolation
3. Apply Savitzky-Golay filter for noise reduction
4. Add trend back
5. Fit cubic spline through observed smoothed points
6. Interpolate to target sampling rate

Reference:
- Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by
  simplified least squares procedures. Analytical chemistry, 36(8), 1627-1639.
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


def impute_savgol_spline(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    poly: int = 3,
    window_ms: float = 11.0,
    detrend_poly: int = 1,
    fs: float = 300.0
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using Savitzky-Golay smoothing + cubic spline interpolation.

    This is a fast, deterministic baseline that:
    - Removes low-frequency drift via polynomial detrending
    - Reduces measurement noise via local polynomial regression (Savitzky-Golay)
    - Interpolates gaps via cubic splines fitted to smoothed observed points

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds (typically 1000 Hz)
        obs_mask: (N,) boolean mask, True = observed, False = missing
        poly: Polynomial order for Savitzky-Golay filter (1-5, default 3)
        window_ms: Window length in milliseconds for Savitzky-Golay (default 11 ms)
        detrend_poly: Polynomial degree for detrending (0=none, 1=linear, 2=quadratic)
        fs: Sampling frequency of input data in Hz (default 300)

    Returns:
        dict with:
            - output_300Hz: (N, D) smoothed trajectory at native rate
            - output_1000Hz: (M, D) interpolated trajectory at target rate
            - mean: None (deterministic method, no uncertainty)
            - var: None
            - extras: dict with filter parameters and diagnostics
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

    # Compute Savitzky-Golay window length (must be odd, >= poly+2)
    window_samples = int(round(window_ms * 1e-3 * fs))
    if window_samples % 2 == 0:
        window_samples += 1  # Make odd
    window_samples = max(poly + 2, window_samples)  # Ensure window >= poly + 2
    window_samples = min(window_samples, N)  # Don't exceed data length

    # Ensure polynomial order is valid
    poly = max(1, min(poly, 5))  # Constrain to [1, 5]
    poly = min(poly, window_samples - 1)  # Must be less than window

    # Ensure detrend polynomial is valid
    n_obs = np.sum(obs_mask)
    detrend_poly = max(0, min(detrend_poly, n_obs - 1))

    out_300 = np.zeros((N, D), dtype=np.float64)
    out_1000 = np.zeros((M, D), dtype=np.float64)

    # Store diagnostics
    diagnostics = {
        'window_samples': window_samples,
        'poly_order': poly,
        'detrend_poly': detrend_poly,
        'n_observed': n_obs,
        'n_total': N,
        'obs_fraction': n_obs / N
    }

    # Process each dimension independently
    for d in range(D):
        y = X[:, d].copy()
        t_obs = t[obs_mask]
        y_obs = y[obs_mask]

        # Step 1: Detrend observed data
        if detrend_poly > 0 and n_obs > detrend_poly:
            # Fit polynomial to observed points only
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                trend_coeffs = np.polyfit(t_obs, y_obs, deg=detrend_poly)

            # Evaluate trend on full time grid
            trend_full = np.polyval(trend_coeffs, t)
            trend_obs = np.polyval(trend_coeffs, t_obs)

            # Detrend
            y_obs_detrended = y_obs - trend_obs
        else:
            trend_full = np.zeros(N)
            y_obs_detrended = y_obs.copy()

        # Step 2: Fill gaps temporarily for Savitzky-Golay
        # (SG filter requires continuous data, gaps filled with linear interp)
        if n_obs < N:
            # Use observed points to fill missing via linear interpolation
            interp_func = interpolate.interp1d(
                t_obs, y_obs_detrended,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            y_filled = interp_func(t)
        else:
            # No gaps, use observed data directly
            y_filled = y_obs_detrended.copy()

        # Handle edge case: if window is too large for data
        effective_window = min(window_samples, len(y_filled))
        if effective_window % 2 == 0:
            effective_window -= 1
        effective_window = max(poly + 2, effective_window)
        effective_poly = min(poly, effective_window - 1)

        # Step 3: Apply Savitzky-Golay filter
        try:
            y_smoothed = signal.savgol_filter(
                y_filled,
                window_length=effective_window,
                polyorder=effective_poly,
                mode='interp'  # Polynomial extrapolation at boundaries
            )
        except Exception as e:
            warnings.warn(f"Savitzky-Golay failed for dim {d}: {e}. Using unsmoothed data.")
            y_smoothed = y_filled.copy()

        # Step 4: Add trend back
        y_smoothed_with_trend = y_smoothed + trend_full

        # Step 5: Fit cubic spline through OBSERVED points of smoothed data
        # Important: only use observed points for spline to avoid overfitting to
        # linearly interpolated gaps
        y_smoothed_obs = y_smoothed_with_trend[obs_mask]

        # Cubic spline requires at least 2 points
        if n_obs >= 2:
            # Check for sufficient spacing (avoid duplicate time points)
            unique_mask = np.concatenate([[True], np.diff(t_obs) > 1e-10])
            t_obs_unique = t_obs[unique_mask]
            y_obs_unique = y_smoothed_obs[unique_mask]

            if len(t_obs_unique) >= 2:
                try:
                    cs = CubicSpline(
                        t_obs_unique,
                        y_obs_unique,
                        bc_type='natural',  # Natural boundary conditions (2nd deriv = 0)
                        extrapolate=True
                    )

                    # Evaluate on native and target grids
                    out_300[:, d] = cs(t)
                    out_1000[:, d] = cs(t_out)

                except Exception as e:
                    warnings.warn(f"CubicSpline failed for dim {d}: {e}. Using linear interpolation.")
                    # Fallback to linear interpolation
                    out_300[:, d] = np.interp(t, t_obs_unique, y_obs_unique)
                    out_1000[:, d] = np.interp(t_out, t_obs_unique, y_obs_unique)
            else:
                warnings.warn(f"Insufficient unique points for dim {d}. Using constant extrapolation.")
                out_300[:, d] = y_obs_unique[0]
                out_1000[:, d] = y_obs_unique[0]
        else:
            # Fallback: use single observed point (constant extrapolation)
            warnings.warn(f"Only {n_obs} observed point(s) for dim {d}. Using constant.")
            out_300[:, d] = y_smoothed_obs[0] if n_obs == 1 else np.nan
            out_1000[:, d] = y_smoothed_obs[0] if n_obs == 1 else np.nan

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

def evaluate_savgol_spline(
    segments: list,
    poly: int = 3,
    window_ms: float = 11.0,
    detrend_poly: int = 1,
    config: ExperimentConfig = CONFIG
) -> Dict[str, float]:
    """
    Evaluate Savitzky-Golay + Spline method on all segments.

    Returns:
        dict of averaged metrics across segments
    """
    print("\n" + "="*80)
    print("EVALUATING: Savitzky-Golay + Cubic Spline")
    print("="*80)
    print(f"Parameters: poly={poly}, window={window_ms:.1f}ms, detrend_poly={detrend_poly}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        # Create output time grid at 1000 Hz
        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        # Run imputation
        result = impute_savgol_spline(
            X_true, t_seg, t_out, obs_mask,
            poly=poly, window_ms=window_ms,
            detrend_poly=detrend_poly, fs=config.fs_in
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
    poly: int = 3,
    window_ms: float = 11.0,
    detrend_poly: int = 1,
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_savgol_spline(
        X_true, t_seg, t_out, obs_mask,
        poly=poly, window_ms=window_ms,
        detrend_poly=detrend_poly, fs=config.fs_in
    )

    X_pred = result['output_300Hz']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=None,
        obs_mask=obs_mask,
        method_name="Savitzky-Golay + Cubic Spline",
        filename="01_savgol_spline_result.png"
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
                method_name="Savitzky-Golay + Cubic Spline (Zoom)",
                filename="01_savgol_spline_zoom.png"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data and segments from starter framework...")

    # Load from starter framework
    t, X, X_detrended, ou_params, segments = main_setup()

    # Evaluate with default parameters
    metrics_default = evaluate_savgol_spline(
        segments,
        poly=3,
        window_ms=11.0,
        detrend_poly=1,
        config=CONFIG
    )

    # Try more aggressive smoothing
    print("\n" + "="*80)
    print("Testing with stronger smoothing (window=25ms)")
    print("="*80)

    metrics_smooth = evaluate_savgol_spline(
        segments,
        poly=3,
        window_ms=25.0,
        detrend_poly=2,
        config=CONFIG
    )

    # Visualize on first segment
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], poly=3, window_ms=11.0, detrend_poly=1, config=CONFIG)

    print("\n✓ Savitzky-Golay + Cubic Spline evaluation complete.")
    print(f"  Default: MAE={metrics_default.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics_default.get('RMSE', np.nan):.3f} nm")
    print(f"  Smooth:  MAE={metrics_smooth.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics_smooth.get('RMSE', np.nan):.3f} nm")
