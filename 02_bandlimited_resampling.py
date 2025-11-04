#!/usr/bin/env python3
"""
Band-Limited Resampling (FIR Low-Pass + Polyphase Resampling)
==============================================================
Signal processing approach assuming band-limited signal.

Method:
1. Assume signal is band-limited to < Nyquist frequency (150 Hz at 300 Hz sampling)
2. Fill gaps temporarily with linear interpolation
3. Apply anti-aliasing low-pass FIR filter
4. Resample using polyphase filter bank (scipy.signal.resample_poly)
5. Back-sample to native grid for evaluation

Pros:
- Mathematically rigorous if signal truly band-limited
- No ringing if spectrum is clean

Cons:
- Linear interpolation of gaps can introduce high-frequency artifacts
- Assumes signal has no content > 150 Hz (may not hold with measurement noise)

References:
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-time signal processing (3rd ed.)
- Harris, F. J. (1978). On the use of windows for harmonic analysis with the DFT.
"""

import sys
import numpy as np
from scipy import signal, interpolate
from scipy.signal import firwin, filtfilt, resample_poly
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional
import warnings

# Add parent directory to path for imports
sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())


def impute_bandlimited(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    fs_in: float = 300.0,
    fs_out: float = 1000.0,
    lp_factor: float = 0.9,
    numtaps: int = 255
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using band-limited resampling with anti-aliasing filter.

    Assumes the signal is band-limited (no content above Nyquist frequency of input).
    Uses FIR low-pass filter followed by polyphase resampling.

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds (typically 1000 Hz)
        obs_mask: (N,) boolean mask, True = observed, False = missing
        fs_in: Input sampling frequency in Hz (default 300)
        fs_out: Output sampling frequency in Hz (default 1000)
        lp_factor: Low-pass cutoff as fraction of Nyquist (0.9 = 90% of Nyquist)
        numtaps: Number of FIR filter taps (default 255, must be odd for type I)

    Returns:
        dict with:
            - output_300Hz: (N, D) resampled trajectory at native rate
            - output_1000Hz: (M, D) resampled trajectory at target rate
            - mean: None (deterministic method)
            - var: None
            - extras: dict with filter parameters
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

    # Determine resampling ratio
    # For 300 → 1000: up=10, down=3
    ratio = fs_out / fs_in
    up = int(fs_out)
    down = int(fs_in)

    # Simplify ratio if possible
    from math import gcd
    g = gcd(up, down)
    up //= g
    down //= g

    # Design anti-aliasing low-pass filter
    # Cutoff frequency: lp_factor * min(Nyquist_in, Nyquist_out)
    # Normalized to Nyquist frequency of input (fs_in / 2)
    nyquist_in = 0.5 * fs_in
    nyquist_out = 0.5 * fs_out
    cutoff_hz = lp_factor * min(nyquist_in, nyquist_out)
    cutoff_normalized = cutoff_hz / nyquist_in  # Normalize to [0, 1]

    # Ensure numtaps is odd for symmetric (type I) FIR filter
    if numtaps % 2 == 0:
        numtaps += 1

    # Design FIR filter using Hamming window
    try:
        fir_taps = firwin(
            numtaps=numtaps,
            cutoff=cutoff_normalized,
            window='hamming',
            pass_zero='lowpass'
        )
    except Exception as e:
        warnings.warn(f"FIR filter design failed: {e}. Using default order.")
        fir_taps = firwin(numtaps=101, cutoff=cutoff_normalized, window='hamming')

    # Store diagnostics
    diagnostics = {
        'fs_in': fs_in,
        'fs_out': fs_out,
        'up': up,
        'down': down,
        'ratio': ratio,
        'cutoff_hz': cutoff_hz,
        'cutoff_normalized': cutoff_normalized,
        'numtaps': len(fir_taps),
        'n_observed': np.sum(obs_mask),
        'obs_fraction': np.mean(obs_mask)
    }

    out_300 = np.zeros((N, D), dtype=np.float64)
    out_1000 = np.zeros((M, D), dtype=np.float64)

    # Process each dimension independently
    for d in range(D):
        y = X[:, d].copy()

        # Step 1: Fill gaps temporarily with linear interpolation
        # This is necessary because FIR filter requires continuous data
        t_obs = t[obs_mask]
        y_obs = y[obs_mask]

        if len(t_obs) < 2:
            # Not enough observed points - use constant
            warnings.warn(f"Insufficient observed points for dim {d}")
            out_300[:, d] = y_obs[0] if len(y_obs) > 0 else 0.0
            out_1000[:, d] = y_obs[0] if len(y_obs) > 0 else 0.0
            continue

        # Linear interpolation to fill gaps
        interp_func = interpolate.interp1d(
            t_obs, y_obs,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        y_filled = interp_func(t)

        # Step 2: Apply anti-aliasing low-pass filter
        # Use filtfilt for zero-phase filtering (no time shift)
        try:
            y_filtered = filtfilt(fir_taps, [1.0], y_filled)
        except Exception as e:
            warnings.warn(f"filtfilt failed for dim {d}: {e}. Using unfiltered data.")
            y_filtered = y_filled

        # Step 3: Resample using polyphase filter
        # resample_poly uses a polyphase FIR filter for efficient resampling
        try:
            y_resampled = resample_poly(y_filtered, up=up, down=down, axis=0)
        except Exception as e:
            warnings.warn(f"resample_poly failed for dim {d}: {e}")
            # Fallback: simple interpolation
            t_temp = np.arange(len(y_filtered)) / fs_in
            t_resamp = np.arange(int(t[-1] * fs_out) + 1) / fs_out
            y_resampled = np.interp(t_resamp, t_temp, y_filtered)

        # Step 4: Truncate or pad to match exact output length
        if len(y_resampled) > M:
            y_resampled = y_resampled[:M]
        elif len(y_resampled) < M:
            # Pad with last value
            pad_len = M - len(y_resampled)
            y_resampled = np.concatenate([y_resampled, np.full(pad_len, y_resampled[-1])])

        out_1000[:, d] = y_resampled

        # Step 5: Back-sample to native 300 Hz grid for evaluation
        # Use cubic spline interpolation
        try:
            cs = CubicSpline(t_out, y_resampled, bc_type='natural', extrapolate=True)
            out_300[:, d] = cs(t)
        except Exception as e:
            warnings.warn(f"Back-sampling failed for dim {d}: {e}")
            # Fallback: linear interpolation
            out_300[:, d] = np.interp(t, t_out, y_resampled)

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

def evaluate_bandlimited(
    segments: list,
    lp_factor: float = 0.9,
    numtaps: int = 255,
    config: ExperimentConfig = CONFIG
) -> Dict[str, float]:
    """
    Evaluate band-limited resampling on all segments.

    Returns:
        dict of averaged metrics across segments
    """
    print("\n" + "="*80)
    print("EVALUATING: Band-Limited Resampling (FIR + Polyphase)")
    print("="*80)
    print(f"Parameters: lp_factor={lp_factor:.2f}, numtaps={numtaps}")
    print(f"  Cutoff: {lp_factor * min(0.5*config.fs_in, 0.5*config.fs_out):.1f} Hz")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        # Create output time grid at 1000 Hz
        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        # Run imputation
        result = impute_bandlimited(
            X_true, t_seg, t_out, obs_mask,
            fs_in=config.fs_in, fs_out=config.fs_out,
            lp_factor=lp_factor, numtaps=numtaps
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
    lp_factor: float = 0.9,
    numtaps: int = 255,
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_bandlimited(
        X_true, t_seg, t_out, obs_mask,
        fs_in=config.fs_in, fs_out=config.fs_out,
        lp_factor=lp_factor, numtaps=numtaps
    )

    X_pred = result['output_300Hz']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=None,
        obs_mask=obs_mask,
        method_name="Band-Limited Resampling",
        filename="02_bandlimited_result.png"
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
                method_name="Band-Limited Resampling (Zoom)",
                filename="02_bandlimited_zoom.png"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data and segments from starter framework...")

    # Load from starter framework
    t, X, X_detrended, ou_params, segments = main_setup()

    # Test different filter parameters
    print("\n" + "="*80)
    print("TESTING MULTIPLE CUTOFF FREQUENCIES")
    print("="*80)

    results = {}

    for lp_factor in [0.7, 0.8, 0.9, 0.95]:
        print(f"\n--- Low-pass factor = {lp_factor:.2f} ---")
        metrics = evaluate_bandlimited(
            segments,
            lp_factor=lp_factor,
            numtaps=255,
            config=CONFIG
        )
        results[f'lp_{lp_factor:.2f}'] = metrics

    # Find best cutoff
    best_lp = min(results.keys(), key=lambda k: results[k].get('MAE', np.inf))
    print("\n" + "="*80)
    print(f"BEST LOW-PASS FACTOR: {best_lp}")
    print("="*80)
    print(f"MAE:  {results[best_lp]['MAE']:.3f} nm")
    print(f"RMSE: {results[best_lp]['RMSE']:.3f} nm")
    print(f"R²:   {results[best_lp]['R2']:.4f}")

    # Visualize with best parameters
    best_lp_val = float(best_lp.split('_')[1])
    print("\nGenerating visualization with best parameters...")
    visualize_sample_result(segments[0], lp_factor=best_lp_val, numtaps=255, config=CONFIG)

    print("\n✓ Band-Limited Resampling evaluation complete.")
