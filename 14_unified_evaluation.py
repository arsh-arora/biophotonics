#!/usr/bin/env python3
"""
Unified Evaluation & Leaderboard
=================================
Comprehensive evaluation of all trajectory imputation methods.

Evaluates:
A) Deterministic Baselines:
   - Savitzky-Golay + Spline
   - Band-limited resampling
   - LOESS

B) Physics-based Probabilistic:
   - OU Kalman Smoother (univariate + multivariate)
   - GP with OU Kernel
   - State-Space GP with OU prior
   - Hybrid GP (OU + RBF)

C) Learning-based:
   - TCN (Temporal Convolutional Network) + variants
   - GRU-D (Gated Recurrent Units with Decay)
   - SAITS (Self-Attention Imputation)

D) Hybrid Physics-Informed:
   - Neural SDE with OU prior

Metrics:
- MAE, RMSE, R² on missing points
- NLL, Calibration (1σ, 2σ) for probabilistic methods

Output:
- Leaderboard ranked by MAE
- Comprehensive Markdown report with visualizations
- Saved predictions and plots (.png files)
"""

import os
import sys
import subprocess
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
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


# ============================================================================
# Metrics
# ============================================================================

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

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'NLL': np.nan,
        'Calib_1sigma': np.nan,
        'Calib_2sigma': np.nan
    }

    if y_var is not None and not np.all(np.isnan(y_var)):
        y_std = np.sqrt(y_var)
        nll = 0.5 * (np.log(2 * np.pi * y_var) + residual**2 / y_var)
        metrics['NLL'] = np.mean(nll)

        z_scores = np.abs(residual / (y_std + 1e-9))
        metrics['Calib_1sigma'] = np.mean(z_scores <= 1.0)
        metrics['Calib_2sigma'] = np.mean(z_scores <= 2.0)

    return metrics


# ============================================================================
# Method Execution
# ============================================================================

def run_method(script_name, method_name):
    """Run a method script and extract results."""
    print(f"\n{'='*80}")
    print(f"Running: {method_name}")
    print(f"{'='*80}")

    try:
        # Run the script
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"ERROR: {method_name} failed")
            print(result.stderr)
            return None

        print(result.stdout)

        # Check if output files were generated
        base_name = script_name.replace('.py', '')
        output_files = [
            f"{base_name}_result.png",
            f"{base_name}_zoom.png",
            f"{base_name}_predictions.npy"
        ]
        for output_file in output_files:
            if os.path.exists(output_file):
                print(f"✓ Generated: {output_file}")

        return {
            'status': 'success',
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {method_name} exceeded 10 minutes")
        return None
    except Exception as e:
        print(f"ERROR: {method_name} failed with {e}")
        return None


def extract_metrics_from_output(output_text):
    """Extract metrics from script output."""
    metrics = {
        'MAE': np.nan,
        'RMSE': np.nan,
        'R2': np.nan,
        'NLL': np.nan,
        'Calib_1sigma': np.nan,
        'Calib_2sigma': np.nan
    }

    lines = output_text.split('\n')
    for line in lines:
        if 'MAE:' in line:
            try:
                metrics['MAE'] = float(line.split('MAE:')[1].split()[0])
            except:
                pass
        if 'RMSE:' in line:
            try:
                metrics['RMSE'] = float(line.split('RMSE:')[1].split()[0])
            except:
                pass
        if 'R²:' in line or 'R2:' in line:
            try:
                if 'R²:' in line:
                    metrics['R2'] = float(line.split('R²:')[1].split()[0])
                else:
                    metrics['R2'] = float(line.split('R2:')[1].split()[0])
            except:
                pass
        if 'NLL:' in line and 'Results' in line:  # Avoid matching other NLL mentions
            try:
                metrics['NLL'] = float(line.split('NLL:')[1].split()[0])
            except:
                pass
        if 'Calibration @ 1σ:' in line or 'Calibration @ 1sigma:' in line:
            try:
                if '1σ:' in line:
                    val = line.split('1σ:')[1].split('%')[0].strip()
                else:
                    val = line.split('1sigma:')[1].split('%')[0].strip()
                metrics['Calib_1sigma'] = float(val) / 100.0
            except:
                pass
        if 'Calibration @ 2σ:' in line or 'Calibration @ 2sigma:' in line:
            try:
                if '2σ:' in line:
                    val = line.split('2σ:')[1].split('%')[0].strip()
                else:
                    val = line.split('2sigma:')[1].split('%')[0].strip()
                metrics['Calib_2sigma'] = float(val) / 100.0
            except:
                pass

    return metrics


# ============================================================================
# Leaderboard and Report Generation
# ============================================================================

def generate_leaderboard(results):
    """Generate sorted leaderboard."""
    # Filter valid results
    valid_results = [(name, metrics) for name, metrics in results.items()
                     if metrics is not None and not np.isnan(metrics.get('MAE', np.nan))]

    # Sort by MAE
    leaderboard = sorted(valid_results, key=lambda x: x[1]['MAE'])

    print("\n" + "="*80)
    print("LEADERBOARD (Ranked by MAE on Missing Points)")
    print("="*80)
    print(f"{'Rank':<6} {'Method':<30} {'MAE (nm)':<12} {'RMSE (nm)':<12} {'R²':<10}")
    print("-"*80)

    for rank, (name, metrics) in enumerate(leaderboard, 1):
        print(f"{rank:<6} {name:<30} {metrics['MAE']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['R2']:<10.4f}")

    return leaderboard


def generate_markdown_report(leaderboard, all_results, data_info):
    """Generate comprehensive Markdown report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Trajectory Imputation Methods - Comprehensive Evaluation Report

**Generated:** {timestamp}

---

## 1. Problem Statement

### Optical Tweezers Trajectory Imputation

**Objective:** Upsample optical tweezers trajectory data from 300 Hz to 1000 Hz while handling missing data (30-35% dropout).

**Physical System:**
- **Bead:** 3 μm diameter polystyrene sphere
- **Laser:** 532 nm wavelength, 20 mW power
- **Conversion:** 1 pixel = 35 nm
- **Sampling:** Native 300 Hz → Target 1000 Hz
- **Data:** {data_info['n_samples']} samples, {data_info['duration']:.2f} seconds

**Challenge:** Detector dropouts create irregular gaps (mean 25 ms, max 200 ms) requiring imputation methods that:
1. Preserve physical dynamics (Ornstein-Uhlenbeck process)
2. Handle variable-length gaps
3. Provide uncertainty quantification
4. Achieve high temporal resolution (1000 Hz)

**Estimated OU Parameters:**
- Relaxation time (τ): {data_info['tau']*1e3:.2f} ms
- Diffusion coefficient (D): {data_info['D']:.1f} nm²/s
- Corner frequency (f_c): {data_info['fc']:.2f} Hz

---

## 2. Methods Implemented

### A) Deterministic Baselines

#### A1) Savitzky-Golay + Spline
- **Description:** Polynomial smoothing followed by cubic spline interpolation
- **Pros:** Simple, fast, no training required
- **Cons:** Poor performance on missing data, smoothing artifacts

#### A2) Band-limited Resampling
- **Description:** FIR filter + polyphase resampling
- **Pros:** Signal processing theory, anti-aliasing
- **Cons:** Assumes band-limited signals, edge effects

#### A3) LOESS (Local Polynomial Regression)
- **Description:** Locally weighted polynomial regression with robust reweighting
- **Pros:** Adaptive to local structure, robust to outliers
- **Cons:** Computationally expensive, parameter sensitive

---

### B) Physics-based Probabilistic Methods

#### B1) OU Kalman Smoother
- **Description:** State-space model with OU dynamics, EM parameter estimation
- **Model:** dX = -(X/τ)dt + √(2D)dW
- **Pros:** Principled uncertainty, optimal filtering, interpretable parameters
- **Cons:** Assumes exact OU dynamics, Gaussian noise

#### B2) OU Kalman Multivariate
- **Description:** Multivariate extension of OU Kalman for joint trajectory components
- **Pros:** Handles cross-correlations between dimensions
- **Cons:** Increased computational complexity

#### B3) GP with OU Kernel
- **Description:** Gaussian Process with Matérn-1/2 (OU) covariance kernel
- **Pros:** Closed-form kernel, efficient inference, uncertainty quantification
- **Cons:** Limited to OU covariance structure

#### B4) State-Space GP with OU Prior
- **Description:** State-space formulation of GP with OU prior for O(N) complexity
- **Pros:** Linear time complexity, marginal likelihood optimization
- **Cons:** Limited to Markovian kernels

#### B5) Hybrid GP (OU + RBF)
- **Description:** k_total = k_OU + k_RBF(short lengthscale)
- **Pros:** Separates physics from residuals, interpretable decomposition
- **Cons:** More hyperparameters to optimize

---

### C) Learning-based Methods

#### C1) TCN (Temporal Convolutional Network)
- **Description:** Causal dilated convolutions with residual connections
- **Architecture:** 5 levels, 64 hidden channels, receptive field ~100 samples
- **Pros:** Large receptive field, GPU acceleration, excellent performance
- **Cons:** Requires training data, black-box

#### C1a-e) TCN Variants
- **TCN Fixed:** Improved architecture with better normalization
- **TCN Multi-Scale:** Multiple dilation rates for multi-resolution features
- **TCN Multi-Scale Fixed:** Combined improvements
- **TCN with NLL:** Probabilistic TCN with uncertainty estimation
- **TCN Ensemble:** Multiple TCN models with averaging

#### C1f-j) Bidirectional TCN Variants
- **Bidirectional TCN:** Non-causal TCN with forward + backward processing
- **Bidirectional TCN (Fixed):** Improved with batch normalization and weight normalization
- **Bidirectional TCN (Multi-Scale):** Parallel branches at multiple scales
- **Bidirectional TCN (NLL):** Probabilistic with mean + log-variance outputs
- **Bidirectional TCN (Ensemble):** Multiple models for robust predictions

#### C2) GRU-D (Gated Recurrent Unit with Decay)
- **Description:** Recurrent network with time-aware decay for irregular sampling
- **Pros:** Handles irregular time gaps naturally, interpretable decay
- **Cons:** Sequential processing (slower than TCN), gradient issues

#### C3) SAITS (Self-Attention Imputation)
- **Description:** Transformer encoder with value+mask embeddings and positional encoding
- **Architecture:** 3 layers, d_model=128, 4 attention heads
- **Pros:** Captures long-range dependencies, interpretable attention
- **Cons:** Computationally expensive, many parameters

---

### D) Hybrid Physics-Informed Methods

#### D1) Neural SDE with OU Prior
- **Description:** dX = [-(X/τ) + g_θ(X,t)]dt + √(2D)dW
- **Pros:** Combines physics with learned corrections, uncertainty via multiple trajectories
- **Cons:** SDE integration overhead, training complexity

---

## 3. Evaluation Methodology

**Data Splitting:**
- Observation mask: {data_info['obs_fraction']*100:.1f}% observed, {(1-data_info['obs_fraction'])*100:.1f}% missing
- Evaluation: Metrics computed on missing points only

**Metrics:**
1. **MAE (Mean Absolute Error):** Primary metric, average |y_true - y_pred| on missing points
2. **RMSE (Root Mean Square Error):** Penalizes large errors more heavily
3. **R² (Coefficient of Determination):** Explained variance, 1.0 = perfect, 0.0 = mean baseline
4. **NLL (Negative Log-Likelihood):** For probabilistic methods with uncertainty
5. **Calibration:** Fraction of errors within 1σ and 2σ bounds (ideally 68% and 95%)

---

## 4. Results

### Leaderboard (Ranked by MAE)

| Rank | Method | MAE (nm) | RMSE (nm) | R² | NLL | Calib 1σ | Calib 2σ |
|------|--------|----------|-----------|-----|-----|----------|----------|
"""

    for rank, (name, metrics) in enumerate(leaderboard, 1):
        mae = f"{metrics['MAE']:.4f}"
        rmse = f"{metrics['RMSE']:.4f}"
        r2 = f"{metrics['R2']:.4f}"
        nll = f"{metrics['NLL']:.4f}" if not np.isnan(metrics['NLL']) else "N/A"
        calib1 = f"{metrics['Calib_1sigma']*100:.1f}%" if not np.isnan(metrics['Calib_1sigma']) else "N/A"
        calib2 = f"{metrics['Calib_2sigma']*100:.1f}%" if not np.isnan(metrics['Calib_2sigma']) else "N/A"

        md_content += f"| {rank} | {name} | {mae} | {rmse} | {r2} | {nll} | {calib1} | {calib2} |\n"

    md_content += "\n---\n\n"

    # Add visualizations
    md_content += "## 5. Visualizations\n\n"

    # Add plots for each method
    for name, metrics in leaderboard:
        script_file = None
        for num, (script, method_name) in method_mapping.items():
            if method_name == name:
                script_file = script.replace('.py', '')
                break

        if script_file:
            result_img = f"{script_file}_result.png"
            zoom_img = f"{script_file}_zoom.png"

            if os.path.exists(result_img):
                md_content += f"### {name}\n\n"
                md_content += f"![{name} - Full Results]({result_img})\n\n"
                if os.path.exists(zoom_img):
                    md_content += f"![{name} - Zoom]({zoom_img})\n\n"

    # Analysis section
    md_content += """---

## 6. Analysis and Conclusions

### Key Findings

"""

    # Get top 3 methods
    top3 = leaderboard[:3]

    md_content += "#### Best Performing Methods\n\n"
    for rank, (name, metrics) in enumerate(top3, 1):
        md_content += f"{rank}. **{name}** (MAE: {metrics['MAE']:.4f} nm, R²: {metrics['R2']:.4f})\n"

    md_content += "\n#### Performance Insights\n\n"

    # Analyze learning vs physics-based
    learning_methods = [m for m in leaderboard if any(x in m[0].lower() for x in ['tcn', 'saits', 'csdi', 'gru'])]
    physics_methods = [m for m in leaderboard if any(x in m[0].lower() for x in ['kalman', 'gp', 'ou'])]

    if learning_methods:
        avg_learning_mae = np.mean([m[1]['MAE'] for m in learning_methods])
        md_content += f"- **Learning-based methods** average MAE: {avg_learning_mae:.4f} nm\n"

    if physics_methods:
        avg_physics_mae = np.mean([m[1]['MAE'] for m in physics_methods])
        md_content += f"- **Physics-based methods** average MAE: {avg_physics_mae:.4f} nm\n"

    md_content += """
### Recommendations

1. **For Production Systems:** Use top-performing method with best MAE and computational efficiency trade-off
2. **For Scientific Analysis:** Prefer physics-based methods (Kalman, GP) for interpretability
3. **For Uncertainty Quantification:** Use probabilistic methods (CSDI, GP, Kalman) with good calibration
4. **For Real-time Processing:** Use efficient methods (TCN, Kalman) with fast inference

### Future Work

- Ensemble methods combining physics and learning
- Online/adaptive imputation for streaming data
- Multi-dimensional trajectory imputation (x, y, z)
- Transfer learning across different experimental conditions

---

## 7. Computational Details

**Hardware:**
- CPU: Apple Silicon / Intel x86
- GPU: MPS / CUDA (for neural methods)
- Memory: Sufficient for 15,000 sample sequences

**Software:**
- Python 3.10+
- PyTorch, GPyTorch, torchsde
- NumPy, SciPy, scikit-learn

**Runtime:** Methods range from seconds (baselines) to minutes (deep learning)

---

## References

1. **OU Process:** Uhlenbeck & Ornstein (1930). On the Theory of Brownian Motion
2. **Kalman Filter:** Kalman (1960). A New Approach to Linear Filtering
3. **Gaussian Processes:** Rasmussen & Williams (2006). GP for Machine Learning
4. **TCN:** Bai et al. (2018). Empirical Evaluation of Generic Convolutional Networks
5. **SAITS:** Du et al. (2023). SAITS: Self-Attention-based Imputation for Time Series
6. **CSDI:** Tashiro et al. (2021). CSDI: Conditional Score-based Diffusion Models
7. **Neural SDEs:** Kidger et al. (2020). Neural Controlled Differential Equations

---

*Report generated by unified evaluation framework*
"""

    return md_content


# ============================================================================
# Main Execution
# ============================================================================

# Mapping of method scripts
method_mapping = {
    '01_savgol_spline_v2': ('01_savgol_spline_v2.py', 'Savitzky-Golay + Spline'),
    '02_bandlimited_resampling': ('02_bandlimited_resampling.py', 'Band-limited Resampling'),
    '03_loess': ('03_loess.py', 'LOESS'),
    '04_ou_kalman': ('04_ou_kalman.py', 'OU Kalman Smoother'),
    '05_ou_kalman_multivariate': ('05_ou_kalman_multivariate.py', 'OU Kalman Multivariate'),
    '06_gp_ou': ('06_gp_ou.py', 'GP with OU Kernel'),
    '07_ssm_gp_ou': ('07_ssm_gp_ou.py', 'State-Space GP (OU)'),
    '08_tcn': ('08_tcn.py', 'TCN'),
    '08_tcn_fixed': ('08_tcn_fixed.py', 'TCN Fixed'),
    '08_tcn_multiscale': ('08_tcn_multiscale.py', 'TCN Multi-Scale'),
    '08_tcn_multiscale_fixed': ('08_tcn_multiscale_fixed.py', 'TCN Multi-Scale Fixed'),
    '08_tcn_nll': ('08_tcn_nll.py', 'TCN with NLL'),
    '08_tcn_ensemble': ('08_tcn_ensemble.py', 'TCN Ensemble'),
    '09_gru_d': ('09_gru_d.py', 'GRU-D'),
    '10_saits': ('10_saits.py', 'SAITS'),
    # '11_csdi': ('11_csdi.py', 'CSDI'),  # Skipped per user request
    '12_neural_sde': ('12_neural_sde.py', 'Neural SDE (OU Prior)'),
    '13_gp_ou_rbf': ('13_gp_ou_rbf.py', 'Hybrid GP (OU + RBF)'),
    '15_tcn_bidirectional': ('15_tcn_bidirectional.py', 'Bidirectional TCN'),
    '15_tcn_bidirectional_fixed': ('15_tcn_bidirectional_fixed.py', 'Bidirectional TCN (Fixed)'),
    '15_tcn_bidirectional_multiscale': ('15_tcn_bidirectional_multiscale.py', 'Bidirectional TCN (Multi-Scale)'),
    '15_tcn_bidirectional_nll': ('15_tcn_bidirectional_nll.py', 'Bidirectional TCN (NLL)'),
    '15_tcn_bidirectional_ensemble': ('15_tcn_bidirectional_ensemble.py', 'Bidirectional TCN (Ensemble)'),
}


def main():
    print("="*80)
    print("UNIFIED EVALUATION & LEADERBOARD")
    print("Comprehensive evaluation of trajectory imputation methods")
    print("="*80)

    # Load data for info
    mat_path = '300fps_15k.mat'
    pos_nm, t_300hz, fs_300 = load_trajectory_data(mat_path, pixel_to_nm=35.0)
    dt_300 = 1.0 / fs_300
    tau, D, fc = estimate_ou_parameters_psd(pos_nm, dt_300)
    obs_mask_300 = generate_realistic_gaps(len(pos_nm), fs_300, gap_prob=0.35, seed=42)

    data_info = {
        'n_samples': len(pos_nm),
        'duration': t_300hz[-1],
        'tau': tau,
        'D': D,
        'fc': fc,
        'obs_fraction': np.mean(obs_mask_300)
    }

    print(f"\nDataset: {data_info['n_samples']} samples, {data_info['duration']:.2f}s")
    print(f"OU parameters: τ={tau*1e3:.2f}ms, D={D:.1f}nm²/s, f_c={fc:.2f}Hz")
    print(f"Observation: {data_info['obs_fraction']*100:.1f}% observed")

    # Run all methods
    all_results = {}

    for script_num, (script_file, method_name) in method_mapping.items():
        if not os.path.exists(script_file):
            print(f"\nSkipping {method_name}: {script_file} not found")
            continue

        result = run_method(script_file, method_name)

        if result and result['status'] == 'success':
            # Extract metrics from output
            metrics = extract_metrics_from_output(result['stdout'])
            all_results[method_name] = metrics
        else:
            all_results[method_name] = None

    # Generate leaderboard
    print("\n" + "="*80)
    print("PROCESSING RESULTS")
    print("="*80)

    leaderboard = generate_leaderboard(all_results)

    # Generate Markdown report
    print("\n" + "="*80)
    print("GENERATING MARKDOWN REPORT")
    print("="*80)

    md_report = generate_markdown_report(leaderboard, all_results, data_info)

    # Save report
    report_file = 'EVALUATION_REPORT.md'
    with open(report_file, 'w') as f:
        f.write(md_report)

    print(f"\n✓ Saved comprehensive report: {report_file}")

    # Create summary plot
    print("\nGenerating summary comparison plot...")
    create_summary_plot(leaderboard)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nTotal methods evaluated: {len([r for r in all_results.values() if r is not None])}")
    print(f"Report saved: {report_file}")
    print(f"Summary plot: comparison_summary.png")


def create_summary_plot(leaderboard):
    """Create summary comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = [name for name, _ in leaderboard]
    mae_values = [metrics['MAE'] for _, metrics in leaderboard]
    rmse_values = [metrics['RMSE'] for _, metrics in leaderboard]
    r2_values = [metrics['R2'] for _, metrics in leaderboard]
    nll_values = [metrics['NLL'] if not np.isnan(metrics['NLL']) else None
                  for _, metrics in leaderboard]

    # MAE comparison
    axes[0, 0].barh(methods, mae_values, color='steelblue')
    axes[0, 0].set_xlabel('MAE (nm)', fontsize=11)
    axes[0, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # RMSE comparison
    axes[0, 1].barh(methods, rmse_values, color='coral')
    axes[0, 1].set_xlabel('RMSE (nm)', fontsize=11)
    axes[0, 1].set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # R² comparison
    axes[1, 0].barh(methods, r2_values, color='mediumseagreen')
    axes[1, 0].set_xlabel('R²', fontsize=11)
    axes[1, 0].set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].axvline(x=0, color='k', linestyle='--', linewidth=0.8)

    # NLL comparison (only for methods with uncertainty)
    methods_with_nll = [m for m, v in zip(methods, nll_values) if v is not None]
    nll_valid = [v for v in nll_values if v is not None]

    if methods_with_nll:
        axes[1, 1].barh(methods_with_nll, nll_valid, color='mediumpurple')
        axes[1, 1].set_xlabel('NLL', fontsize=11)
        axes[1, 1].set_title('Negative Log-Likelihood (Probabilistic Methods)', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 1].text(0.5, 0.5, 'No probabilistic methods\nwith uncertainty',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('comparison_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
