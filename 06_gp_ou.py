#!/usr/bin/env python3
"""
Gaussian Process Regression with OU (Exponential) Kernel
========================================================
Non-parametric Bayesian approach using Matérn-1/2 (exponential) covariance.

Kernel:
    k(t, t') = σ² exp(-|t - t'| / ℓ) + σ_n² δ(t, t')

where:
    σ² = signal variance (outputscale)
    ℓ = length scale (related to τ by ℓ ≈ τ)
    σ_n² = observation noise variance

The exponential kernel is equivalent to:
- Matérn-1/2 kernel
- OU process covariance (for continuous-time GP)
- Spectral density: Lorentzian (1/(1 + f²))

Advantages:
- Non-parametric (no commitment to specific dynamics)
- Exact posterior mean and variance
- Handles arbitrary gap patterns
- Can use sparse approximations for large N

Disadvantages:
- Computationally expensive (O(N³) exact, O(NM²) sparse)
- Hyperparameter optimization sensitive to initialization
- May overfit if lengthscale not constrained

References:
- Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes for machine learning.
- Wilson, A. G., et al. (2015). Kernel interpolation for scalable structured Gaussian processes.
- Gardner, J. R., et al. (2018). GPyTorch: Blackbox matrix-matrix Gaussian process inference.
"""

import sys
import math
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, Optional
import warnings

# Add parent directory to path for imports
sys.path.append('.')
from pathlib import Path
if Path('00_starter_framework.py').exists():
    exec(open('00_starter_framework.py').read())

# Try importing GPyTorch and PyTorch
try:
    import torch
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.means import ZeroMean
    from gpytorch.kernels import ScaleKernel, MaternKernel, GridInterpolationKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.distributions import MultivariateNormal
    GPYTORCH_AVAILABLE = True

    # Device selection
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')  # Apple Silicon
    else:
        DEVICE = torch.device('cpu')

except ImportError:
    GPYTORCH_AVAILABLE = False
    DEVICE = None
    print("⚠ GPyTorch not available. Install with: pip install gpytorch")


class ExactOUGP(ExactGP):
    """
    Exact GP with Matérn-1/2 (exponential/OU) kernel.

    Equivalent to OU process covariance in continuous time.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactOUGP, self).__init__(train_x, train_y, likelihood)

        # Zero mean (data should be centered)
        self.mean_module = ZeroMean()

        # Matérn-1/2 = Exponential kernel
        # This is the OU process kernel: k(Δt) = σ² exp(-|Δt|/ℓ)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=0.5, ard_num_dims=1)  # nu=0.5 is exponential
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SparseOUGP(ExactGP):
    """
    Sparse GP with SKI (Structured Kernel Interpolation) for O(N) complexity.

    Uses grid interpolation to approximate kernel matrix for scalability.
    """
    def __init__(self, train_x, train_y, likelihood, grid_size=128):
        super(SparseOUGP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = ZeroMean()

        # Base kernel (Matérn-1/2)
        base_kernel = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=1))

        # Wrap in grid interpolation for efficiency
        self.covar_module = GridInterpolationKernel(
            base_kernel,
            grid_size=grid_size,
            num_dims=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp_model(
    model: ExactGP,
    likelihood: GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int = 300,
    lr: float = 0.05,
    patience: int = 20,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Train GP model by optimizing marginal log-likelihood.

    Args:
        model: GP model
        likelihood: Gaussian likelihood
        train_x: (N, 1) training inputs
        train_y: (N,) training targets
        n_iter: maximum number of iterations
        lr: learning rate
        patience: early stopping patience
        verbose: print training progress

    Returns:
        dict with training diagnostics
    """
    model.train()
    likelihood.train()

    # Optimizer for hyperparameters
    # Use model.hyperparameters() to get trainable params
    optimizer = torch.optim.Adam(model.hyperparameters(), lr=lr)

    mll = ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    patience_counter = 0
    losses = []

    for i in range(n_iter):
        optimizer.zero_grad()

        # Forward pass
        output = model(train_x)
        loss = -mll(output, train_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at iteration {i+1}")
            break

        if verbose and (i % 50 == 0 or i == n_iter - 1):
            # Extract hyperparameters
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            outputscale = model.covar_module.outputscale.item()
            noise = likelihood.noise.item()
            print(f"    Iter {i+1:3d}: Loss={loss_val:.3f}, "
                  f"ℓ={lengthscale:.4f}s, σ²={outputscale:.1f}, "
                  f"noise={noise:.2f}")

    # Extract final hyperparameters
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    outputscale = model.covar_module.outputscale.item()
    noise = likelihood.noise.item()

    return {
        'lengthscale': lengthscale,
        'outputscale': outputscale,
        'noise': noise,
        'final_loss': best_loss,
        'n_iter': i + 1,
        'converged': patience_counter >= patience
    }


def impute_gp_ou(
    X: np.ndarray,
    t: np.ndarray,
    t_out: np.ndarray,
    obs_mask: np.ndarray,
    use_sparse: bool = True,
    grid_size: int = 256,
    n_iter: int = 300,
    lr: float = 0.05,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Impute trajectory using GP regression with OU (exponential) kernel.

    Args:
        X: (N, D) trajectory at native sampling rate
        t: (N,) time array in seconds
        t_out: (M,) output time array in seconds (typically 1000 Hz)
        obs_mask: (N,) boolean mask, True = observed, False = missing
        use_sparse: Use sparse (SKI) approximation for efficiency
        grid_size: Grid size for SKI (if use_sparse=True)
        n_iter: Maximum training iterations
        lr: Learning rate for Adam optimizer
        verbose: Print training progress

    Returns:
        dict with:
            - output_300Hz: (N, D) posterior mean at native rate
            - output_1000Hz: (M, D) posterior mean at target rate
            - mean: (N, D) posterior mean
            - var: (N, D) posterior variance
            - extras: dict with learned hyperparameters per dimension
    """
    if not GPYTORCH_AVAILABLE:
        raise ImportError("GPyTorch is required for this method. Install with: pip install gpytorch")

    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    if X.ndim == 1:
        X = X[:, None]

    N, D = X.shape
    M = len(t_out)

    # Validate inputs
    assert len(t) == N
    assert len(obs_mask) == N
    assert np.any(obs_mask), "No observed points"

    n_obs = np.sum(obs_mask)

    # Prepare output arrays
    mu_300 = np.zeros((N, D), dtype=np.float64)
    var_300 = np.zeros((N, D), dtype=np.float64)
    mu_1000 = np.zeros((M, D), dtype=np.float64)

    # Prepare tensors
    t_obs = t[obs_mask]
    train_x = torch.tensor(t_obs, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (n_obs, 1)
    test_x_300 = torch.tensor(t, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (N, 1)
    test_x_1000 = torch.tensor(t_out, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (M, 1)

    hyperparams_per_dim = []

    # Process each dimension independently
    for d in range(D):
        if verbose:
            print(f"\n  Dimension {d}:")

        # Extract observed data for this dimension
        y_obs = X[obs_mask, d]

        # Center data (GP assumes zero mean)
        y_mean = np.mean(y_obs)
        y_obs_centered = y_obs - y_mean

        train_y = torch.tensor(y_obs_centered, dtype=torch.float32, device=DEVICE)

        # Initialize likelihood
        likelihood = GaussianLikelihood().to(DEVICE)

        # Initialize model
        if use_sparse and n_obs > 512:
            # Use sparse approximation for large datasets
            model = SparseOUGP(train_x, train_y, likelihood, grid_size=grid_size).to(DEVICE)
            if verbose:
                print(f"    Using sparse GP with grid_size={grid_size}")
        else:
            # Use exact GP
            model = ExactOUGP(train_x, train_y, likelihood).to(DEVICE)
            if verbose:
                print(f"    Using exact GP")

        # Train model
        hyperparams = train_gp_model(
            model, likelihood, train_x, train_y,
            n_iter=n_iter, lr=lr, verbose=verbose
        )
        hyperparams_per_dim.append(hyperparams)

        # Make predictions
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Predictions at 300 Hz
            pred_300 = likelihood(model(test_x_300))
            mu_300_d = pred_300.mean.cpu().numpy() + y_mean
            var_300_d = pred_300.variance.cpu().numpy()

            # Predictions at 1000 Hz
            pred_1000 = likelihood(model(test_x_1000))
            mu_1000_d = pred_1000.mean.cpu().numpy() + y_mean

        mu_300[:, d] = mu_300_d
        var_300[:, d] = var_300_d
        mu_1000[:, d] = mu_1000_d

    # Aggregate diagnostics
    extras = {
        'hyperparams_per_dim': hyperparams_per_dim,
        'lengthscale_mean': float(np.mean([h['lengthscale'] for h in hyperparams_per_dim])),
        'outputscale_mean': float(np.mean([h['outputscale'] for h in hyperparams_per_dim])),
        'noise_mean': float(np.mean([h['noise'] for h in hyperparams_per_dim])),
        'use_sparse': use_sparse,
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

def evaluate_gp_ou(
    segments: list,
    use_sparse: bool = True,
    grid_size: int = 256,
    n_iter: int = 300,
    config: ExperimentConfig = CONFIG,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate GP-OU on all segments."""

    if not GPYTORCH_AVAILABLE:
        print("⚠ GPyTorch not available. Skipping evaluation.")
        return {}

    print("\n" + "="*80)
    print(f"EVALUATING: GP with OU Kernel ({'Sparse SKI' if use_sparse else 'Exact'})")
    print("="*80)
    print(f"Parameters: n_iter={n_iter}, grid_size={grid_size if use_sparse else 'N/A'}")
    print()

    all_metrics = []

    for seg_id, seg in enumerate(segments):
        t_seg = seg['t']
        X_true = seg['X_true']
        obs_mask = seg['obs_mask']

        t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

        result = impute_gp_ou(
            X_true, t_seg, t_out, obs_mask,
            use_sparse=use_sparse,
            grid_size=grid_size,
            n_iter=n_iter,
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
              f"ℓ={params['lengthscale_mean']*1000:.1f}ms, "
              f"σ²={params['outputscale_mean']:.0f}")

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
    use_sparse: bool = True,
    config: ExperimentConfig = CONFIG
):
    """Visualize imputation on a sample segment."""
    t_seg = segment['t']
    X_true = segment['X_true']
    obs_mask = segment['obs_mask']

    t_out = np.arange(0.0, t_seg[-1] + 1e-9, 1.0 / config.fs_out)

    result = impute_gp_ou(
        X_true, t_seg, t_out, obs_mask,
        use_sparse=use_sparse,
        verbose=True
    )

    X_pred = result['output_300Hz']
    X_var = result['var']

    # Plot
    plot_imputation_comparison(
        t_seg, X_true, X_pred, X_var=X_var,
        obs_mask=obs_mask,
        method_name=f"GP with OU Kernel ({'Sparse' if use_sparse else 'Exact'})",
        filename="06_gp_ou_result.png"
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
                method_name="GP with OU Kernel (Zoom)",
                filename="06_gp_ou_zoom.png"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not GPYTORCH_AVAILABLE:
        print("⚠ GPyTorch is not installed. Please install with:")
        print("  pip install gpytorch torch")
        sys.exit(1)

    print("Loading data and segments from starter framework...")
    t, X, X_detrended, ou_params, segments = main_setup()

    print(f"\nRunning on device: {DEVICE}")

    # Evaluate with sparse approximation (recommended for speed)
    metrics = evaluate_gp_ou(
        segments,
        use_sparse=True,
        grid_size=256,
        n_iter=300,
        config=CONFIG,
        verbose=False
    )

    # Visualize on first segment
    print("\nGenerating visualization...")
    visualize_sample_result(segments[0], use_sparse=True, config=CONFIG)

    print("\n✓ GP with OU kernel evaluation complete.")
    print(f"  Final: MAE={metrics.get('MAE', np.nan):.3f} nm, "
          f"RMSE={metrics.get('RMSE', np.nan):.3f} nm, "
          f"R²={metrics.get('R2', np.nan):.4f}")
