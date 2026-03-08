"""Hill Mixture MMM - Bayesian Marketing Mix Modeling with Mixture of Hill Curves.

A research implementation for the paper:
"Non-parametric Bayesian Mixture Models for Marketing Mix Response Curves"

Key modules:
- transforms: Core mathematical transforms (adstock, hill)
- models: NumPyro model definitions
- data: Data generation with multiple DGP scenarios
- inference: MCMC execution and evaluation
- metrics: Effective K, parameter recovery
- data_loader: Real data loading utilities
- baseline: Baseline model implementations
"""

from .data import DGP_CONFIGS, DGPConfig, compute_prior_config, generate_data
from .inference import (
    compute_convergence_diagnostics,
    compute_label_invariant_diagnostics,
    compute_loo,
    compute_predictions,
    compute_predictive_metrics,
    compute_waic,
    run_inference,
)
from .metrics import compute_delta_loo, compute_effective_k, compute_parameter_recovery
from .models import (
    model_hill_mixture_hierarchical_reparam,
    model_single_hill,
)
from .transforms import adstock_geometric, hill, hill_matrix

__version__ = "0.1.0"

__all__ = [
    # Data
    "DGPConfig",
    "DGP_CONFIGS",
    "generate_data",
    "compute_prior_config",
    # Inference
    "run_inference",
    "compute_convergence_diagnostics",
    "compute_label_invariant_diagnostics",
    "compute_loo",
    "compute_predictions",
    "compute_predictive_metrics",
    "compute_waic",
    # Metrics
    "compute_delta_loo",
    "compute_effective_k",
    "compute_parameter_recovery",
    # Models
    "model_single_hill",
    "model_hill_mixture_hierarchical_reparam",
    # Transforms
    "adstock_geometric",
    "hill",
    "hill_matrix",
]
