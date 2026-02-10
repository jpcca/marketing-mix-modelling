"""Hill Mixture MMM - Bayesian Marketing Mix Modeling with Mixture of Hill Curves.

A research implementation for the paper:
"Non-parametric Bayesian Mixture Models for Marketing Mix Response Curves"

Key modules:
- transforms: Core mathematical transforms (adstock, hill)
- models: NumPyro model definitions
- data: Data generation with multiple DGP scenarios
- inference: MCMC execution and evaluation
- metrics: Effective K, parameter recovery
- benchmark: Experimental benchmark runner
"""

from .benchmark import (
    export_results_csv,
    export_results_json,
    print_benchmark_table,
    run_benchmark_suite,
    run_single_experiment,
    summarize_benchmark,
)
from .data import DGP_CONFIGS, DGPConfig, compute_prior_config, generate_data
from .models import (
    model_hill_mixture_hierarchical_reparam,
    model_single_hill,
)
from .transforms import adstock_geometric, hill, hill_matrix

__version__ = "0.1.0"

__all__ = [
    # Benchmark
    "run_benchmark_suite",
    "run_single_experiment",
    "summarize_benchmark",
    "print_benchmark_table",
    "export_results_csv",
    "export_results_json",
    # Data
    "DGPConfig",
    "DGP_CONFIGS",
    "generate_data",
    "compute_prior_config",
    # Models
    "model_single_hill",
    "model_hill_mixture_hierarchical_reparam",
    # Transforms
    "adstock_geometric",
    "hill",
    "hill_matrix",
]
