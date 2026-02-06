"""Benchmark runner for Hill Mixture MMM experiments.

Runs the full experimental matrix:
- 4 DGP scenarios (single, mixture_k2, mixture_k3, mixture_k5)
- 3 models (single_hill, mixture_k3, sparse_k5)
- Multiple seeds for variance estimation

Usage:
    from hill_mmm.benchmark import run_benchmark_suite
    results = run_benchmark_suite(seeds=[0, 1, 2])
    print(results.to_markdown())
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpyro
import pandas as pd

from .data import DGP_CONFIGS, DGPConfig, compute_prior_config, generate_data
from .inference import (
    compute_convergence_diagnostics,
    compute_loo,
    compute_predictions,
    compute_predictive_metrics,
    compute_waic,
    run_inference,
)
from .metrics import compute_delta_loo, compute_effective_k, compute_parameter_recovery
from .models import (
    model_hill_mixture_k2,
    model_hill_mixture_hierarchical_reparam,
    model_single_hill,
)


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    name: str
    fn: Callable
    kwargs: dict


# Default model configurations
MODEL_SPECS = [
    ModelSpec("single_hill", model_single_hill, {}),
    ModelSpec("mixture_k2", model_hill_mixture_k2, {}),
    ModelSpec("hierarchical_reparam_k3", model_hill_mixture_hierarchical_reparam, {"K": 3}),
]


def run_single_experiment(
    dgp_config: DGPConfig,
    model_spec: ModelSpec,
    train_ratio: float = 0.75,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
) -> dict:
    """Run a single DGP x Model experiment.

    Args:
        dgp_config: Data generating process configuration
        model_spec: Model specification
        train_ratio: Fraction of data for training
        num_warmup: MCMC warmup iterations
        num_samples: MCMC sample iterations
        num_chains: Number of MCMC chains

    Returns:
        Dict with all evaluation metrics
    """
    # Generate data
    x, y, meta = generate_data(dgp_config)
    T = len(y)
    T_train = int(T * train_ratio)

    x_train, y_train = x[:T_train], y[:T_train]
    x_test, y_test = x[T_train:], y[T_train:]

    # Compute priors from training data
    prior_config = compute_prior_config(x_train, y_train)

    # Run inference
    mcmc = run_inference(
        model_spec.fn,
        x_train,
        y_train,
        seed=dgp_config.seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        prior_config=prior_config,
        **model_spec.kwargs,
    )

    # Compute all metrics
    convergence = compute_convergence_diagnostics(mcmc)
    loo = compute_loo(mcmc)
    waic = compute_waic(mcmc)
    effective_k = compute_effective_k(mcmc)
    param_recovery = compute_parameter_recovery(mcmc, meta)

    # Predictive metrics on train and test
    pred_train = compute_predictions(
        mcmc, model_spec.fn, x_train, prior_config=prior_config, **model_spec.kwargs
    )
    pred_test = compute_predictions(
        mcmc, model_spec.fn, x_test, prior_config=prior_config, **model_spec.kwargs
    )

    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

    return {
        "dgp": dgp_config.dgp_type,
        "K_true": meta["K_true"],
        "model": model_spec.name,
        "seed": dgp_config.seed,
        "T": T,
        "T_train": T_train,
        # Convergence
        "max_rhat": convergence["max_rhat"],
        "min_ess_bulk": convergence["min_ess_bulk"],
        "converged": convergence["converged"],
        # Model comparison
        "elpd_loo": loo.get("elpd_loo"),
        "loo_se": loo.get("se"),
        "p_loo": loo.get("p_loo"),
        "elpd_waic": waic.get("elpd_waic"),
        "waic_se": waic.get("se"),
        # Predictive
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "train_coverage_90": train_metrics["coverage_90"],
        "test_coverage_90": test_metrics["coverage_90"],
        # Effective K
        "effective_k_mean": effective_k["effective_k_mean"],
        "effective_k_std": effective_k["effective_k_std"],
        # Parameter recovery
        "alpha_in_ci": param_recovery.get("alpha", {}).get("in_ci"),
        "sigma_in_ci": param_recovery.get("sigma", {}).get("in_ci"),
        "alpha_true": meta["alpha_true"],
        "alpha_est": param_recovery.get("alpha", {}).get("mean"),
    }


def run_benchmark_suite(
    dgp_names: list[str] | None = None,
    model_names: list[str] | None = None,
    seeds: list[int] | None = None,
    train_ratio: float = 0.75,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full benchmark suite.

    Args:
        dgp_names: List of DGP names to test (default: all)
        model_names: List of model names to test (default: all)
        seeds: List of random seeds (default: [0, 1, 2, 3, 4])
        train_ratio: Fraction for training
        num_warmup: MCMC warmup
        num_samples: MCMC samples
        num_chains: MCMC chains
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    # Set up multi-chain
    numpyro.set_host_device_count(num_chains)

    # Defaults
    if dgp_names is None:
        dgp_names = list(DGP_CONFIGS.keys())
    if model_names is None:
        model_names = [m.name for m in MODEL_SPECS]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    # Filter model specs
    models = [m for m in MODEL_SPECS if m.name in model_names]

    results = []
    total = len(dgp_names) * len(models) * len(seeds)
    current = 0

    for dgp_name in dgp_names:
        base_config = DGP_CONFIGS[dgp_name]

        for seed in seeds:
            # Create config with this seed
            config = DGPConfig(
                dgp_type=base_config.dgp_type,
                T=base_config.T,
                sigma=base_config.sigma,
                alpha=base_config.alpha,
                intercept=base_config.intercept,
                slope=base_config.slope,
                seed=seed,
            )

            for model_spec in models:
                current += 1
                if verbose:
                    print(
                        f"[{current}/{total}] DGP={dgp_name}, Model={model_spec.name}, Seed={seed}"
                    )

                result = run_single_experiment(
                    config,
                    model_spec,
                    train_ratio=train_ratio,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                )
                results.append(result)

    df = pd.DataFrame(results)

    # Add delta LOO relative to single_hill baseline
    df = _add_delta_loo(df)

    return df


def _add_delta_loo(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta LOO columns relative to single_hill baseline."""
    df = df.copy()
    df["delta_loo"] = None
    df["delta_loo_significant"] = None

    for (dgp, seed), group in df.groupby(["dgp", "seed"]):
        baseline_row = group[group["model"] == "single_hill"]
        if len(baseline_row) == 0:
            continue

        baseline_loo = {
            "elpd_loo": baseline_row["elpd_loo"].iloc[0],
            "se": baseline_row["loo_se"].iloc[0],
        }

        for idx in group.index:
            if df.loc[idx, "model"] == "single_hill":
                df.loc[idx, "delta_loo"] = 0.0
                df.loc[idx, "delta_loo_significant"] = False
            else:
                model_loo = {
                    "elpd_loo": df.loc[idx, "elpd_loo"],
                    "se": df.loc[idx, "loo_se"],
                }
                delta = compute_delta_loo(model_loo, baseline_loo)
                df.loc[idx, "delta_loo"] = delta["delta_loo"]
                df.loc[idx, "delta_loo_significant"] = delta["significant"]

    return df


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark results across seeds.

    Args:
        df: Raw results DataFrame

    Returns:
        Summary DataFrame with mean ± std across seeds
    """
    metrics = [
        "elpd_loo",
        "test_rmse",
        "train_rmse",
        "test_coverage_90",
        "effective_k_mean",
        "delta_loo",
    ]

    summary = df.groupby(["dgp", "K_true", "model"])[metrics].agg(["mean", "std"]).round(2)

    return summary


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print benchmark results as a formatted table.

    Args:
        df: Summary DataFrame from summarize_benchmark
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for dgp in df["dgp"].unique():
        K_true = df[df["dgp"] == dgp]["K_true"].iloc[0]
        print(f"\nDGP: {dgp} (K_true={K_true})")
        print("-" * 60)

        dgp_data = df[df["dgp"] == dgp]

        print(
            f"{'Model':<15} {'LOO':>10} {'Test RMSE':>12} {'Cov 90%':>10} {'Eff K':>8} {'ΔLOO':>10}"
        )
        print("-" * 60)

        for model in dgp_data["model"].unique():
            model_data = dgp_data[dgp_data["model"] == model]
            loo = model_data["elpd_loo"].mean()
            test_rmse = model_data["test_rmse"].mean()
            cov = model_data["test_coverage_90"].mean()
            eff_k = model_data["effective_k_mean"].mean()
            delta = model_data["delta_loo"].mean()

            delta_str = f"{delta:+.1f}" if not pd.isna(delta) else "N/A"
            print(
                f"{model:<15} {loo:>10.1f} {test_rmse:>12.3f} "
                f"{cov:>10.1%} {eff_k:>8.2f} {delta_str:>10}"
            )
