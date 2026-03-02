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
from pathlib import Path

import numpyro
import pandas as pd

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


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    name: str
    fn: Callable
    kwargs: dict


# Default model configurations
# All mixture models use the same hierarchical reparameterized structure with K as parameter
MODEL_SPECS = [
    ModelSpec("single_hill", model_single_hill, {}),
    ModelSpec("mixture_k2", model_hill_mixture_hierarchical_reparam, {"K": 2}),
    ModelSpec("mixture_k3", model_hill_mixture_hierarchical_reparam, {"K": 3}),
    ModelSpec("mixture_k5", model_hill_mixture_hierarchical_reparam, {"K": 5}),
]


def _build_retry_schedule(num_warmup: int, num_samples: int, is_mixture_model: bool) -> list[dict]:
    """Build deterministic retry schedule for convergence."""
    base_attempt = [
        {
            "seed_offset": 0,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "target_accept_prob": 0.90,
        }
    ]
    if not is_mixture_model:
        return base_attempt

    # Fixed schedule avoids cherry-picking while improving difficult mixture cases.
    return base_attempt + [
        {
            "seed_offset": 1000,
            "num_warmup": max(num_warmup, 2000),
            "num_samples": max(num_samples, 3000),
            "target_accept_prob": 0.95,
        },
        {
            "seed_offset": 3000,
            "num_warmup": max(num_warmup, 2000),
            "num_samples": max(num_samples, 3000),
            "target_accept_prob": 0.95,
        },
        {
            "seed_offset": 4000,
            "num_warmup": max(num_warmup, 2000),
            "num_samples": max(num_samples, 3000),
            "target_accept_prob": 0.95,
        },
        {
            "seed_offset": 2000,
            "num_warmup": max(num_warmup, 3000),
            "num_samples": max(num_samples, 4000),
            "target_accept_prob": 0.97,
        },
    ]


def _is_effectively_converged(convergence: dict, rhat_log_lik: float | None, rhat_threshold: float) -> bool:
    """Evaluate convergence using standard and label-invariant diagnostics."""
    if bool(convergence["converged"]):
        return True
    if rhat_log_lik is None:
        return False
    return bool(rhat_log_lik < rhat_threshold)


def _prepare_experiment_data(dgp_config: DGPConfig, train_ratio: float) -> dict:
    """Generate data, split train/test, and compute prior config."""
    x, y, meta = generate_data(dgp_config)
    T = len(y)
    T_train = int(T * train_ratio)
    x_train, y_train = x[:T_train], y[:T_train]
    x_test, y_test = x[T_train:], y[T_train:]
    prior_config = compute_prior_config(x_train, y_train)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "prior_config": prior_config,
        "meta": meta,
        "T": T,
        "T_train": T_train,
    }


def _fit_with_retries(
    dgp_config: DGPConfig,
    model_spec: ModelSpec,
    x_train,
    y_train,
    prior_config: dict,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
) -> dict:
    """Run inference with deterministic retry schedule."""
    is_mixture_model = "K" in model_spec.kwargs
    retry_schedule = _build_retry_schedule(num_warmup, num_samples, is_mixture_model)

    mcmc = None
    convergence = None
    rhat_log_lik = None
    rhat_threshold = 1.01
    converged_effective = False
    retry_attempt = 0
    inference_seed = dgp_config.seed
    used_warmup = num_warmup
    used_samples = num_samples
    used_target_accept = 0.90

    for attempt_idx, attempt in enumerate(retry_schedule):
        inference_seed = dgp_config.seed + int(attempt["seed_offset"])
        used_warmup = int(attempt["num_warmup"])
        used_samples = int(attempt["num_samples"])
        used_target_accept = float(attempt["target_accept_prob"])

        mcmc = run_inference(
            model_spec.fn,
            x_train,
            y_train,
            seed=inference_seed,
            num_warmup=used_warmup,
            num_samples=used_samples,
            num_chains=num_chains,
            prior_config=prior_config,
            target_accept_prob=used_target_accept,
            **model_spec.kwargs,
        )
        convergence = compute_convergence_diagnostics(mcmc)

        rhat_log_lik = None
        if is_mixture_model:
            label_invariant = compute_label_invariant_diagnostics(mcmc, x_train, y_train)
            rhat_log_lik = float(label_invariant["rhat_log_lik"])
            rhat_threshold = float(label_invariant["threshold"])

        converged_effective = _is_effectively_converged(convergence, rhat_log_lik, rhat_threshold)
        retry_attempt = attempt_idx
        if converged_effective:
            break

    assert mcmc is not None
    assert convergence is not None

    return {
        "mcmc": mcmc,
        "convergence": convergence,
        "is_mixture_model": is_mixture_model,
        "rhat_log_lik": rhat_log_lik,
        "rhat_threshold": rhat_threshold,
        "converged_effective": converged_effective,
        "retry_attempt": retry_attempt,
        "inference_seed": inference_seed,
        "used_warmup": used_warmup,
        "used_samples": used_samples,
        "used_target_accept": used_target_accept,
    }


def _compute_train_test_metrics(
    mcmc,
    model_spec: ModelSpec,
    prior_config: dict,
    x_train,
    y_train,
    x_test,
    y_test,
) -> tuple[dict, dict]:
    """Compute posterior predictive metrics on train and test splits."""
    pred_train = compute_predictions(
        mcmc, model_spec.fn, x_train, prior_config=prior_config, **model_spec.kwargs
    )
    pred_test = compute_predictions(
        mcmc, model_spec.fn, x_test, prior_config=prior_config, **model_spec.kwargs
    )

    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

    return train_metrics, test_metrics


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
    prepared = _prepare_experiment_data(dgp_config, train_ratio)
    fit = _fit_with_retries(
        dgp_config=dgp_config,
        model_spec=model_spec,
        x_train=prepared["x_train"],
        y_train=prepared["y_train"],
        prior_config=prepared["prior_config"],
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )

    mcmc = fit["mcmc"]
    convergence = fit["convergence"]
    loo = compute_loo(mcmc)
    waic = compute_waic(mcmc)
    effective_k = compute_effective_k(mcmc)
    param_recovery = compute_parameter_recovery(mcmc, prepared["meta"])
    train_metrics, test_metrics = _compute_train_test_metrics(
        mcmc=mcmc,
        model_spec=model_spec,
        prior_config=prepared["prior_config"],
        x_train=prepared["x_train"],
        y_train=prepared["y_train"],
        x_test=prepared["x_test"],
        y_test=prepared["y_test"],
    )

    return {
        "dgp": dgp_config.dgp_type,
        "K_true": prepared["meta"]["K_true"],
        "model": model_spec.name,
        "seed": dgp_config.seed,
        "T": prepared["T"],
        "T_train": prepared["T_train"],
        # Convergence
        "max_rhat": convergence["max_rhat"],
        "min_ess_bulk": convergence["min_ess_bulk"],
        "converged_standard": convergence["converged"],
        "rhat_log_lik": fit["rhat_log_lik"],
        "rhat_threshold": fit["rhat_threshold"] if fit["is_mixture_model"] else None,
        "converged": fit["converged_effective"],
        "retry_attempt": fit["retry_attempt"],
        "inference_seed": fit["inference_seed"],
        "num_warmup_used": fit["used_warmup"],
        "num_samples_used": fit["used_samples"],
        "target_accept_prob_used": fit["used_target_accept"],
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
        "alpha_true": prepared["meta"]["alpha_true"],
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


def export_results_csv(df: pd.DataFrame, path: str | Path, include_summary: bool = True) -> None:
    """Export benchmark results to CSV.

    Args:
        df: Raw results DataFrame from run_benchmark_suite
        path: Output file path (without extension)
        include_summary: Also export summary statistics
    """
    from pathlib import Path as _Path

    output_path = _Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export raw results
    raw_path = output_path.with_suffix(".csv")
    df.to_csv(raw_path, index=False)
    print(f"Raw results exported to {raw_path}")

    # Export summary if requested
    if include_summary:
        summary = summarize_benchmark(df)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
        summary.to_csv(summary_path)
        print(f"Summary exported to {summary_path}")


def export_results_json(df: pd.DataFrame, path: str | Path, include_summary: bool = True) -> None:
    """Export benchmark results to JSON.

    Args:
        df: Raw results DataFrame from run_benchmark_suite
        path: Output file path (without extension)
        include_summary: Also export summary statistics
    """
    from pathlib import Path as _Path

    output_path = _Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export raw results
    raw_path = output_path.with_suffix(".json")
    df.to_json(raw_path, orient="records", indent=2)
    print(f"Raw results exported to {raw_path}")

    # Export summary if requested
    if include_summary:
        summary = summarize_benchmark(df)
        # Convert multi-index summary to JSON-friendly format
        summary_reset = summary.reset_index()
        summary_reset.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in summary_reset.columns
        ]
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        summary_reset.to_json(summary_path, orient="records", indent=2)
        print(f"Summary exported to {summary_path}")
