#!/usr/bin/env python
"""Run Hill Mixture MMM benchmarks (synthetic and/or real data).

Unified benchmark runner for paper experiments. By default runs both
synthetic and real data experiments. Use flags to run specific subsets.

Usage:
    # Run all experiments (synthetic + real data)
    python scripts/run_benchmark.py

    # Run only synthetic experiments
    python scripts/run_benchmark.py --synthetic-only

    # Run only real data experiments
    python scripts/run_benchmark.py --real-only

    # Quick test mode (reduced samples/seeds)
    python scripts/run_benchmark.py --quick

    # Specific DGPs or models
    python scripts/run_benchmark.py --dgp single mixture_k2 --model single_hill mixture_k2

    # Custom output directory
    python scripts/run_benchmark.py --output results/my_experiment
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Add parent src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import numpyro
import pandas as pd

from hill_mixture_mmm.data import DGP_CONFIGS, DGPConfig, compute_prior_config, generate_data
from hill_mixture_mmm.inference import (
    compute_comprehensive_mixture_diagnostics,
    compute_convergence_diagnostics,
    compute_label_invariant_diagnostics,
    compute_loo,
    compute_predictions,
    compute_predictive_metrics,
    compute_waic,
    relabel_samples_by_k,
    run_inference,
)
from hill_mixture_mmm.metrics import compute_delta_loo, compute_effective_k, compute_parameter_recovery
from hill_mixture_mmm.models import model_hill_mixture_hierarchical_reparam, model_single_hill


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


def _is_effectively_converged(
    convergence: dict, rhat_log_lik: float | None, rhat_threshold: float
) -> bool:
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
        mcmc,
        model_spec.fn,
        x_test,
        prior_config=prior_config,
        history_x=x_train,
        **model_spec.kwargs,
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

    for (dgp, seed), group in df.groupby(["dgp", "seed"]):  # type: ignore[misc]
        baseline_row = group[group["model"] == "single_hill"]
        if len(baseline_row) == 0:
            continue

        baseline_loo = {
            "elpd_loo": baseline_row["elpd_loo"].iloc[0],  # type: ignore[union-attr]
            "se": baseline_row["loo_se"].iloc[0],  # type: ignore[union-attr]
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
        Summary DataFrame with mean +/- std across seeds
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

    return summary  # type: ignore[return-value]


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print benchmark results as a formatted table.

    Args:
        df: Summary DataFrame from summarize_benchmark
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for dgp in df["dgp"].unique():
        K_true = df[df["dgp"] == dgp]["K_true"].iloc[0]  # type: ignore[union-attr]
        print(f"\nDGP: {dgp} (K_true={K_true})")
        print("-" * 60)

        dgp_data = df[df["dgp"] == dgp]

        print(
            f"{'Model':<15} {'LOO':>10} {'Test RMSE':>12} {'Cov 90%':>10} {'Eff K':>8} {'\u0394LOO':>10}"
        )
        print("-" * 60)

        for model in dgp_data["model"].unique():  # type: ignore[union-attr]
            model_data = dgp_data[dgp_data["model"] == model]
            loo = model_data["elpd_loo"].mean()
            test_rmse = model_data["test_rmse"].mean()
            cov = model_data["test_coverage_90"].mean()
            eff_k = model_data["effective_k_mean"].mean()
            delta = model_data["delta_loo"].mean()

            delta_str = f"{delta:+.1f}" if not pd.isna(delta) else "N/A"  # type: ignore[arg-type]
            print(
                f"{model:<15} {loo:>10.1f} {test_rmse:>12.3f} "
                f"{cov:>10.1%} {eff_k:>8.2f} {delta_str:>10}"
            )


def export_results_csv(
    df: pd.DataFrame, path: str | Path, include_summary: bool = True
) -> None:
    """Export benchmark results to CSV.

    Args:
        df: Raw results DataFrame from run_benchmark_suite
        path: Output file path (without extension)
        include_summary: Also export summary statistics
    """
    output_path = Path(path)
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


def export_results_json(
    df: pd.DataFrame, path: str | Path, include_summary: bool = True
) -> None:
    """Export benchmark results to JSON.

    Args:
        df: Raw results DataFrame from run_benchmark_suite
        path: Output file path (without extension)
        include_summary: Also export summary statistics
    """
    output_path = Path(path)
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
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in summary_reset.columns
        ]
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        summary_reset.to_json(summary_path, orient="records", indent=2)
        print(f"Summary exported to {summary_path}")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    # Synthetic experiments
    synthetic_dgps: list[str]
    synthetic_models: list[str]
    synthetic_seeds: list[int]

    # Real data experiments
    real_n_orgs: int
    real_models: list[str]
    real_seeds: list[int]

    # MCMC settings
    num_warmup: int
    num_samples: int
    num_chains: int

    # Data settings
    train_ratio: float

    # Output
    output_dir: str


def get_default_config() -> BenchmarkConfig:
    """Get default full experiment configuration."""
    return BenchmarkConfig(
        # Synthetic: 4 DGPs x 4 models x 5 seeds = 80 experiments
        synthetic_dgps=["single", "mixture_k2", "mixture_k3", "mixture_k5"],
        synthetic_models=["single_hill", "mixture_k2", "mixture_k3", "mixture_k5"],
        synthetic_seeds=[0, 1, 2, 3, 4],
        # Real: 10 orgs x 3 models x 3 seeds = 90 experiments
        real_n_orgs=10,
        real_models=["single_hill", "mixture_k2", "mixture_k3"],
        real_seeds=[0, 1, 2],
        # MCMC - NOTE: real data uses num_warmup=2000 (see run_real_data_experiments)
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        # Data
        train_ratio=0.75,
        # Output
        output_dir="results/benchmark",
    )


def get_quick_config() -> BenchmarkConfig:
    """Get quick test configuration."""
    return BenchmarkConfig(
        synthetic_dgps=["single", "mixture_k2"],
        synthetic_models=["single_hill", "mixture_k2"],
        synthetic_seeds=[0],
        real_n_orgs=1,
        real_models=["single_hill", "mixture_k2"],
        real_seeds=[0],
        num_warmup=200,
        num_samples=200,
        num_chains=2,
        train_ratio=0.75,
        output_dir="results/benchmark_quick",
    )


def run_synthetic_experiments(config: BenchmarkConfig) -> pd.DataFrame:
    """Run synthetic data experiments."""
    print("=" * 60)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("=" * 60)
    n_exp = len(config.synthetic_dgps) * len(config.synthetic_models) * len(config.synthetic_seeds)
    print(f"DGPs: {config.synthetic_dgps}")
    print(f"Models: {config.synthetic_models}")
    print(f"Seeds: {config.synthetic_seeds}")
    print(f"Total: {n_exp} experiments")
    print()

    start_time = time.time()

    results = run_benchmark_suite(
        dgp_names=config.synthetic_dgps,
        model_names=config.synthetic_models,
        seeds=config.synthetic_seeds,
        train_ratio=config.train_ratio,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\nSynthetic experiments completed in {elapsed / 60:.1f} minutes")

    return results


def run_real_data_experiments(config: BenchmarkConfig) -> pd.DataFrame:
    """Run real data experiments."""
    from hill_mixture_mmm.data_loader import load_real_data, select_representative_timeseries

    print("=" * 60)
    print("REAL DATA EXPERIMENTS")
    print("=" * 60)
    n_exp = config.real_n_orgs * len(config.real_models) * len(config.real_seeds)
    print(f"Organizations: {config.real_n_orgs}")
    print(f"Models: {config.real_models}")
    print(f"Seeds: {config.real_seeds}")
    print(f"Total: {n_exp} experiments")
    print()

    # Load real data
    csv_path = Path("data/conjura_mmm_data.csv")
    if not csv_path.exists():
        print(f"WARNING: Real data not found at {csv_path}")
        print("Skipping real data experiments.")
        return pd.DataFrame()

    df_real = load_real_data(str(csv_path))

    # Select top N organizations by data quantity
    selected_org_ids = select_representative_timeseries(
        str(csv_path),
        n=config.real_n_orgs,
        selection_criteria="most_data",
        min_length=200,
        min_channels=1,
        seed=42,
    )

    # Get model specs
    model_lookup = {m.name: m for m in MODEL_SPECS}

    results = []
    total = n_exp
    current = 0
    start_time = time.time()

    for org_id in selected_org_ids:
        org_df = df_real[df_real["organization_id"] == org_id]
        if len(org_df) == 0:
            print(f"WARNING: No rows found for organization_id={org_id}, skipping")
            continue

        # Prepare data
        x = np.asarray(org_df["spend"].values)
        y = np.asarray(org_df["revenue"].values)
        T = len(y)
        T_train = int(T * config.train_ratio)

        x_train, y_train = x[:T_train], y[:T_train]
        x_test, y_test = x[T_train:], y[T_train:]

        prior_config = compute_prior_config(x_train, y_train)

        for model_name in config.real_models:
            model_spec = model_lookup.get(model_name)
            if model_spec is None:
                print(f"WARNING: Model {model_name} not found, skipping")
                continue

            for seed in config.real_seeds:
                current += 1
                print(f"[{current}/{total}] Org={org_id}, Model={model_name}, Seed={seed}")

                exp_start = time.time()

                try:
                    # Run inference (real data uses 2000 warmup per PLAN.md)
                    real_warmup = 2000
                    mcmc = run_inference(
                        model_spec.fn,
                        x_train,
                        y_train,
                        seed=seed,
                        num_warmup=real_warmup,
                        num_samples=config.num_samples,
                        num_chains=config.num_chains,
                        prior_config=prior_config,
                        **model_spec.kwargs,
                    )

                    exp_time = time.time() - exp_start

                    # Get samples and apply relabeling for mixture models
                    samples = mcmc.get_samples()

                    is_mixture = "k" in samples and len(samples["k"].shape) > 1
                    if is_mixture:
                        samples = relabel_samples_by_k(samples)

                    # Compute diagnostics
                    convergence = compute_convergence_diagnostics(mcmc)
                    loo = compute_loo(mcmc)
                    waic = compute_waic(mcmc)

                    # Compute mixture diagnostics for mixture models
                    mixture_diag = None
                    if is_mixture:
                        try:
                            mixture_diag = compute_comprehensive_mixture_diagnostics(
                                mcmc, x_train, y_train
                            )
                        except Exception as e:
                            print(f"  WARNING: Mixture diagnostics failed: {e}")

                    # Compute predictions
                    pred_train = compute_predictions(
                        mcmc,
                        model_spec.fn,
                        x_train,
                        prior_config=prior_config,
                        **model_spec.kwargs,
                    )
                    pred_test = compute_predictions(
                        mcmc,
                        model_spec.fn,
                        x_test,
                        prior_config=prior_config,
                        history_x=x_train,
                        **model_spec.kwargs,
                    )

                    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
                    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

                    result = {
                        "org_id": str(org_id),
                        "model": model_name,
                        "seed": seed,
                        "T": T,
                        "T_train": T_train,
                        "T_test": T - T_train,
                        # Convergence (standard)
                        "max_rhat": convergence["max_rhat"],
                        "min_ess_bulk": convergence["min_ess_bulk"],
                        "converged": convergence["converged"],
                        # LOO
                        "elpd_loo": loo.get("elpd_loo"),
                        "loo_se": loo.get("se"),
                        "p_loo": loo.get("p_loo"),
                        "pareto_k_bad": loo.get("pareto_k_bad", 0),
                        "pareto_k_very_bad": loo.get("pareto_k_very_bad", 0),
                        # WAIC
                        "elpd_waic": waic.get("elpd_waic"),
                        "waic_se": waic.get("se"),
                        "p_waic": waic.get("p_waic"),
                        # Mixture diagnostics (label-invariant)
                        "rhat_log_lik": (
                            mixture_diag["label_invariant"]["rhat_log_lik"]
                            if mixture_diag
                            else None
                        ),
                        "rhat_relabeled_max": (
                            mixture_diag["relabeled"]["max_rhat"] if mixture_diag else None
                        ),
                        "switching_rate": (
                            mixture_diag["label_switching"]["switching_rate"]
                            if mixture_diag
                            else None
                        ),
                        # Predictions
                        "train_rmse": train_metrics["rmse"],
                        "test_rmse": test_metrics["rmse"],
                        "train_coverage_90": train_metrics["coverage_90"],
                        "test_coverage_90": test_metrics["coverage_90"],
                        # Computation
                        "time_seconds": exp_time,
                        # Status
                        "status": "success",
                    }

                except Exception as e:
                    print(f"  ERROR: {e}")
                    result = {
                        "org_id": str(org_id),
                        "model": model_name,
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                    }

                results.append(result)

    elapsed = time.time() - start_time
    print(f"\nReal data experiments completed in {elapsed / 60:.1f} minutes")

    return pd.DataFrame(results)


def save_results(
    synthetic_results: pd.DataFrame | None,
    real_results: pd.DataFrame | None,
    config: BenchmarkConfig,
) -> None:
    """Save experiment results to CSV and JSON."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")

    # Save synthetic results
    if synthetic_results is not None and len(synthetic_results) > 0:
        csv_path = output_dir / f"synthetic_{timestamp}.csv"
        synthetic_results.to_csv(csv_path, index=False)
        print(f"Synthetic results saved to {csv_path}")

        json_path = output_dir / f"synthetic_{timestamp}.json"
        synthetic_results.to_json(json_path, orient="records", indent=2)

        # Summary
        summary = summarize_benchmark(synthetic_results)
        summary_path = output_dir / f"synthetic_{timestamp}_summary.csv"
        summary.to_csv(summary_path)

    # Save real results
    if real_results is not None and len(real_results) > 0:
        csv_path = output_dir / f"real_{timestamp}.csv"
        real_results.to_csv(csv_path, index=False)
        print(f"Real results saved to {csv_path}")

        json_path = output_dir / f"real_{timestamp}.json"
        real_results.to_json(json_path, orient="records", indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    if synthetic_results is not None and len(synthetic_results) > 0:
        print(f"\nSynthetic experiments: {len(synthetic_results)}")
        print(f"  Converged: {synthetic_results['converged'].sum()} / {len(synthetic_results)}")
        print(f"  Mean ELPD-LOO: {synthetic_results['elpd_loo'].mean():.1f}")
        print(f"  Mean test RMSE: {synthetic_results['test_rmse'].mean():.3f}")

    if real_results is not None and len(real_results) > 0:
        success = real_results[real_results["status"] == "success"]
        print(f"\nReal data experiments: {len(real_results)}")
        print(f"  Successful: {len(success)} / {len(real_results)}")
        if len(success) > 0:
            print(f"  Converged: {success['converged'].sum()} / {len(success)}")
            # Robust summaries: median and mean for key metrics
            elpd_arr = np.array(success["elpd_loo"])
            rmse_arr = np.array(success["test_rmse"])
            print(
                f"  ELPD-LOO - Mean: {np.nanmean(elpd_arr):.1f}, Median: {np.nanmedian(elpd_arr):.1f}"
            )
            print(
                f"  Test RMSE - Mean: {np.nanmean(rmse_arr):.3f}, Median: {np.nanmedian(rmse_arr):.3f}"
            )
            # WAIC if available
            if "elpd_waic" in success.columns:
                waic_arr = np.array(success["elpd_waic"])
                valid_waic = waic_arr[~np.isnan(waic_arr)]
                if len(valid_waic) > 0:
                    print(
                        f"  ELPD-WAIC - Mean: {np.mean(valid_waic):.1f}, Median: {np.median(valid_waic):.1f}"
                    )
            # Pareto-k summary for LOO quality
            if "pareto_k_bad" in success.columns:
                total_bad = success["pareto_k_bad"].sum()
                print(f"  Total Pareto-k > 0.7: {total_bad}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Hill Mixture MMM benchmarks (synthetic and/or real data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py                    # Full suite (synthetic + real)
  python scripts/run_benchmark.py --synthetic-only   # Synthetic only
  python scripts/run_benchmark.py --real-only        # Real data only
  python scripts/run_benchmark.py --quick            # Quick test mode
  python scripts/run_benchmark.py --dgp single mixture_k2 --synthetic-only
""",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with reduced experiments",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only synthetic experiments",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Run only real data experiments",
    )
    parser.add_argument(
        "--dgp",
        nargs="+",
        default=None,
        help="DGP scenarios to run (default: all)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (default: [0,1,2,3,4] for synthetic, [0,1,2] for real)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help="Number of MCMC chains (default: 4, quick: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides default)",
    )

    args = parser.parse_args()

    # Get config
    config = get_quick_config() if args.quick else get_default_config()

    # Override config with CLI args
    if args.dgp:
        config.synthetic_dgps = args.dgp
    if args.model:
        config.synthetic_models = args.model
        config.real_models = [
            m for m in args.model if m in ["single_hill", "mixture_k2", "mixture_k3"]
        ]
    if args.seeds:
        config.synthetic_seeds = args.seeds
        config.real_seeds = args.seeds
    if args.chains:
        config.num_chains = args.chains
    if args.output:
        config.output_dir = args.output

    # Set up multi-chain
    numpyro.set_host_device_count(config.num_chains)

    print("\n" + "#" * 60)
    print("# HILL MIXTURE MMM BENCHMARK")
    print("#" * 60)
    print(f"\nConfiguration: {'QUICK' if args.quick else 'FULL'}")
    print(f"Output directory: {config.output_dir}")
    print()

    synthetic_results = None
    real_results = None

    # Run experiments
    if not args.real_only:
        synthetic_results = run_synthetic_experiments(config)

    if not args.synthetic_only:
        real_results = run_real_data_experiments(config)

    # Save results
    save_results(synthetic_results, real_results, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
