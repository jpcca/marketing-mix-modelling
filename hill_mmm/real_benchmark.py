"""Benchmark runner for real Conjura MMM data.

Runs model evaluation on real eCommerce time series data.
Each organisation is treated independently (no data sharing).

Usage:
    from hill_mmm.real_benchmark import run_real_benchmark

    results = run_real_benchmark(
        csv_path="data/conjura_mmm_data.csv",
        n_timeseries=5,
    )
    print(results.to_markdown())
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpyro
import pandas as pd

from .benchmark import MODEL_SPECS, ModelSpec
from .data import compute_prior_config
from .data_loader import (
    LoadedData,
    TimeSeriesConfig,
    load_timeseries,
    select_representative_timeseries,
)
from .inference import (
    compute_convergence_diagnostics,
    compute_loo,
    compute_predictions,
    compute_predictive_metrics,
    compute_waic,
    run_inference,
)
from .metrics import compute_delta_loo, compute_effective_k


@dataclass
class RealBenchmarkConfig:
    """Configuration for real data benchmarking.

    Attributes:
        csv_path: Path to Conjura dataset CSV
        target_col: Target variable to model
        aggregate_spend: Sum all channels (Phase 1)
        n_timeseries: Number of time series to test (None = all)
        train_ratio: Train/test split ratio
        num_warmup: MCMC warmup iterations
        num_samples: MCMC sample iterations
        num_chains: Number of MCMC chains
        seed: Random seed for reproducibility
    """

    csv_path: str | Path = "data/conjura_mmm_data.csv"
    target_col: str = "all_purchases"
    aggregate_spend: bool = True
    n_timeseries: int | None = None
    train_ratio: float = 0.75
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 4
    seed: int = 42


def run_real_experiment(
    data: LoadedData,
    model_spec: ModelSpec,
    train_ratio: float = 0.75,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single experiment on real data.

    Args:
        data: Loaded time series data
        model_spec: Model specification
        train_ratio: Fraction for training
        num_warmup: MCMC warmup iterations
        num_samples: MCMC sample iterations
        num_chains: MCMC chains
        seed: Random seed

    Returns:
        Dict with evaluation metrics
    """
    x, y = data.x, data.y
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
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        prior_config=prior_config,
        **model_spec.kwargs,
    )

    # Compute metrics
    convergence = compute_convergence_diagnostics(mcmc)
    loo = compute_loo(mcmc)
    waic = compute_waic(mcmc)
    effective_k = compute_effective_k(mcmc)

    # Predictive metrics
    pred_train = compute_predictions(
        mcmc, model_spec.fn, x_train, prior_config=prior_config, **model_spec.kwargs
    )
    pred_test = compute_predictions(
        mcmc, model_spec.fn, x_test, prior_config=prior_config, **model_spec.kwargs
    )

    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

    return {
        "organisation_id": data.meta.get("organisation_id"),
        "territory": data.meta.get("territory"),
        "vertical": data.meta.get("organisation_vertical"),
        "model": model_spec.name,
        "seed": seed,
        "T": T,
        "T_train": T_train,
        "n_channels": data.meta.get("n_channels"),
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
        # Data characteristics
        "total_spend": data.meta.get("total_spend"),
        "total_target": data.meta.get("total_target"),
        "spend_nonzero_ratio": data.meta.get("spend_nonzero_ratio"),
    }


def run_real_benchmark(
    config: RealBenchmarkConfig | None = None,
    model_names: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run benchmark on real Conjura data.

    Args:
        config: Benchmark configuration (default: RealBenchmarkConfig())
        model_names: Models to test (default: all)
        verbose: Print progress

    Returns:
        DataFrame with benchmark results
    """
    if config is None:
        config = RealBenchmarkConfig()

    # Set up multi-chain
    numpyro.set_host_device_count(config.num_chains)

    # Select time series
    if config.n_timeseries is not None:
        org_ids = select_representative_timeseries(
            config.csv_path,
            n=config.n_timeseries,
            seed=config.seed,
        )
    else:
        from .data_loader import list_timeseries

        ts_info = list_timeseries(config.csv_path)
        org_ids = ts_info["organisation_id"].tolist()

    # Filter models
    if model_names is None:
        model_names = [m.name for m in MODEL_SPECS]
    models = [m for m in MODEL_SPECS if m.name in model_names]

    results = []
    total = len(org_ids) * len(models)
    current = 0

    for org_id in org_ids:
        # Load data for this organisation
        try:
            ts_config = TimeSeriesConfig(
                organisation_id=org_id,
                target_col=config.target_col,
                aggregate_spend=config.aggregate_spend,
            )
            data = load_timeseries(config.csv_path, ts_config)
        except ValueError as e:
            if verbose:
                print(f"Skipping {org_id}: {e}")
            continue

        for model_spec in models:
            current += 1
            if verbose:
                print(f"[{current}/{total}] Org={org_id[:8]}..., Model={model_spec.name}")

            try:
                result = run_real_experiment(
                    data,
                    model_spec,
                    train_ratio=config.train_ratio,
                    num_warmup=config.num_warmup,
                    num_samples=config.num_samples,
                    num_chains=config.num_chains,
                    seed=config.seed,
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

    df = pd.DataFrame(results)

    # Add delta LOO relative to single_hill baseline
    if len(df) > 0:
        df = _add_delta_loo_real(df)

    return df


def _add_delta_loo_real(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta LOO columns relative to single_hill baseline."""
    df = df.copy()
    df["delta_loo"] = None
    df["delta_loo_significant"] = None

    for org_id, group in df.groupby("organisation_id"):
        baseline_row = group[group["model"] == "single_hill"]
        if len(baseline_row) == 0:
            continue

        elpd_val = baseline_row["elpd_loo"].iloc[0]  # type: ignore[union-attr]
        se_val = baseline_row["loo_se"].iloc[0]  # type: ignore[union-attr]
        baseline_loo = {
            "elpd_loo": float(elpd_val),
            "se": float(se_val),
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


def summarize_real_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize real benchmark results by model.

    Args:
        df: Raw results DataFrame

    Returns:
        Summary DataFrame with mean ± std across organisations
    """
    metrics = [
        "elpd_loo",
        "test_rmse",
        "train_rmse",
        "test_coverage_90",
        "effective_k_mean",
        "delta_loo",
    ]

    result = df.groupby("model")[metrics].agg(["mean", "std", "count"]).round(2)

    return pd.DataFrame(result)


def print_real_benchmark_table(df: pd.DataFrame) -> None:
    """Print real benchmark results.

    Args:
        df: Results DataFrame
    """
    print("\n" + "=" * 80)
    print("REAL DATA BENCHMARK RESULTS")
    print("=" * 80)

    # Summary by model
    _summary = summarize_real_benchmark(df)  # noqa: F841 - computed for potential future use
    print("\nModel Summary (across organisations):")
    print("-" * 60)

    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        n_orgs = len(model_data)
        loo_mean = model_data["elpd_loo"].mean()
        test_rmse = model_data["test_rmse"].mean()
        cov = model_data["test_coverage_90"].mean()
        eff_k = model_data["effective_k_mean"].mean()
        delta = float(model_data["delta_loo"].mean())
        converged = float(model_data["converged"].mean())

        is_valid_delta = not pd.isna(delta)
        delta_str = f"{delta:+.1f}" if is_valid_delta else "N/A"
        print(
            f"{model:<15} n={n_orgs:<3} LOO={loo_mean:>7.1f}  "
            f"RMSE={test_rmse:>7.2f}  Cov90={cov:>5.1%}  "
            f"EffK={eff_k:>4.1f}  ΔLOO={delta_str:>6}  Conv={converged:.0%}"
        )

    # Per-organisation details
    print("\n\nPer-Organisation Results:")
    print("-" * 80)

    for org_id in df["organisation_id"].unique():
        org_data = df[df["organisation_id"] == org_id]
        vertical_val = org_data["vertical"].iloc[0]  # type: ignore[union-attr]
        T_val = org_data["T"].iloc[0]  # type: ignore[union-attr]
        vertical = str(vertical_val)
        T = int(T_val)
        print(f"\n{org_id[:20]}...  ({vertical}, T={T})")

        for _, row in org_data.iterrows():
            row_delta = row["delta_loo"]
            is_valid = row_delta is not None and not bool(pd.isna(row_delta))
            delta_str = f"{row_delta:+.1f}" if is_valid else "N/A"
            sig = "*" if row.get("delta_loo_significant", False) else " "
            print(
                f"  {row['model']:<15} LOO={row['elpd_loo']:>7.1f}  "
                f"RMSE={row['test_rmse']:>7.2f}  ΔLOO={delta_str}{sig}"
            )
