#!/usr/bin/env python3
"""Inference accuracy comparison between HalfNormal(0.1) and LogNormal hyperpriors.

Compares predictive performance using real data (Conjura dataset):
- Train/test split (80/20)
- LOO-CV and WAIC
- Test RMSE, MAE
- 90% CI coverage

Usage:
    # Run in each worktree:
    python scripts/run_inference_accuracy_comparison.py --label halfnormal
    python scripts/run_inference_accuracy_comparison.py --label lognormal

    # Results saved to: results/inference_accuracy_<label>.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import numpyro
import pandas as pd

# Set device count for parallel chains BEFORE any JAX imports
numpyro.set_host_device_count(2)

from hill_mmm.data import compute_prior_config  # noqa: E402
from hill_mmm.data_loader import (  # noqa: E402
    TimeSeriesConfig,
    load_timeseries,
    select_representative_timeseries,
)
from hill_mmm.inference import (  # noqa: E402
    compute_convergence_diagnostics,
    compute_loo,
    compute_predictions,
    compute_waic,
    run_inference,
)
from hill_mmm.models import model_hill_mixture_hierarchical_reparam  # noqa: E402


def compute_inference_metrics(
    mcmc,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    prior_config: dict,
    K: int = 3,
) -> dict:
    """Compute comprehensive inference accuracy metrics."""
    # Convergence diagnostics
    diag = compute_convergence_diagnostics(mcmc)

    # LOO-CV and WAIC (computed from mcmc directly)
    try:
        loo_result = compute_loo(mcmc)
        loo_score = float(loo_result["elpd_loo"])
        loo_se = float(loo_result.get("se", np.nan))
    except Exception as e:
        print(f"  LOO computation failed: {e}")
        loo_score = np.nan
        loo_se = np.nan

    try:
        waic_result = compute_waic(mcmc)
        waic_score = float(waic_result["elpd_waic"])
        waic_se = float(waic_result.get("se", np.nan))
    except Exception as e:
        print(f"  WAIC computation failed: {e}")
        waic_score = np.nan
        waic_se = np.nan

    # Predictive performance on test set
    predictions = compute_predictions(
        mcmc, model_hill_mixture_hierarchical_reparam, x_test, prior_config, K=K
    )
    # Get y samples and compute metrics
    y_samples = predictions.get("y", predictions.get("mu_expected", None))
    if y_samples is not None:
        y_pred_mean = np.mean(y_samples, axis=0)
    else:
        y_pred_mean = np.zeros_like(y_test)

    # RMSE and MAE
    rmse = float(np.sqrt(np.mean((y_test - y_pred_mean) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred_mean)))

    # 90% CI coverage using quantiles
    if y_samples is not None:
        q05 = np.quantile(y_samples, 0.05, axis=0)
        q95 = np.quantile(y_samples, 0.95, axis=0)
        coverage_90 = float(np.mean((y_test >= q05) & (y_test <= q95)))
        q25 = np.quantile(y_samples, 0.25, axis=0)
        q75 = np.quantile(y_samples, 0.75, axis=0)
        coverage_50 = float(np.mean((y_test >= q25) & (y_test <= q75)))
    else:
        coverage_90 = np.nan
        coverage_50 = np.nan

    return {
        "max_rhat": diag["max_rhat"],
        "min_ess": diag["min_ess_bulk"],
        "converged": diag["max_rhat"] <= 1.05 and diag["min_ess_bulk"] >= 100,
        "loo": loo_score,
        "loo_se": loo_se,
        "waic": waic_score,
        "waic_se": waic_se,
        "test_rmse": rmse,
        "test_mae": mae,
        "coverage_90": coverage_90,
        "coverage_50": coverage_50,
    }


def run_inference_comparison(
    csv_path: str,
    label: str,
    n_orgs: int = 3,
    warmup: int = 500,
    samples: int = 500,
    chains: int = 2,
    K: int = 3,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    """Run inference accuracy comparison on representative organizations."""
    print(f"Selecting {n_orgs} representative organizations...")
    org_ids = select_representative_timeseries(
        csv_path,
        n=n_orgs,
        seed=seed,
        min_length=200,
        min_channels=2,
    )
    print(f"Selected: {org_ids}")

    results = []

    for i, org_id in enumerate(org_ids):
        print(f"\n[{i + 1}/{n_orgs}] Testing organization: {org_id[:12]}...")

        try:
            # Load data
            config = TimeSeriesConfig(
                organisation_id=org_id,
                aggregate_spend=True,
            )
            data = load_timeseries(csv_path, config)
            T = len(data.y)

            # Train/test split
            split_idx = int(T * train_ratio)
            x_train, y_train = data.x[:split_idx], data.y[:split_idx]
            x_test, y_test = data.x[split_idx:], data.y[split_idx:]

            print(f"  Data: T={T}, train={len(y_train)}, test={len(y_test)}")

            # Compute data-adaptive priors
            prior_config = compute_prior_config(x_train, y_train)

            # Run MCMC
            start_time = time.time()
            mcmc = run_inference(
                model_fn=model_hill_mixture_hierarchical_reparam,
                x=x_train,
                y=y_train,
                seed=seed + i,
                num_warmup=warmup,
                num_samples=samples,
                num_chains=chains,
                prior_config=prior_config,
                K=K,
            )
            elapsed = time.time() - start_time

            # Compute metrics
            metrics = compute_inference_metrics(
                mcmc, x_train, y_train, x_test, y_test, prior_config, K=K
            )

            result = {
                "label": label,
                "org_id": org_id,
                "T": T,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "time_sec": elapsed,
                "error": None,
                **metrics,
            }

            status = "✓" if metrics["converged"] else "✗"
            print(
                f"  Result: R-hat={metrics['max_rhat']:.3f}, "
                f"Test RMSE={metrics['test_rmse']:.2f}, "
                f"90% Coverage={metrics['coverage_90']:.1%} {status}"
            )

        except Exception as e:
            result = {
                "label": label,
                "org_id": org_id,
                "T": None,
                "train_size": None,
                "test_size": None,
                "time_sec": None,
                "max_rhat": None,
                "min_ess": None,
                "converged": False,
                "loo": None,
                "loo_se": None,
                "waic": None,
                "waic_se": None,
                "test_rmse": None,
                "test_mae": None,
                "coverage_90": None,
                "coverage_50": None,
                "error": str(e),
            }
            print(f"  ERROR: {e}")

        results.append(result)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame) -> None:
    """Print formatted summary."""
    label = df["label"].iloc[0]
    print("\n" + "=" * 80)
    print(f"INFERENCE ACCURACY SUMMARY ({label.upper()})")
    print("=" * 80)

    valid = df[df["error"].isna()]
    if len(valid) == 0:
        print("No valid results.")
        return

    print(
        f"\n{'Org ID':<16} | {'RMSE':>8} | {'MAE':>8} | {'90% Cov':>8} | {'LOO':>10} | {'Conv':>6}"
    )
    print("-" * 80)

    for _, row in valid.iterrows():
        conv = "✓" if bool(row["converged"]) else "✗"
        print(
            f"{row['org_id'][:14]:<16} | {row['test_rmse']:>8.2f} | {row['test_mae']:>8.2f} | "
            f"{row['coverage_90']:>7.1%} | {row['loo']:>10.1f} | {conv:>6}"
        )

    print("-" * 80)
    print("\nAGGREGATE METRICS:")
    print(f"  Mean Test RMSE:    {valid['test_rmse'].mean():.2f} (±{valid['test_rmse'].std():.2f})")
    print(f"  Mean Test MAE:     {valid['test_mae'].mean():.2f} (±{valid['test_mae'].std():.2f})")
    print(f"  Mean 90% Coverage: {valid['coverage_90'].mean():.1%} (target: 90%)")
    print(f"  Mean 50% Coverage: {valid['coverage_50'].mean():.1%} (target: 50%)")
    print(f"  Mean LOO:          {valid['loo'].mean():.1f} (±{valid['loo'].std():.1f})")
    print(f"  Convergence Rate:  {valid['converged'].mean():.0%}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inference accuracy comparison")
    parser.add_argument(
        "--data",
        type=str,
        default="data/conjura_mmm_data.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=["halfnormal", "lognormal"],
        help="Label for this experiment (halfnormal or lognormal)",
    )
    parser.add_argument(
        "--n-orgs",
        type=int,
        default=3,
        help="Number of organizations to test",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="MCMC warmup iterations",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="MCMC sampling iterations",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of parallel chains",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Number of mixture components",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Verify data file exists
    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Running inference accuracy comparison: {args.label.upper()}")
    print(
        f"Settings: warmup={args.warmup}, samples={args.samples}, chains={args.chains}, K={args.K}"
    )

    # Run comparison
    df = run_inference_comparison(
        csv_path=args.data,
        label=args.label,
        n_orgs=args.n_orgs,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        K=args.K,
        seed=args.seed,
    )

    # Save results
    output_file = output_dir / f"inference_accuracy_{args.label}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print_summary(df)

    # Exit code based on convergence
    n_converged = df["converged"].sum()
    sys.exit(0 if n_converged == len(df) else 1)


if __name__ == "__main__":
    main()
