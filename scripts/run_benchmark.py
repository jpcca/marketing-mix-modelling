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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import numpyro
import pandas as pd


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
        # Synthetic: 4 DGPs × 4 models × 5 seeds = 80 experiments
        synthetic_dgps=["single", "mixture_k2", "mixture_k3", "mixture_k5"],
        synthetic_models=["single_hill", "mixture_k2", "mixture_k3", "sparse_k5"],
        synthetic_seeds=[0, 1, 2, 3, 4],
        # Real: 5 orgs × 3 models × 3 seeds = 45 experiments
        real_n_orgs=5,
        real_models=["single_hill", "mixture_k2", "mixture_k3"],
        real_seeds=[0, 1, 2],
        # MCMC
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
    from hill_mmm.benchmark import run_benchmark_suite

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
    from hill_mmm.benchmark import MODEL_SPECS
    from hill_mmm.data import compute_prior_config
    from hill_mmm.data_loader import load_real_data, select_representative_timeseries
    from hill_mmm.inference import (
        compute_convergence_diagnostics,
        compute_loo,
        compute_predictions,
        compute_predictive_metrics,
        relabel_samples_by_k,
        run_inference,
    )

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
    selected = select_representative_timeseries(
        df_real,
        n_timeseries=config.real_n_orgs,
        selection_criteria="most_data",
    )

    # Get model specs
    model_lookup = {m.name: m for m in MODEL_SPECS}

    results = []
    total = n_exp
    current = 0
    start_time = time.time()

    for org_id, org_df in selected.groupby("organization_id"):
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
                    # Run inference
                    mcmc = run_inference(
                        model_spec.fn,
                        x_train,
                        y_train,
                        seed=seed,
                        num_warmup=config.num_warmup,
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

                    # Compute predictions
                    pred_train = compute_predictions(
                        mcmc, model_spec.fn, x_train, prior_config=prior_config, **model_spec.kwargs
                    )
                    pred_test = compute_predictions(
                        mcmc, model_spec.fn, x_test, prior_config=prior_config, **model_spec.kwargs
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
                        # Convergence
                        "max_rhat": convergence["max_rhat"],
                        "min_ess_bulk": convergence["min_ess_bulk"],
                        "converged": convergence["converged"],
                        # LOO
                        "elpd_loo": loo.get("elpd_loo"),
                        "loo_se": loo.get("se"),
                        "p_loo": loo.get("p_loo"),
                        "pareto_k_bad": loo.get("n_pareto_k_bad", 0),
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
    from hill_mmm.benchmark import summarize_benchmark

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
            print(f"  Mean ELPD-LOO: {success['elpd_loo'].mean():.1f}")
            print(f"  Mean test RMSE: {success['test_rmse'].mean():.3f}")


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
