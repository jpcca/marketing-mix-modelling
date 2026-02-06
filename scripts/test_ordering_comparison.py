#!/usr/bin/env python3
"""Compare in-MCMC ordering constraint vs post-hoc relabeling.

This experiment validates whether post-hoc relabeling produces
equivalent or better results compared to the cumsum-based ordering
constraint used in the hierarchical model.

Theoretical motivation:
- In-MCMC ordering via cumsum(abs(increments)) introduces non-smooth
  transformations that may degrade HMC efficiency
- Post-hoc relabeling preserves cleaner HMC geometry while achieving
  identifiability through label sorting after inference

Usage:
    # Full experiment (recommended)
    uv run python scripts/test_ordering_comparison.py --use-synthetic --T 200

    # Quick test
    uv run python scripts/test_ordering_comparison.py --use-synthetic --T 100 --warmup 100 --samples 100

Output:
    Results are saved to results/ordering_comparison/:
    - results.json: Full results with all metrics
    - results.csv: Summary table
    - results.txt: Human-readable report
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpyro

# Set device count for parallel chains BEFORE any JAX imports
numpyro.set_host_device_count(2)

import numpy as np  # noqa: E402

from hill_mmm.data import DGPConfig, compute_prior_config, generate_data  # noqa: E402
from hill_mmm.inference import (  # noqa: E402
    compute_convergence_diagnostics,
    compute_predictions,
    compute_predictive_metrics,
    relabel_samples_by_k,
    run_inference,
)
from hill_mmm.models import (  # noqa: E402
    model_hill_mixture_hierarchical_reparam,
    model_hill_mixture_unconstrained,
)


def run_comparison(
    x: np.ndarray,
    y: np.ndarray,
    prior_config: dict,
    warmup: int = 1000,
    samples: int = 2000,
    chains: int = 4,
    K: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Run comparison between constrained and unconstrained models.

    Args:
        x: (T,) spend values
        y: (T,) response values
        prior_config: Prior hyperparameters
        warmup: MCMC warmup iterations
        samples: MCMC sampling iterations
        chains: Number of parallel chains
        K: Number of mixture components
        seed: Random seed

    Returns:
        Dict with comparison results for both models
    """
    results = {}

    # ==== 1. Constrained Model (existing approach) ====
    print("\n[1/2] Running CONSTRAINED model (cumsum ordering)...")
    start = time.time()
    mcmc_constrained = run_inference(
        model_fn=model_hill_mixture_hierarchical_reparam,
        x=x,
        y=y,
        seed=seed,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        prior_config=prior_config,
        K=K,
    )
    time_constrained = time.time() - start

    diag_constrained = compute_convergence_diagnostics(mcmc_constrained)
    pred_constrained = compute_predictions(
        mcmc_constrained,
        model_hill_mixture_hierarchical_reparam,
        x,
        prior_config=prior_config,
        K=K,
    )
    pred_metrics_constrained = compute_predictive_metrics(y, pred_constrained["y"])

    k_constrained = np.array(mcmc_constrained.get_samples()["k"])
    results["constrained"] = {
        "max_rhat": float(diag_constrained["max_rhat"]),
        "min_ess": float(diag_constrained["min_ess_bulk"]),
        "converged": bool(diag_constrained["converged"]),
        "rmse": float(pred_metrics_constrained["rmse"]),
        "coverage_90": float(pred_metrics_constrained["coverage_90"]),
        "time_sec": float(time_constrained),
        "k_means": k_constrained.mean(axis=0).tolist(),
        "k_std": k_constrained.std(axis=0).tolist(),
    }

    print(f"  R-hat: {diag_constrained['max_rhat']:.3f}")
    print(f"  ESS:   {diag_constrained['min_ess_bulk']:.0f}")
    print(f"  RMSE:  {pred_metrics_constrained['rmse']:.2f}")

    # ==== 2. Unconstrained Model + Post-hoc Relabeling ====
    print("\n[2/2] Running UNCONSTRAINED model + post-hoc relabeling...")
    start = time.time()
    mcmc_unconstrained = run_inference(
        model_fn=model_hill_mixture_unconstrained,
        x=x,
        y=y,
        seed=seed,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        prior_config=prior_config,
        K=K,
    )
    time_unconstrained = time.time() - start

    diag_unconstrained = compute_convergence_diagnostics(mcmc_unconstrained)

    # Apply post-hoc relabeling
    samples_raw = mcmc_unconstrained.get_samples()
    samples_relabeled = relabel_samples_by_k(samples_raw)

    # Compute predictive metrics
    pred_unconstrained = compute_predictions(
        mcmc_unconstrained,
        model_hill_mixture_unconstrained,
        x,
        prior_config=prior_config,
        K=K,
    )
    pred_metrics_unconstrained = compute_predictive_metrics(y, pred_unconstrained["y"])

    k_raw = np.array(samples_raw["k"])
    k_relabeled = samples_relabeled["k"]
    results["unconstrained"] = {
        "max_rhat": float(diag_unconstrained["max_rhat"]),
        "min_ess": float(diag_unconstrained["min_ess_bulk"]),
        "converged": bool(diag_unconstrained["converged"]),
        "rmse": float(pred_metrics_unconstrained["rmse"]),
        "coverage_90": float(pred_metrics_unconstrained["coverage_90"]),
        "time_sec": float(time_unconstrained),
        "k_raw_means": k_raw.mean(axis=0).tolist(),
        "k_relabeled_means": k_relabeled.mean(axis=0).tolist(),
        "k_relabeled_std": k_relabeled.std(axis=0).tolist(),
    }

    print(f"  R-hat: {diag_unconstrained['max_rhat']:.3f}")
    print(f"  ESS:   {diag_unconstrained['min_ess_bulk']:.0f}")
    print(f"  RMSE:  {pred_metrics_unconstrained['rmse']:.2f}")

    return results


def generate_report(results: dict[str, Any], config: dict[str, Any]) -> str:
    """Generate human-readable report."""
    c = results["constrained"]
    u = results["unconstrained"]

    lines = [
        "=" * 80,
        "ORDERING CONSTRAINT COMPARISON EXPERIMENT",
        "=" * 80,
        "",
        "CONFIGURATION:",
        f"  Timestamp:   {config['timestamp']}",
        f"  Data:        {config['data_source']}",
        f"  T:           {config['T']}",
        f"  K:           {config['K']}",
        f"  Warmup:      {config['warmup']}",
        f"  Samples:     {config['samples']}",
        f"  Chains:      {config['chains']}",
        f"  Seed:        {config['seed']}",
        "",
        "=" * 80,
        "RESULTS COMPARISON",
        "=" * 80,
        "",
        f"{'Metric':<25} | {'Constrained':>15} | {'Unconstrained':>15} | {'Winner':>12}",
        "-" * 80,
    ]

    # R-hat (lower is better)
    rhat_winner = "Constrained" if c["max_rhat"] <= u["max_rhat"] else "Unconstrained"
    lines.append(
        f"{'Max R-hat':<25} | {c['max_rhat']:>15.4f} | {u['max_rhat']:>15.4f} | {rhat_winner:>12}"
    )

    # ESS (higher is better)
    ess_winner = "Constrained" if c["min_ess"] >= u["min_ess"] else "Unconstrained"
    lines.append(
        f"{'Min ESS (bulk)':<25} | {c['min_ess']:>15.0f} | {u['min_ess']:>15.0f} | {ess_winner:>12}"
    )

    # RMSE (lower is better)
    rmse_winner = "Constrained" if c["rmse"] <= u["rmse"] else "Unconstrained"
    lines.append(f"{'RMSE':<25} | {c['rmse']:>15.2f} | {u['rmse']:>15.2f} | {rmse_winner:>12}")

    # Coverage (closer to 0.90 is better)
    cov_diff_c = abs(c["coverage_90"] - 0.90)
    cov_diff_u = abs(u["coverage_90"] - 0.90)
    cov_winner = "Constrained" if cov_diff_c <= cov_diff_u else "Unconstrained"
    lines.append(
        f"{'90% Coverage':<25} | {c['coverage_90']:>14.2%} | {u['coverage_90']:>14.2%} | {cov_winner:>12}"
    )

    # Time (lower is better)
    time_winner = "Constrained" if c["time_sec"] <= u["time_sec"] else "Unconstrained"
    lines.append(
        f"{'Time (sec)':<25} | {c['time_sec']:>15.1f} | {u['time_sec']:>15.1f} | {time_winner:>12}"
    )

    # Convergence status
    conv_c = "Yes" if c["converged"] else "No"
    conv_u = "Yes" if u["converged"] else "No"
    conv_winner = "-"
    if c["converged"] and not u["converged"]:
        conv_winner = "Constrained"
    elif u["converged"] and not c["converged"]:
        conv_winner = "Unconstrained"
    lines.append(
        f"{'Converged (R-hat<1.05)':<25} | {conv_c:>15} | {conv_u:>15} | {conv_winner:>12}"
    )

    lines.extend(
        [
            "-" * 80,
            "",
            "K VALUES ANALYSIS:",
            f"  Constrained k means:   {c['k_means']}",
            f"  Constrained k std:     {c['k_std']}",
            f"  Relabeled k means:     {u['k_relabeled_means']}",
            f"  Relabeled k std:       {u['k_relabeled_std']}",
            f"  Raw k means (before relabeling): {u['k_raw_means']}",
            "",
        ]
    )

    # Check ordering
    k_c_ordered = all(c["k_means"][i] < c["k_means"][i + 1] for i in range(len(c["k_means"]) - 1))
    k_u_ordered = all(
        u["k_relabeled_means"][i] < u["k_relabeled_means"][i + 1]
        for i in range(len(u["k_relabeled_means"]) - 1)
    )
    lines.append(f"  Constrained k properly ordered: {k_c_ordered}")
    lines.append(f"  Relabeled k properly ordered:   {k_u_ordered}")

    # Overall recommendation
    wins = {"Constrained": 0, "Unconstrained": 0}
    for w in [rhat_winner, ess_winner, rmse_winner, cov_winner, time_winner]:
        if w in wins:
            wins[w] += 1

    lines.extend(
        [
            "",
            "=" * 80,
            "CONCLUSION",
            "=" * 80,
            f"  Constrained wins: {wins['Constrained']}/5",
            f"  Unconstrained wins: {wins['Unconstrained']}/5",
            "",
        ]
    )

    if wins["Unconstrained"] > wins["Constrained"]:
        lines.append("RECOMMENDATION: Post-hoc relabeling (unconstrained) performs better.")
        lines.append(
            "  Consider switching to model_hill_mixture_unconstrained + relabel_samples_by_k()"
        )
    elif wins["Constrained"] > wins["Unconstrained"]:
        lines.append("RECOMMENDATION: In-MCMC ordering (constrained) performs better.")
        lines.append("  The current approach is working well.")
    else:
        lines.append("RECOMMENDATION: Both approaches perform similarly.")
        lines.append(
            "  Post-hoc relabeling is theoretically cleaner but not clearly superior empirically."
        )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_results(results: dict[str, Any], config: dict[str, Any], output_dir: Path) -> None:
    """Save results to JSON, CSV, and TXT files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save summary as CSV
    csv_path = output_dir / "results.csv"
    c = results["constrained"]
    u = results["unconstrained"]
    csv_lines = [
        "metric,constrained,unconstrained,winner",
        f"max_rhat,{c['max_rhat']:.4f},{u['max_rhat']:.4f},{'constrained' if c['max_rhat'] <= u['max_rhat'] else 'unconstrained'}",
        f"min_ess,{c['min_ess']:.0f},{u['min_ess']:.0f},{'constrained' if c['min_ess'] >= u['min_ess'] else 'unconstrained'}",
        f"rmse,{c['rmse']:.4f},{u['rmse']:.4f},{'constrained' if c['rmse'] <= u['rmse'] else 'unconstrained'}",
        f"coverage_90,{c['coverage_90']:.4f},{u['coverage_90']:.4f},{'constrained' if abs(c['coverage_90'] - 0.9) <= abs(u['coverage_90'] - 0.9) else 'unconstrained'}",
        f"time_sec,{c['time_sec']:.1f},{u['time_sec']:.1f},{'constrained' if c['time_sec'] <= u['time_sec'] else 'unconstrained'}",
        f"converged,{c['converged']},{u['converged']},-",
    ]
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"Saved CSV: {csv_path}")

    # Save human-readable report
    txt_path = output_dir / "results.txt"
    report = generate_report(results, config)
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Saved TXT: {txt_path}")

    # Print the report to console as well
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description="Compare in-MCMC ordering vs post-hoc relabeling")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data instead of real data",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/conjura_mmm_data.csv",
        help="Path to real dataset (if not using synthetic)",
    )
    parser.add_argument(
        "--T", type=int, default=200, help="Time series length (for synthetic data)"
    )
    parser.add_argument("--warmup", type=int, default=1000, help="MCMC warmup iterations")
    parser.add_argument("--samples", type=int, default=2000, help="MCMC sampling iterations")
    parser.add_argument("--chains", type=int, default=4, help="Number of parallel chains")
    parser.add_argument("--K", type=int, default=3, help="Number of mixture components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dgp",
        type=str,
        default="mixture_k3",
        choices=["single", "mixture_k2", "mixture_k3", "mixture_k5"],
        help="Data generating process for synthetic data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ordering_comparison",
        help="Output directory for results",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 75)
    print("ORDERING CONSTRAINT COMPARISON EXPERIMENT")
    print("=" * 75)
    print(
        f"Settings: warmup={args.warmup}, samples={args.samples}, chains={args.chains}, K={args.K}"
    )

    # Load or generate data
    if args.use_synthetic:
        print(f"\nUsing SYNTHETIC data: DGP={args.dgp}, T={args.T}")
        dgp_config = DGPConfig(dgp_type=args.dgp, T=args.T, seed=args.seed)
        x, y, _ = generate_data(dgp_config)
        prior_config = compute_prior_config(x, y)
        data_source = f"synthetic_{args.dgp}_T{args.T}"
    else:
        # Use real data
        if not Path(args.data).exists():
            print(f"\nERROR: Data file not found: {args.data}")
            print("Use --use-synthetic to run with synthetic data.")
            sys.exit(1)

        print(f"\nUsing REAL data from: {args.data}")
        from hill_mmm.data_loader import (
            TimeSeriesConfig,
            load_timeseries,
            select_representative_timeseries,
        )

        # Select first representative organization
        org_ids = select_representative_timeseries(
            args.data, n=1, seed=args.seed, min_length=200, min_channels=2
        )
        ts_config = TimeSeriesConfig(organisation_id=org_ids[0], aggregate_spend=True)
        data = load_timeseries(args.data, ts_config)
        x, y = data.x, data.y
        prior_config = compute_prior_config(x, y)
        data_source = f"real_{org_ids[0][:16]}"
        print(f"Organization: {org_ids[0][:16]}...")

    print(f"Data shape: T={len(y)}, x_range=[{x.min():.1f}, {x.max():.1f}]")

    # Configuration for saving
    config = {
        "timestamp": timestamp,
        "data_source": data_source,
        "T": len(y),
        "K": args.K,
        "warmup": args.warmup,
        "samples": args.samples,
        "chains": args.chains,
        "seed": args.seed,
    }

    # Run comparison
    results = run_comparison(
        x=x,
        y=y,
        prior_config=prior_config,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        K=args.K,
        seed=args.seed,
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, config, output_dir)


if __name__ == "__main__":
    main()
