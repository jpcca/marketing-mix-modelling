#!/usr/bin/env python
"""Test script for label switching diagnostics.

This script validates the improved convergence diagnostics for mixture models:
1. Label-invariant diagnostics (R-hat on log-likelihood)
2. Rank-normalized R-hat
3. Diagnostics on relabeled samples
4. Label switching detection

Usage:
    python scripts/test_label_switching_diagnostics.py [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax  # noqa: E402

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

from hill_mmm.data import DGPConfig, generate_data  # noqa: E402
from hill_mmm.inference import (  # noqa: E402
    check_label_switching,
    compute_comprehensive_mixture_diagnostics,
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


def run_diagnostics_comparison(
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
) -> dict:
    """Run comprehensive diagnostic comparison between model variants.

    Args:
        num_warmup: Warmup iterations
        num_samples: Samples per chain
        num_chains: Number of chains
        seed: Random seed

    Returns:
        Dict with comparison results
    """
    print("=" * 70)
    print("LABEL SWITCHING DIAGNOSTICS VALIDATION")
    print("=" * 70)

    # Generate synthetic data
    print("\n[1/5] Generating synthetic data (mixture_k3)...")
    config = DGPConfig(dgp_type="mixture_k3", T=200, seed=seed)
    x, y, meta = generate_data(config)
    print(f"  Data shape: x={x.shape}, y={y.shape}")
    print(f"  True k values: {meta.get('k', 'N/A')}")

    results = {"data": {"T": len(x), "dgp": "mixture_k3", "seed": seed}}

    # =========================================================================
    # Model 1: Constrained (ordered k)
    # =========================================================================
    print("\n[2/5] Running constrained model (ordered k)...")
    t0 = time.time()
    mcmc_constrained = run_inference(
        model_hill_mixture_hierarchical_reparam,
        x,
        y,
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        K=3,
    )
    time_constrained = time.time() - t0
    print(f"  Completed in {time_constrained:.1f}s")

    # Standard diagnostics
    std_diag_constrained = compute_convergence_diagnostics(mcmc_constrained)
    print(f"  Standard R-hat: {std_diag_constrained['max_rhat']:.3f}")
    print(f"  Standard ESS: {std_diag_constrained['min_ess_bulk']:.0f}")

    # Comprehensive diagnostics
    print("  Computing comprehensive diagnostics...")
    comp_diag_constrained = compute_comprehensive_mixture_diagnostics(
        mcmc_constrained, x, y, method="rank"
    )
    print(f"  Log-likelihood R-hat: {comp_diag_constrained['label_invariant']['rhat_log_lik']:.3f}")
    print(f"  Status: {comp_diag_constrained['status']}")

    # Predictive metrics
    pred_constrained = compute_predictions(
        mcmc_constrained, model_hill_mixture_hierarchical_reparam, x, K=3
    )
    metrics_constrained = compute_predictive_metrics(y, pred_constrained["y"])
    print(f"  RMSE: {metrics_constrained['rmse']:.2f}")
    print(f"  Coverage 90%: {metrics_constrained['coverage_90']:.1%}")

    results["constrained"] = {
        "time_sec": time_constrained,
        "standard_diagnostics": {
            "max_rhat": std_diag_constrained["max_rhat"],
            "min_ess_bulk": std_diag_constrained["min_ess_bulk"],
            "converged": std_diag_constrained["converged"],
        },
        "comprehensive_diagnostics": {
            "label_invariant": comp_diag_constrained["label_invariant"],
            "relabeled": comp_diag_constrained["relabeled"],
            "switching": comp_diag_constrained["label_switching"],
            "status": comp_diag_constrained["status"],
            "recommendation": comp_diag_constrained["recommendation"],
        },
        "predictive": {
            "rmse": metrics_constrained["rmse"],
            "coverage_90": metrics_constrained["coverage_90"],
        },
    }

    # =========================================================================
    # Model 2: Unconstrained + Post-hoc relabeling
    # =========================================================================
    print("\n[3/5] Running unconstrained model (no ordering)...")
    t0 = time.time()
    mcmc_unconstrained = run_inference(
        model_hill_mixture_unconstrained,
        x,
        y,
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        K=3,
    )
    time_unconstrained = time.time() - t0
    print(f"  Completed in {time_unconstrained:.1f}s")

    # Standard diagnostics (before relabeling)
    std_diag_unconstrained = compute_convergence_diagnostics(mcmc_unconstrained)
    print(f"  Standard R-hat (raw): {std_diag_unconstrained['max_rhat']:.3f}")

    # Check label switching
    samples_raw = mcmc_unconstrained.get_samples()
    switching_info = check_label_switching(samples_raw)
    print(f"  Label switching rate: {switching_info['switching_rate']:.1%}")
    print(f"  Unique orderings: {switching_info['n_unique_orderings']}")

    # Relabel samples
    print("  Relabeling samples by k...")
    samples_relabeled = relabel_samples_by_k(samples_raw)
    k_raw_mean = samples_raw["k"].mean(axis=0)
    k_relabeled_mean = samples_relabeled["k"].mean(axis=0)
    print(f"  Raw k means: {k_raw_mean}")
    print(f"  Relabeled k means: {k_relabeled_mean}")

    # Comprehensive diagnostics
    print("  Computing comprehensive diagnostics...")
    comp_diag_unconstrained = compute_comprehensive_mixture_diagnostics(
        mcmc_unconstrained, x, y, method="rank"
    )
    print(
        f"  Log-likelihood R-hat: {comp_diag_unconstrained['label_invariant']['rhat_log_lik']:.3f}"
    )
    print(f"  Status: {comp_diag_unconstrained['status']}")

    # Predictive metrics
    pred_unconstrained = compute_predictions(
        mcmc_unconstrained, model_hill_mixture_unconstrained, x, K=3
    )
    metrics_unconstrained = compute_predictive_metrics(y, pred_unconstrained["y"])
    print(f"  RMSE: {metrics_unconstrained['rmse']:.2f}")
    print(f"  Coverage 90%: {metrics_unconstrained['coverage_90']:.1%}")

    results["unconstrained"] = {
        "time_sec": time_unconstrained,
        "standard_diagnostics": {
            "max_rhat": std_diag_unconstrained["max_rhat"],
            "min_ess_bulk": std_diag_unconstrained["min_ess_bulk"],
            "converged": std_diag_unconstrained["converged"],
        },
        "label_switching": {
            "switching_rate": switching_info["switching_rate"],
            "n_unique_orderings": switching_info["n_unique_orderings"],
            "k_raw_mean": k_raw_mean.tolist(),
            "k_relabeled_mean": k_relabeled_mean.tolist(),
        },
        "comprehensive_diagnostics": {
            "label_invariant": comp_diag_unconstrained["label_invariant"],
            "relabeled": comp_diag_unconstrained["relabeled"],
            "switching": comp_diag_unconstrained["label_switching"],
            "status": comp_diag_unconstrained["status"],
            "recommendation": comp_diag_unconstrained["recommendation"],
        },
        "predictive": {
            "rmse": metrics_unconstrained["rmse"],
            "coverage_90": metrics_unconstrained["coverage_90"],
        },
    }

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n[4/5] Comparison Summary")
    print("=" * 70)
    print(f"{'Metric':<40} {'Constrained':>12} {'Unconstrained':>12}")
    print("-" * 70)
    print(f"{'Time (sec)':<40} {time_constrained:>12.1f} {time_unconstrained:>12.1f}")
    print(
        f"{'Standard R-hat':<40} {std_diag_constrained['max_rhat']:>12.3f} {std_diag_unconstrained['max_rhat']:>12.3f}"
    )
    print(
        f"{'Log-likelihood R-hat':<40} {comp_diag_constrained['label_invariant']['rhat_log_lik']:>12.3f} {comp_diag_unconstrained['label_invariant']['rhat_log_lik']:>12.3f}"
    )
    print(
        f"{'RMSE':<40} {metrics_constrained['rmse']:>12.2f} {metrics_unconstrained['rmse']:>12.2f}"
    )
    print(
        f"{'Coverage 90%':<40} {metrics_constrained['coverage_90']:>12.1%} {metrics_unconstrained['coverage_90']:>12.1%}"
    )
    print(
        f"{'Status':<40} {comp_diag_constrained['status']:>12} {comp_diag_unconstrained['status']:>12}"
    )
    print("=" * 70)

    # =========================================================================
    # Key Findings
    # =========================================================================
    print("\n[5/5] Key Findings")
    print("-" * 70)

    # Compare standard vs label-invariant diagnostics
    std_diff = abs(
        std_diag_constrained["max_rhat"] - comp_diag_constrained["label_invariant"]["rhat_log_lik"]
    )
    if std_diff > 0.2:
        print("[!] SIGNIFICANT DIFFERENCE between standard and label-invariant R-hat")
        print("    Standard R-hat may be inflated by label switching artifact")
        print(
            f"    Standard: {std_diag_constrained['max_rhat']:.3f} vs Log-lik: {comp_diag_constrained['label_invariant']['rhat_log_lik']:.3f}"
        )
    else:
        print("[OK] Standard and label-invariant R-hat are consistent")

    # Check if unconstrained shows significant label switching
    if switching_info["switching_rate"] > 0.1:
        print("[!] SIGNIFICANT LABEL SWITCHING detected in unconstrained model")
        print(f"    Switching rate: {switching_info['switching_rate']:.1%}")
        print("    This confirms label switching is occurring during MCMC")
    else:
        print(f"[OK] Low label switching rate: {switching_info['switching_rate']:.1%}")

    # Check predictive equivalence
    rmse_diff = abs(metrics_constrained["rmse"] - metrics_unconstrained["rmse"])
    if rmse_diff < 0.5:
        print("[OK] Predictive performance is equivalent between models")
        print(f"    RMSE difference: {rmse_diff:.2f}")
    else:
        print(f"[!] Predictive performance differs: RMSE diff = {rmse_diff:.2f}")

    # Overall recommendation
    print("\nRECOMMENDATION:")
    print(f"  Constrained model: {comp_diag_constrained['recommendation']}")
    print(f"  Unconstrained model: {comp_diag_unconstrained['recommendation']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test label switching diagnostics")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    if args.quick:
        config = {
            "num_warmup": 200,
            "num_samples": 400,
            "num_chains": 2,
        }
        print("Running in QUICK mode (reduced samples)")
    else:
        config = {
            "num_warmup": 500,
            "num_samples": 1000,
            "num_chains": 4,
        }

    results = run_diagnostics_comparison(**config, seed=args.seed)

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    # Return success if at least one model shows reasonable convergence
    constrained_ok = results["constrained"]["comprehensive_diagnostics"]["status"] in [
        "converged",
        "partial",
    ]
    unconstrained_ok = results["unconstrained"]["comprehensive_diagnostics"]["status"] in [
        "converged",
        "partial",
    ]

    if constrained_ok or unconstrained_ok:
        print("\n[SUCCESS] Diagnostics validation completed successfully.")
        return 0
    else:
        print("\n[WARNING] Both models show non-convergence. May need longer chains.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
