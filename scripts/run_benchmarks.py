#!/usr/bin/env python
"""Run the Hill Mixture MMM benchmark suite.

Usage:
    python scripts/run_benchmarks.py                    # Full suite (slow)
    python scripts/run_benchmarks.py --quick            # Quick test (2 seeds, fewer samples)
    python scripts/run_benchmarks.py --dgp single mixture_k3  # Specific DGPs
    python scripts/run_benchmarks.py --output results.csv     # Save results
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from hill_mmm.benchmark import (
    print_benchmark_table,
    run_benchmark_suite,
    summarize_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description="Run Hill Mixture MMM benchmark suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (2 seeds, 500 samples)",
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
        help="Random seeds (default: 0-4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )

    args = parser.parse_args()

    # Configure based on mode
    if args.quick:
        seeds = args.seeds or [0, 1]
        num_warmup = 500
        num_samples = 500
        print("Running in QUICK mode (2 seeds, 500 samples)")
    else:
        seeds = args.seeds or [0, 1, 2, 3, 4]
        num_warmup = 1000
        num_samples = 2000
        print(f"Running FULL benchmark ({len(seeds)} seeds, {num_samples} samples)")

    print(f"DGPs: {args.dgp or 'all'}")
    print(f"Models: {args.model or 'all'}")
    print(f"Seeds: {seeds}")
    print()

    # Run benchmark
    results = run_benchmark_suite(
        dgp_names=args.dgp,
        model_names=args.model,
        seeds=seeds,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=args.chains,
        verbose=True,
    )

    # Print results
    print_benchmark_table(results)

    # Save if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

        # Also save summary
        summary_path = args.output.replace(".csv", "_summary.csv")
        summary = summarize_benchmark(results)
        summary.to_csv(summary_path)
        print(f"Summary saved to {summary_path}")

    return results


if __name__ == "__main__":
    main()
