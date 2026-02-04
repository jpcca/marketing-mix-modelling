#!/usr/bin/env python
"""Compare single Hill vs mixture models on real Conjura data.

Runs all 3 model types and compares their LOO-CV scores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from hill_mmm.benchmark import MODEL_SPECS
from hill_mmm.data import compute_prior_config
from hill_mmm.data_loader import (
    TimeSeriesConfig,
    load_timeseries,
    select_representative_timeseries,
)
from hill_mmm.inference import (
    compute_convergence_diagnostics,
    compute_loo,
    compute_predictions,
    compute_predictive_metrics,
    run_inference,
)


def main():
    csv_path = "data/conjura_mmm_data.csv"

    # MCMC settings (reduced for speed, but enough for comparison)
    NUM_WARMUP = 200
    NUM_SAMPLES = 400
    NUM_CHAINS = 2
    TRAIN_RATIO = 0.75

    print("=" * 70)
    print("Model Comparison: Single Hill vs Mixture Models")
    print("=" * 70)

    # Select 1 representative organization
    print("\n[1] Selecting organisation...")
    org_ids = select_representative_timeseries(
        csv_path, n=1, min_length=300, min_channels=2, seed=42
    )
    org_id = org_ids[0]
    print(f"    Selected: {org_id[:20]}...")

    # Load data
    print("\n[2] Loading data...")
    config = TimeSeriesConfig(
        organisation_id=org_id,
        target_col="all_purchases",
        aggregate_spend=True,
    )
    data = load_timeseries(csv_path, config)

    # Split train/test manually
    T = len(data.y)
    T_train = int(T * TRAIN_RATIO)
    x_train, y_train = data.x[:T_train], data.y[:T_train]
    x_test, y_test = data.x[T_train:], data.y[T_train:]

    print(f"    T={T}, Train={T_train}, Test={T - T_train}")

    # Compute priors
    prior_config = compute_prior_config(x_train, y_train)

    results = []

    # Run all 3 models
    for spec in MODEL_SPECS:
        print(f"\n[3] Running {spec.name}...")

        try:
            # Run inference (note: prior_config is a keyword argument)
            mcmc = run_inference(
                spec.fn,
                x_train,
                y_train,
                seed=42,
                num_warmup=NUM_WARMUP,
                num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS,
                prior_config=prior_config,
                **spec.kwargs,
            )

            # Check convergence
            diag = compute_convergence_diagnostics(mcmc)
            print(f"    R-hat: {diag['max_rhat']:.3f}, ESS: {diag['min_ess_bulk']:.0f}")

            # Compute LOO
            loo_result = compute_loo(mcmc)

            # Compute predictions (train and test separately)
            preds_train = compute_predictions(mcmc, spec.fn, x_train, prior_config, **spec.kwargs)
            preds_test = compute_predictions(mcmc, spec.fn, x_test, prior_config, **spec.kwargs)

            # Compute metrics (y_true first, y_samples second)
            metrics_train = compute_predictive_metrics(y_train, preds_train["y"])
            metrics_test = compute_predictive_metrics(y_test, preds_test["y"])

            results.append(
                {
                    "model": spec.name,
                    "elpd_loo": loo_result["elpd_loo"],
                    "p_loo": loo_result["p_loo"],
                    "train_rmse": metrics_train["rmse"],
                    "test_rmse": metrics_test["rmse"],
                    "train_coverage": metrics_train["coverage_90"],
                    "test_coverage": metrics_test["coverage_90"],
                    "converged": diag["converged"],
                    "max_rhat": diag["max_rhat"],
                }
            )

            print(f"    ELPD-LOO: {loo_result['elpd_loo']:.1f}")
            print(f"    Test RMSE: {metrics_test['rmse']:.2f}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    ERROR: {e}")
            results.append(
                {
                    "model": spec.name,
                    "elpd_loo": np.nan,
                    "p_loo": np.nan,
                    "train_rmse": np.nan,
                    "test_rmse": np.nan,
                    "train_coverage": np.nan,
                    "test_coverage": np.nan,
                    "converged": False,
                    "max_rhat": np.nan,
                }
            )

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # Delta LOO comparison
    if len(df) > 1 and not bool(df["elpd_loo"].isna().all()):
        best_idx = df["elpd_loo"].idxmax()
        best_model = df.loc[best_idx, "model"]
        best_elpd = df.loc[best_idx, "elpd_loo"]

        print(f"\nBest model by ELPD-LOO: {best_model}")
        print("\nDelta ELPD from best:")
        for _, row in df.iterrows():
            delta = row["elpd_loo"] - best_elpd
            print(f"  {row['model']}: {delta:+.1f}")

    # Save results
    df.to_csv("results/model_comparison.csv", index=False)
    print("\nResults saved to results/model_comparison.csv")


if __name__ == "__main__":
    import os

    os.makedirs("results", exist_ok=True)
    main()
