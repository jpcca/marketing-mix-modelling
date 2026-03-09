"""Opt-in real-data benchmark test with pass/fail quality gates."""

import os
from pathlib import Path

import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    BenchmarkThresholds,
    ComparisonThresholds,
    assert_case_passes,
    assert_comparison_passes,
    plot_case_comparison,
    run_real_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data_loader import TimeSeriesConfig


DEFAULT_REAL_ORG_ID = "72a86a208d24d68b80be0e44a8a4872d"


def _require_real_benchmark() -> None:
    """Skip unless the caller explicitly opted into the long-running real benchmark."""
    enabled = os.getenv("HILL_MMM_RUN_REAL_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "real benchmark is opt-in; set HILL_MMM_RUN_REAL_BENCHMARK=1 to run this test"
        )


@pytest.mark.slow
def test_real_data_benchmark(benchmark_output_root: Path) -> None:
    """Run an end-to-end real-data benchmark and fail on convergence or quality regressions."""
    _require_real_benchmark()

    csv_path = Path("data/conjura_mmm_data.csv")
    org_id = os.getenv("HILL_MMM_REAL_BENCHMARK_ORG", DEFAULT_REAL_ORG_ID)
    timeseries_config = TimeSeriesConfig(
        organisation_id=org_id,
        aggregate_spend=True,
        min_series_length=200,
    )

    single_result = run_real_benchmark_case(
        csv_path=csv_path,
        timeseries_config=timeseries_config,
        model_name="single_hill",
        config=BenchmarkRunConfig(
            seed=int(os.getenv("HILL_MMM_REAL_SINGLE_SEED", "0")),
            num_warmup=int(os.getenv("HILL_MMM_REAL_WARMUP", "2000")),
            num_samples=int(os.getenv("HILL_MMM_REAL_SAMPLES", "2000")),
            num_chains=int(os.getenv("HILL_MMM_REAL_CHAINS", "4")),
            progress_bar=False,
            allow_mixture_retries=False,
        ),
        label="real_single_hill",
    )
    assert_case_passes(
        single_result,
        BenchmarkThresholds(
            max_rhat=1.05,
            min_ess_bulk=100.0,
            max_label_invariant_rhat=None,
            min_test_coverage_90=0.75,
            max_test_rmse=15.0,
        ),
    )

    mixture_result = run_real_benchmark_case(
        csv_path=csv_path,
        timeseries_config=timeseries_config,
        model_name="mixture_k3",
        config=BenchmarkRunConfig(
            seed=int(os.getenv("HILL_MMM_REAL_MIXTURE_SEED", "2")),
            num_warmup=int(os.getenv("HILL_MMM_REAL_WARMUP", "2000")),
            num_samples=int(os.getenv("HILL_MMM_REAL_SAMPLES", "2000")),
            num_chains=int(os.getenv("HILL_MMM_REAL_CHAINS", "4")),
            progress_bar=False,
            allow_mixture_retries=False,
        ),
        label="real_mixture_k3",
    )
    assert_case_passes(
        mixture_result,
        BenchmarkThresholds(
            max_rhat=1.05,
            min_ess_bulk=100.0,
            max_label_invariant_rhat=1.01,
            min_test_coverage_90=0.75,
            max_test_rmse=15.0,
            effective_k_bounds=(1.5, 3.5),
        ),
    )

    assert_comparison_passes(
        single_result,
        mixture_result,
        ComparisonThresholds(
            min_delta_loo=-50.0,
            max_candidate_rmse_ratio=1.50,
        ),
    )

    single_artifacts = save_case_artifacts(single_result, benchmark_output_root)
    mixture_artifacts = save_case_artifacts(mixture_result, benchmark_output_root)
    comparison_path = plot_case_comparison(
        [single_result, mixture_result],
        benchmark_output_root / "real_model_comparison.png",
        title=f"Real Benchmark Comparison ({org_id})",
    )

    for path in [*single_artifacts.values(), *mixture_artifacts.values(), comparison_path]:
        assert path.exists(), f"Expected real benchmark artifact at {path}"
