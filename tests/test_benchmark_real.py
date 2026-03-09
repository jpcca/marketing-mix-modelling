"""Opt-in real-data benchmark smoke/full tests with pass/fail quality gates."""

from __future__ import annotations

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
    resolve_comparison_artifact_dir,
    run_real_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data_loader import TimeSeriesConfig


DEFAULT_REAL_ORG_ID = "72a86a208d24d68b80be0e44a8a4872d"

# Match scripts/run_benchmark.py: quick=[0], default=[0,1,2]
SMOKE_REAL_SEEDS = [0]
FULL_REAL_SEEDS = [0, 1, 2]


def _require_real_benchmark() -> None:
    """Skip unless the caller explicitly opted into the real-data smoke benchmark."""
    enabled = os.getenv("HILL_MMM_RUN_REAL_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip("real benchmark is opt-in; set HILL_MMM_RUN_REAL_BENCHMARK=1 to run it")


def _require_full_real_benchmark() -> None:
    """Skip unless the caller explicitly opted into the full multi-seed real benchmark."""
    enabled = os.getenv("HILL_MMM_RUN_FULL_REAL_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full real benchmark is opt-in; set HILL_MMM_RUN_FULL_REAL_BENCHMARK=1 to run it"
        )


def _real_run_config(seed: int) -> BenchmarkRunConfig:
    """Return real-data inference settings for one benchmark seed."""
    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=int(os.getenv("HILL_MMM_REAL_WARMUP", "2000")),
        num_samples=int(os.getenv("HILL_MMM_REAL_SAMPLES", "2000")),
        num_chains=int(os.getenv("HILL_MMM_REAL_CHAINS", "4")),
        progress_bar=False,
        allow_mixture_retries=False,
    )


def _run_real_seed(seed: int, benchmark_output_root: Path) -> None:
    """Run the paired single/mixture real-data benchmark for one seed."""
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
        config=_real_run_config(seed),
        label=f"real_single_hill_seed{seed}",
    )
    single_artifacts = save_case_artifacts(single_result, benchmark_output_root)
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
        config=_real_run_config(seed),
        label=f"real_mixture_k3_seed{seed}",
    )
    mixture_artifacts = save_case_artifacts(mixture_result, benchmark_output_root)
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

    comparison_dir = resolve_comparison_artifact_dir(benchmark_output_root, "real")
    comparison_path = plot_case_comparison(
        [single_result, mixture_result],
        comparison_dir / f"real_model_comparison_seed{seed}.png",
        title=f"Real Benchmark Comparison ({org_id}, seed={seed})",
    )

    for path in [*single_artifacts.values(), *mixture_artifacts.values(), comparison_path]:
        assert path.exists(), f"Expected real benchmark artifact at {path}"


@pytest.mark.slow
@pytest.mark.benchmark_smoke
@pytest.mark.parametrize("seed", SMOKE_REAL_SEEDS)
def test_real_data_benchmark_smoke(seed: int, benchmark_output_root: Path) -> None:
    """Quick real-data benchmark smoke test using the original quick seed."""
    _require_real_benchmark()
    _run_real_seed(seed, benchmark_output_root)


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("seed", FULL_REAL_SEEDS)
def test_real_data_benchmark_full(seed: int, benchmark_output_root: Path) -> None:
    """Full real-data benchmark using the original multi-seed schedule."""
    _require_full_real_benchmark()
    _run_real_seed(seed, benchmark_output_root)
