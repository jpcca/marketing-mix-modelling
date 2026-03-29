"""Opt-in real-data benchmark smoke/full tests with pass/fail quality gates."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    BenchmarkThresholds,
    assert_case_passes,
    plot_case_comparison,
    resolve_comparison_artifact_dir,
    run_real_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data_loader import TimeSeriesConfig


DEFAULT_REAL_ORG_ID = "72a86a208d24d68b80be0e44a8a4872d"

SMOKE_REAL_SEEDS = [0]
FULL_REAL_SEEDS = [0, 1, 2]
SMOKE_REAL_MODELS = ["single_hill", "mixture_k2"]
FULL_REAL_MODELS = ["single_hill", "mixture_k2", "mixture_k3"]


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
    )


def _real_thresholds() -> BenchmarkThresholds:
    """Return reportability gates for real-data benchmark runs."""
    return BenchmarkThresholds(
        max_rhat=None,
        min_ess_bulk=None,
        min_ess_tail=None,
        min_ess_bulk_per_chain=None,
        min_ess_tail_per_chain=None,
        max_label_invariant_rhat=None,
        min_label_invariant_ess_bulk_per_chain=None,
        min_label_invariant_ess_tail_per_chain=None,
        max_relabeled_rhat=None,
        min_relabeled_ess_bulk_per_chain=None,
        min_relabeled_ess_tail_per_chain=None,
        max_divergences=None,
        min_bfmi=None,
        max_tree_depth_hits=None,
        min_test_coverage_90=None,
        max_test_mape=None,
        max_test_mu_mape=None,
        min_test_mu_coverage_90=None,
        require_alpha_in_ci=False,
        require_sigma_in_ci=False,
        effective_k_bounds=None,
        max_pareto_k_bad=None,
        max_pareto_k_very_bad=None,
        require_reportable_diagnostics=True,
    )


def _run_real_seed(seed: int, benchmark_output_root: Path, model_names: list[str]) -> None:
    """Run the selected real-data benchmark models for one seed."""
    csv_path = Path("data/conjura_mmm_data.csv")
    org_id = os.getenv("HILL_MMM_REAL_BENCHMARK_ORG", DEFAULT_REAL_ORG_ID)
    timeseries_config = TimeSeriesConfig(
        organisation_id=org_id,
        aggregate_spend=True,
        min_series_length=200,
    )

    results = []
    artifact_paths = []
    for model_name in model_names:
        result = run_real_benchmark_case(
            csv_path=csv_path,
            timeseries_config=timeseries_config,
            model_name=model_name,
            config=_real_run_config(seed),
            label=f"real_{model_name}_seed{seed}",
        )
        artifact_paths.extend(save_case_artifacts(result, benchmark_output_root).values())
        assert_case_passes(result, _real_thresholds())
        results.append(result)

    comparison_dir = resolve_comparison_artifact_dir(benchmark_output_root, "real")
    comparison_path = plot_case_comparison(
        results,
        comparison_dir / f"real_model_comparison_seed{seed}.png",
        title=f"Real Benchmark Comparison ({org_id}, seed={seed})",
    )

    for path in [*artifact_paths, comparison_path]:
        assert path.exists(), f"Expected real benchmark artifact at {path}"


@pytest.mark.slow
@pytest.mark.benchmark_smoke
@pytest.mark.parametrize("seed", SMOKE_REAL_SEEDS)
def test_real_data_benchmark_smoke(seed: int, benchmark_output_root: Path) -> None:
    """Quick real-data benchmark smoke test using the original quick seed."""
    _require_real_benchmark()
    _run_real_seed(seed, benchmark_output_root, SMOKE_REAL_MODELS)


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("seed", FULL_REAL_SEEDS)
def test_real_data_benchmark_full(seed: int, benchmark_output_root: Path) -> None:
    """Full real-data benchmark using the original multi-seed schedule."""
    _require_full_real_benchmark()
    _run_real_seed(seed, benchmark_output_root, FULL_REAL_MODELS)
