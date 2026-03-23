"""Synthetic benchmark smoke/full test matrix for the Hill MMM models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import time

import pytest

from hill_mixture_mmm.paper_figures import (
    DEFAULT_FIGURE_IDS,
    generate_publication_figures,
    load_synthetic_results_from_artifacts,
)
from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    BenchmarkThresholds,
    assert_case_passes,
    run_synthetic_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data import DGPConfig
from hill_mixture_mmm.metrics import compute_across_seed_component_stability


SMOKE_DGP_NAMES = ["single", "mixture_k2"]
SMOKE_MODEL_NAMES = ["single_hill", "mixture_k2"]
FULL_DGP_NAMES = ["single", "mixture_k2", "mixture_k3"]
FULL_MODEL_NAMES = ["single_hill", "mixture_k2", "mixture_k3"]

# Match scripts/run_benchmark.py: quick=[0], default=[0,1,2,3,4]
SMOKE_SYNTHETIC_SEEDS = [0]
FULL_SYNTHETIC_SEEDS = [0, 1, 2, 3, 4]
FULL_SYNTHETIC_EXPECTED_CASES = (
    len(FULL_DGP_NAMES) * len(FULL_MODEL_NAMES) * len(FULL_SYNTHETIC_SEEDS)
)
PAPER_FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"


def _require_full_synthetic_benchmark() -> None:
    """Skip unless the caller explicitly opts into the multi-seed synthetic benchmark."""
    enabled = os.getenv("HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full synthetic benchmark is opt-in; set "
            "HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK=1 to run it"
        )


@pytest.fixture(scope="module", autouse=True)
def _generate_selected_publication_figures_after_full_benchmark() -> None:
    """Generate paper figures once the full synthetic benchmark has finished."""
    started_at = time()
    yield

    enabled = os.getenv("HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return

    summary_paths = sorted(
        path
        for path in PAPER_FIGURES_DIR.glob("synthetic/*/*_seed*_summary.json")
        if path.stat().st_mtime >= started_at - 1.0
    )
    if not summary_paths:
        return

    results = load_synthetic_results_from_artifacts(
        PAPER_FIGURES_DIR,
        summary_paths=summary_paths,
    )
    if len(results) != FULL_SYNTHETIC_EXPECTED_CASES:
        return

    generated = generate_publication_figures(
        output_dir=PAPER_FIGURES_DIR,
        artifact_root=PAPER_FIGURES_DIR,
        summary_paths=summary_paths,
        figure_ids=DEFAULT_FIGURE_IDS,
    )
    for path in generated.values():
        assert path.exists(), f"Expected publication figure at {path}"


def _synthetic_run_config(dgp_name: str, model_name: str, seed: int) -> BenchmarkRunConfig:
    """Return an inference configuration suitable for one synthetic benchmark cell."""
    num_chains = int(os.getenv("HILL_MMM_SYNTHETIC_CHAINS", "2"))
    if model_name == "single_hill":
        warmup = 500 if dgp_name == "mixture_k3" else 400
        samples = warmup
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=warmup,
            num_samples=samples,
            num_chains=num_chains,
            progress_bar=False,
        )

    warmup = 450
    samples = 450

    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=num_chains,
        progress_bar=False,
    )


def _synthetic_thresholds(dgp_name: str, model_name: str) -> BenchmarkThresholds:
    """Return paper-level reportability gates for one synthetic benchmark cell."""
    del model_name
    max_test_mape = 5.0 if dgp_name == "single" else None
    max_test_mu_mape = 5.0 if dgp_name == "mixture_k2" else None
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
        max_test_mape=max_test_mape,
        max_test_mu_mape=max_test_mu_mape,
        min_test_mu_coverage_90=None,
        require_alpha_in_ci=False,
        require_sigma_in_ci=False,
        effective_k_bounds=None,
        max_pareto_k_bad=None,
        max_pareto_k_very_bad=None,
        require_reportable_diagnostics=True,
        require_truth_metrics=True,
    )


def _synthetic_smoke_thresholds(dgp_name: str, model_name: str) -> BenchmarkThresholds:
    """Return lighter smoke-test gates for short synthetic runs."""
    del model_name
    max_test_mape = 5.0 if dgp_name == "single" else None
    max_test_mu_mape = 5.0 if dgp_name == "mixture_k2" else None
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
        max_test_mape=max_test_mape,
        max_test_mu_mape=max_test_mu_mape,
        min_test_mu_coverage_90=None,
        require_alpha_in_ci=False,
        require_sigma_in_ci=False,
        effective_k_bounds=None,
        max_pareto_k_bad=None,
        max_pareto_k_very_bad=None,
        require_reportable_diagnostics=False,
        require_truth_metrics=True,
    )


def _run_and_assert_case(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
    thresholds: BenchmarkThresholds,
) -> None:
    """Run one synthetic benchmark cell and assert its quality gates."""
    result = run_synthetic_benchmark_case(
        dgp_config=DGPConfig(dgp_type=dgp_name, T=200, seed=seed),
        model_name=model_name,
        config=_synthetic_run_config(dgp_name, model_name, seed),
        label=f"synthetic_{dgp_name}_{model_name}_seed{seed}",
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    assert_case_passes(result, thresholds)

    for path in artifacts.values():
        assert path.exists(), f"Expected synthetic artifact at {path}"


def _case_summary_path(
    benchmark_output_root: Path,
    dgp_name: str,
    model_name: str,
    seed: int,
) -> Path:
    """Return the saved summary path for one synthetic benchmark seed."""
    return (
        benchmark_output_root
        / "synthetic"
        / model_name
        / f"synthetic_{dgp_name}_{model_name}_seed{seed}_summary.json"
    )


@pytest.mark.slow
@pytest.mark.benchmark_smoke
@pytest.mark.parametrize("seed", SMOKE_SYNTHETIC_SEEDS)
@pytest.mark.parametrize("dgp_name", SMOKE_DGP_NAMES)
@pytest.mark.parametrize("model_name", SMOKE_MODEL_NAMES)
def test_synthetic_benchmark_smoke_matrix(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    """Quick synthetic benchmark smoke test with the original quick seed set."""
    _run_and_assert_case(
        dgp_name,
        model_name,
        seed,
        benchmark_output_root,
        _synthetic_smoke_thresholds(dgp_name, model_name),
    )


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("seed", FULL_SYNTHETIC_SEEDS)
@pytest.mark.parametrize("dgp_name", FULL_DGP_NAMES)
@pytest.mark.parametrize("model_name", FULL_MODEL_NAMES)
def test_synthetic_benchmark_full_matrix(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    """Full synthetic benchmark with the original multi-seed benchmark schedule."""
    _require_full_synthetic_benchmark()
    _run_and_assert_case(
        dgp_name,
        model_name,
        seed,
        benchmark_output_root,
        _synthetic_thresholds(dgp_name, model_name),
    )


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("dgp_name", FULL_DGP_NAMES)
@pytest.mark.parametrize("model_name", FULL_MODEL_NAMES)
def test_synthetic_benchmark_full_across_seed_stability(
    dgp_name: str,
    model_name: str,
    benchmark_output_root: Path,
) -> None:
    """Summarize across-seed component stability after the full synthetic sweep."""
    _require_full_synthetic_benchmark()

    summaries = []
    for seed in FULL_SYNTHETIC_SEEDS:
        summary_path = _case_summary_path(benchmark_output_root, dgp_name, model_name, seed)
        assert summary_path.exists(), f"Expected benchmark summary at {summary_path}"
        with summary_path.open("r", encoding="utf-8") as fh:
            summaries.append(json.load(fh))

    stability = compute_across_seed_component_stability(summaries)
    stability_path = (
        benchmark_output_root
        / "synthetic"
        / model_name
        / f"synthetic_{dgp_name}_{model_name}_across_seed_stability.json"
    )
    with stability_path.open("w", encoding="utf-8") as fh:
        json.dump(stability, fh, indent=2)

    assert stability["num_seeds"] == len(FULL_SYNTHETIC_SEEDS)
    assert stability["pair_count"] == (len(FULL_SYNTHETIC_SEEDS) * (len(FULL_SYNTHETIC_SEEDS) - 1)) // 2
    assert stability_path.exists(), f"Expected across-seed stability summary at {stability_path}"
