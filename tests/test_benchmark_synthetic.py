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


DGP_NAMES = ["single", "mixture_k2", "mixture_k3", "mixture_k5"]
MODEL_NAMES = ["single_hill", "mixture_k2", "mixture_k3", "mixture_k5"]

# Match scripts/run_benchmark.py: quick=[0], default=[0,1,2,3,4]
SMOKE_SYNTHETIC_SEEDS = [0]
FULL_SYNTHETIC_SEEDS = [0, 1, 2, 3, 4]
FULL_SYNTHETIC_EXPECTED_CASES = len(DGP_NAMES) * len(MODEL_NAMES) * len(FULL_SYNTHETIC_SEEDS)
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
        warmup = 500 if dgp_name in {"mixture_k3", "mixture_k5"} else 400
        samples = warmup
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=warmup,
            num_samples=samples,
            num_chains=num_chains,
            progress_bar=False,
        )

    if model_name == "mixture_k5":
        warmup = 500
        samples = 500
    else:
        warmup = 450
        samples = 450

    return BenchmarkRunConfig(
        seed=seed,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=num_chains,
        progress_bar=False,
    )


def _effective_k_bounds(dgp_name: str, model_name: str) -> tuple[float, float] | None:
    """Return expected effective-K bounds for mixture models."""
    if model_name == "single_hill":
        return None

    bounds = {
        ("single", "mixture_k2"): (1.2, 1.9),
        ("single", "mixture_k3"): (1.7, 2.4),
        ("single", "mixture_k5"): (1.6, 2.3),
        ("mixture_k2", "mixture_k2"): (1.6, 2.1),
        ("mixture_k2", "mixture_k3"): (2.2, 2.9),
        ("mixture_k2", "mixture_k5"): (2.1, 3.4),
        ("mixture_k3", "mixture_k2"): (1.9, 2.1),
        ("mixture_k3", "mixture_k3"): (2.6, 3.1),
        ("mixture_k3", "mixture_k5"): (3.0, 4.2),
        ("mixture_k5", "mixture_k2"): (1.9, 2.1),
        ("mixture_k5", "mixture_k3"): (2.6, 3.1),
        ("mixture_k5", "mixture_k5"): (3.0, 4.6),
    }
    return bounds[(dgp_name, model_name)]


def _synthetic_thresholds(dgp_name: str, model_name: str) -> BenchmarkThresholds:
    """Return pass/fail thresholds for one synthetic benchmark cell."""
    rmse_limits = {
        "single": 4.0,
        "mixture_k2": 4.0,
        "mixture_k3": 5.6,
        "mixture_k5": 5.6,
    }
    coverage_floors = {
        "single": 0.78,
        "mixture_k2": 0.84,
        "mixture_k3": 0.88,
        "mixture_k5": 0.88,
    }

    if model_name == "single_hill":
        return BenchmarkThresholds(
            max_rhat=1.01,
            max_label_invariant_rhat=None,
            max_relabeled_rhat=None,
            min_test_coverage_90=coverage_floors[dgp_name],
            max_test_rmse=rmse_limits[dgp_name],
            max_test_mu_rmse=1.0 if dgp_name == "single" else None,
            require_alpha_in_ci=(dgp_name == "single"),
            require_sigma_in_ci=(dgp_name == "single"),
            max_pareto_k_bad=0,
            max_pareto_k_very_bad=0,
        )

    max_test_mu_rmse = None
    if dgp_name == "single":
        max_test_mu_rmse = 2.0
    elif model_name == "mixture_k3" and dgp_name == "mixture_k3":
        max_test_mu_rmse = 6.0

    return BenchmarkThresholds(
        max_rhat=None,
        min_ess_bulk=None,
        min_ess_tail=None,
        min_ess_bulk_per_chain=None,
        min_ess_tail_per_chain=None,
        max_label_invariant_rhat=1.01,
        min_label_invariant_ess_bulk_per_chain=100.0,
        min_label_invariant_ess_tail_per_chain=100.0,
        max_relabeled_rhat=1.01,
        min_relabeled_ess_bulk_per_chain=100.0,
        min_relabeled_ess_tail_per_chain=100.0,
        min_test_coverage_90=coverage_floors[dgp_name],
        max_test_rmse=rmse_limits[dgp_name],
        max_test_mu_rmse=max_test_mu_rmse,
        effective_k_bounds=_effective_k_bounds(dgp_name, model_name),
        max_pareto_k_bad=0,
        max_pareto_k_very_bad=0,
    )


def _synthetic_smoke_thresholds(dgp_name: str, model_name: str) -> BenchmarkThresholds:
    """Return lighter smoke-test thresholds for short synthetic runs."""
    base = _synthetic_thresholds(dgp_name, model_name)
    if model_name == "single_hill":
        return BenchmarkThresholds(
            require_effective_convergence=False,
            max_rhat=1.05,
            min_ess_bulk=80.0,
            min_ess_tail=None,
            min_ess_bulk_per_chain=None,
            min_ess_tail_per_chain=None,
            max_label_invariant_rhat=None,
            max_relabeled_rhat=None,
            max_divergences=base.max_divergences,
            min_bfmi=base.min_bfmi,
            max_tree_depth_hits=None,
            min_test_coverage_90=base.min_test_coverage_90,
            max_test_rmse=base.max_test_rmse,
            max_test_mu_rmse=base.max_test_mu_rmse,
            min_test_mu_coverage_90=base.min_test_mu_coverage_90,
            require_alpha_in_ci=base.require_alpha_in_ci,
            require_sigma_in_ci=base.require_sigma_in_ci,
            effective_k_bounds=base.effective_k_bounds,
            max_pareto_k_bad=base.max_pareto_k_bad,
            max_pareto_k_very_bad=base.max_pareto_k_very_bad,
        )

    return BenchmarkThresholds(
        require_effective_convergence=False,
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
        min_test_coverage_90=base.min_test_coverage_90,
        max_test_rmse=base.max_test_rmse,
        max_test_mu_rmse=base.max_test_mu_rmse,
        min_test_mu_coverage_90=base.min_test_mu_coverage_90,
        require_alpha_in_ci=base.require_alpha_in_ci,
        require_sigma_in_ci=base.require_sigma_in_ci,
        effective_k_bounds=base.effective_k_bounds,
        max_pareto_k_bad=None,
        max_pareto_k_very_bad=None,
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
@pytest.mark.parametrize("dgp_name", DGP_NAMES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
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
@pytest.mark.parametrize("dgp_name", DGP_NAMES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
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
@pytest.mark.parametrize("dgp_name", DGP_NAMES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
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
