"""Synthetic benchmark smoke/full test matrix for the Hill MMM models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkRunConfig,
    BenchmarkThresholds,
    assert_case_passes,
    run_synthetic_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.data import DGPConfig


DGP_NAMES = ["single", "mixture_k2", "mixture_k3", "mixture_k5"]
MODEL_NAMES = ["single_hill", "mixture_k2", "mixture_k3", "mixture_k5"]

# Match scripts/run_benchmark.py: quick=[0], default=[0,1,2,3,4]
SMOKE_SYNTHETIC_SEEDS = [0]
FULL_SYNTHETIC_SEEDS = [0, 1, 2, 3, 4]


def _require_full_synthetic_benchmark() -> None:
    """Skip unless the caller explicitly opts into the multi-seed synthetic benchmark."""
    enabled = os.getenv("HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full synthetic benchmark is opt-in; set "
            "HILL_MMM_RUN_FULL_SYNTHETIC_BENCHMARK=1 to run it"
        )


def _synthetic_run_config(dgp_name: str, model_name: str, seed: int) -> BenchmarkRunConfig:
    """Return an inference configuration suitable for one synthetic benchmark cell."""
    if model_name == "single_hill":
        warmup = 500 if dgp_name in {"mixture_k3", "mixture_k5"} else 400
        samples = warmup
        return BenchmarkRunConfig(
            seed=seed,
            num_warmup=warmup,
            num_samples=samples,
            num_chains=2,
            progress_bar=False,
            allow_mixture_retries=False,
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
        num_chains=2,
        progress_bar=False,
        allow_mixture_retries=True,
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
            max_rhat=1.05,
            min_ess_bulk=80.0,
            max_label_invariant_rhat=None,
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
        max_label_invariant_rhat=1.01,
        min_test_coverage_90=coverage_floors[dgp_name],
        max_test_rmse=rmse_limits[dgp_name],
        max_test_mu_rmse=max_test_mu_rmse,
        effective_k_bounds=_effective_k_bounds(dgp_name, model_name),
        max_pareto_k_bad=0,
        max_pareto_k_very_bad=0,
    )


def _run_and_assert_case(
    dgp_name: str,
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    """Run one synthetic benchmark cell and assert its quality gates."""
    result = run_synthetic_benchmark_case(
        dgp_config=DGPConfig(dgp_type=dgp_name, T=200, seed=seed),
        model_name=model_name,
        config=_synthetic_run_config(dgp_name, model_name, seed),
        label=f"synthetic_{dgp_name}_{model_name}_seed{seed}",
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    assert_case_passes(result, _synthetic_thresholds(dgp_name, model_name))

    for path in artifacts.values():
        assert path.exists(), f"Expected synthetic artifact at {path}"


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
    _run_and_assert_case(dgp_name, model_name, seed, benchmark_output_root)


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
    _run_and_assert_case(dgp_name, model_name, seed, benchmark_output_root)
