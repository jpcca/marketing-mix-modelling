"""Test-driven synthetic benchmark cases for the Hill MMM models."""

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


@pytest.mark.slow
def test_single_hill_synthetic_benchmark(benchmark_output_root: Path) -> None:
    """Single-Hill should recover a single-component DGP and produce artifacts."""
    result = run_synthetic_benchmark_case(
        dgp_config=DGPConfig(dgp_type="single", T=200, seed=42),
        model_name="single_hill",
        config=BenchmarkRunConfig(
            seed=42,
            num_warmup=300,
            num_samples=300,
            num_chains=2,
            progress_bar=False,
            allow_mixture_retries=False,
        ),
        label="synthetic_single_single_hill",
    )

    assert_case_passes(
        result,
        BenchmarkThresholds(
            max_rhat=1.05,
            min_ess_bulk=80.0,
            min_test_coverage_90=0.90,
            max_test_rmse=3.50,
            max_test_mu_rmse=1.00,
            require_alpha_in_ci=True,
            require_sigma_in_ci=True,
            max_pareto_k_bad=0,
            max_pareto_k_very_bad=0,
        ),
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    for path in artifacts.values():
        assert path.exists(), f"Expected synthetic artifact at {path}"


@pytest.mark.slow
def test_mixture_k3_synthetic_benchmark(benchmark_output_root: Path) -> None:
    """Mixture-K3 should converge on a K=3 synthetic DGP and recover effective K."""
    result = run_synthetic_benchmark_case(
        dgp_config=DGPConfig(dgp_type="mixture_k3", T=200, seed=42),
        model_name="mixture_k3",
        config=BenchmarkRunConfig(
            seed=42,
            num_warmup=400,
            num_samples=400,
            num_chains=2,
            progress_bar=False,
            allow_mixture_retries=True,
        ),
        label="synthetic_mixture_k3_model",
    )

    assert_case_passes(
        result,
        BenchmarkThresholds(
            max_rhat=1.05,
            min_ess_bulk=100.0,
            max_label_invariant_rhat=1.01,
            min_test_coverage_90=0.85,
            max_test_rmse=7.00,
            max_test_mu_rmse=6.00,
            effective_k_bounds=(2.0, 3.2),
            max_pareto_k_bad=0,
            max_pareto_k_very_bad=0,
        ),
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    for path in artifacts.values():
        assert path.exists(), f"Expected synthetic artifact at {path}"
