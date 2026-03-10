"""Unit tests for publication-style benchmark convergence checks."""

from __future__ import annotations

import numpy as np
import pytest

from hill_mixture_mmm.benchmark import (
    BenchmarkCaseResult,
    BenchmarkThresholds,
    assert_case_passes,
)
from hill_mixture_mmm.inference import compute_hmc_diagnostics


class _DummyMCMC:
    """Minimal MCMC stub for sampler-diagnostic tests."""

    def __init__(self, extra_fields: dict[str, np.ndarray]):
        self._extra_fields = extra_fields

    def get_extra_fields(self, group_by_chain: bool = False):
        assert group_by_chain
        return self._extra_fields


def _make_mixture_result(*, converged: bool = True, relabeled_max_rhat: float = 1.0) -> BenchmarkCaseResult:
    """Return a minimal benchmark result suitable for assert_case_passes tests."""
    return BenchmarkCaseResult(
        label="synthetic_case",
        domain="synthetic",
        dataset_name="mixture_k3",
        model_name="mixture_k3",
        seed=0,
        train_ratio=0.75,
        x_train=np.array([1.0], dtype=np.float32),
        x_test=np.array([1.0], dtype=np.float32),
        y_train=np.array([1.0], dtype=np.float32),
        y_test=np.array([1.0], dtype=np.float32),
        dates_train=None,
        dates_test=None,
        train_metrics={"rmse": 1.0, "coverage_90": 0.9, "y_pred_mean": np.array([1.0])},
        test_metrics={"rmse": 1.0, "coverage_90": 0.9, "y_pred_mean": np.array([1.0])},
        loo={"elpd_loo": -1.0, "se": 0.1, "pareto_k_bad": 0, "pareto_k_very_bad": 0},
        waic={"elpd_waic": -1.0, "se": 0.1},
        convergence={
            "max_rhat": 1.0,
            "min_ess_bulk": 400.0,
            "min_ess_tail": 400.0,
            "converged": True,
            "ess_sufficient": True,
        },
        hmc_diagnostics={
            "num_divergences": 0,
            "has_divergence": False,
            "bfmi_by_chain": [0.8, 0.9],
            "min_bfmi": 0.8,
            "bfmi_ok": True,
            "max_tree_depth": 10,
            "tree_depth_hits": 0,
            "max_tree_depth_hit": False,
            "max_num_steps": 16,
            "mean_accept_prob": 0.95,
        },
        label_invariant={
            "rhat_log_lik": 1.0,
            "rhat_scalars": {"sigma": 1.0},
            "ess_bulk_log_lik": 400.0,
            "ess_tail_log_lik": 400.0,
            "ess_bulk_scalars": {"sigma": 400.0},
            "ess_tail_scalars": {"sigma": 400.0},
            "min_ess_bulk": 400.0,
            "min_ess_tail": 400.0,
            "max_rhat": 1.0,
            "converged": True,
            "method": "rank",
            "threshold": 1.01,
        },
        relabeled={
            "component_rhats": {"A": {"per_component": [relabeled_max_rhat], "max": relabeled_max_rhat}},
            "component_ess_bulk": {"A": {"per_component": [400.0], "min": 400.0}},
            "component_ess_tail": {"A": {"per_component": [400.0], "min": 400.0}},
            "max_rhat": relabeled_max_rhat,
            "min_ess_bulk": 400.0,
            "min_ess_tail": 400.0,
            "converged": relabeled_max_rhat < 1.01,
            "method": "rank",
            "threshold": 1.01,
        },
        label_switching={
            "switching_rate": 0.0,
            "n_unique_orderings": 1,
            "mode_ordering": [0, 1, 2],
            "mode_count": 10,
            "top_orderings": [([0, 1, 2], 10)],
        },
        component_summary=None,
        component_recovery=None,
        converged=converged,
        effective_k={"effective_k_mean": 2.5, "effective_k_std": 0.1},
        parameter_recovery=None,
        latent_train=None,
        latent_test=None,
        meta=None,
        samples={},
        fit_summary={
            "inference_seed": 0,
            "num_warmup_used": 200,
            "num_samples_used": 200,
            "num_chains_used": 2,
            "target_accept_prob_used": 0.9,
            "max_tree_depth_used": 10,
        },
    )


def test_compute_hmc_diagnostics_reports_bfmi_and_tree_depth_hits():
    """Sampler diagnostics should summarize divergences, BFMI, and tree depth saturation."""
    mcmc = _DummyMCMC(
        {
            "diverging": np.array(
                [
                    [False, False, False, False, False, True, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False, False, False],
                ]
            ),
            "energy": np.array(
                [
                    np.linspace(0.0, 10.0, 11),
                    np.linspace(0.0, 5.0, 11),
                ]
            ),
            "num_steps": np.array(
                [
                    [16, 16, 16, 16, 16, 512, 16, 16, 16, 16, 16],
                    [4, 8, 16, 8, 4, 8, 16, 8, 4, 8, 16],
                ]
            ),
            "accept_prob": np.array(
                [
                    [0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9],
                    [0.95, 0.95, 0.9, 0.9, 0.95, 0.95, 0.9, 0.9, 0.95, 0.95, 0.9],
                ]
            ),
        }
    )

    diagnostics = compute_hmc_diagnostics(mcmc, max_tree_depth=10)

    assert diagnostics["num_divergences"] == 1
    assert diagnostics["tree_depth_hits"] == 1
    assert diagnostics["max_num_steps"] == 512
    assert diagnostics["min_bfmi"] < 0.3


def test_assert_case_passes_requires_relabeled_component_convergence_for_mixtures():
    """Mixture benchmark thresholds should fail when relabeled component R-hat is too high."""
    result = _make_mixture_result(converged=False, relabeled_max_rhat=1.02)

    with pytest.raises(AssertionError, match="relabeled_max_rhat=1.020 exceeds 1.010"):
        assert_case_passes(
            result,
            BenchmarkThresholds(
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
            ),
        )
