"""Tests for benchmark metrics."""

import numpy as np

from hill_mixture_mmm.metrics import (
    compute_across_seed_component_stability,
    compute_latent_recovery,
    compute_permutation_invariant_component_recovery,
    summarize_component_posterior,
)


class TestLatentRecovery:
    """Tests for latent mean recovery metrics."""

    def test_perfect_recovery_has_zero_error_and_full_coverage(self):
        """Exact latent mean samples should yield zero error and full coverage."""
        mu_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mu_samples = np.tile(mu_true, (4, 1))

        metrics = compute_latent_recovery(mu_true, mu_samples)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["coverage_90"] == 1.0

    def test_invalid_shapes_raise_value_error(self):
        """Mismatched latent shapes should raise a clear error."""
        mu_true = np.array([1.0, 2.0], dtype=np.float32)
        mu_samples = np.array([1.0, 2.0], dtype=np.float32)

        try:
            compute_latent_recovery(mu_true, mu_samples)
        except ValueError as exc:
            assert "shape (n_samples, T)" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid mu_samples shape")


class TestComponentMetrics:
    """Tests for component recovery and across-seed stability metrics."""

    def test_permutation_invariant_component_recovery_handles_swapped_components(self):
        """Matching against the true DGP should be invariant to component ordering."""
        meta = {
            "A_true": np.array([20.0, 40.0], dtype=np.float32),
            "k_true": np.array([5.0, 12.0], dtype=np.float32),
            "n_true": np.array([2.0, 1.2], dtype=np.float32),
            "pi_true": np.array([0.6, 0.4], dtype=np.float32),
            "s_median": 10.0,
        }
        samples = {
            "A": np.array([[40.0, 20.0], [40.5, 19.5]], dtype=np.float32),
            "k": np.array([[12.0, 5.0], [12.1, 4.9]], dtype=np.float32),
            "n": np.array([[1.2, 2.0], [1.18, 2.02]], dtype=np.float32),
            "pis": np.array([[0.4, 0.6], [0.41, 0.59]], dtype=np.float32),
        }

        recovery = compute_permutation_invariant_component_recovery(samples, meta)

        assert recovery is not None
        assert recovery["K_true_active"] == 2
        assert recovery["K_posterior_active"] == 2
        assert recovery["weighted_curve_nrmse"] < 0.05
        assert recovery["weighted_pi_abs_error"] < 0.05
        assert recovery["unmatched_reference_weight"] == 0.0
        assert recovery["unmatched_candidate_weight"] == 0.0

    def test_across_seed_stability_is_low_for_equivalent_component_sets(self):
        """Across-seed stability should treat relabeled-equivalent summaries as stable."""
        summary_seed0 = {
            "label": "case_seed0",
            "seed": 0,
            "component_summary": summarize_component_posterior(
                {
                    "A": np.array([[20.0, 40.0], [20.5, 39.5]], dtype=np.float32),
                    "k": np.array([[5.0, 12.0], [5.1, 11.9]], dtype=np.float32),
                    "n": np.array([[2.0, 1.2], [2.05, 1.18]], dtype=np.float32),
                    "pis": np.array([[0.6, 0.4], [0.59, 0.41]], dtype=np.float32),
                },
                scale_reference=10.0,
            ),
        }
        summary_seed1 = {
            "label": "case_seed1",
            "seed": 1,
            "component_summary": summarize_component_posterior(
                {
                    "A": np.array([[40.0, 20.0], [39.8, 20.1]], dtype=np.float32),
                    "k": np.array([[12.0, 5.0], [12.2, 5.1]], dtype=np.float32),
                    "n": np.array([[1.2, 2.0], [1.19, 1.98]], dtype=np.float32),
                    "pis": np.array([[0.4, 0.6], [0.39, 0.61]], dtype=np.float32),
                },
                scale_reference=10.0,
            ),
        }

        stability = compute_across_seed_component_stability([summary_seed0, summary_seed1])

        assert stability["num_seeds"] == 2
        assert stability["pair_count"] == 1
        assert stability["active_k_consistency"] == 1.0
        assert stability["weighted_curve_nrmse"]["mean"] < 0.05
        assert stability["weighted_pi_abs_error"]["mean"] < 0.05
