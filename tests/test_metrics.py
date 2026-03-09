"""Tests for benchmark metrics."""

import numpy as np

from hill_mixture_mmm.metrics import compute_latent_recovery


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
