"""Tests for synthetic-data helpers and prior configuration."""

import numpy as np

from hill_mixture_mmm.data import (
    A_PRIOR_RANGE_FRACTION,
    DGPConfig,
    compute_prior_config,
    generate_data,
)


class TestComputePriorConfig:
    """Tests for empirical-Bayes prior construction."""

    def test_intercept_prior_is_shifted_below_response_mean(self):
        """Intercept prior should reserve room for positive Hill effects."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([10.0, 12.0, 18.0, 22.0], dtype=np.float32)

        prior = compute_prior_config(x, y)

        y_mean = float(np.mean(y))
        y_min = float(np.min(y))
        y_range = float(np.max(y) - np.min(y))
        expected = max(y_min, y_mean - y_range * A_PRIOR_RANGE_FRACTION)

        assert prior["intercept_loc"] == expected
        assert prior["intercept_loc"] < y_mean
        assert np.isclose(np.exp(prior["A_loc"]), y_range * A_PRIOR_RANGE_FRACTION, atol=1e-6)


class TestSyntheticLatentTargets:
    """Tests that synthetic metadata exposes the correct latent benchmark target."""

    def test_single_dgp_expected_latent_matches_realized_latent(self):
        """Single-Hill data should have identical realized and expected latent means."""
        _, _, meta = generate_data(DGPConfig(dgp_type="single", T=50, seed=0))

        np.testing.assert_allclose(meta["mu_expected_true"], meta["mu_true"])

    def test_mixture_dgp_stores_expected_latent_mean(self):
        """Mixture DGP should store the expectation over latent segments."""
        _, _, meta = generate_data(DGPConfig(dgp_type="mixture_k2", T=200, seed=0))

        expected = meta["baseline"] + np.sum(meta["pi_true"][None, :] * meta["hill_mat"], axis=1)

        np.testing.assert_allclose(meta["mu_expected_true"], expected)
        assert not np.allclose(meta["mu_expected_true"], meta["mu_true"])
