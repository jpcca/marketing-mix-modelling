"""Tests for inference utilities."""

import numpy as np

from hill_mixture_mmm.baseline import linear_baseline, standardized_time_index
from hill_mixture_mmm.inference import compute_mixture_log_likelihood
from hill_mixture_mmm.transforms import adstock_geometric, hill_matrix


def _reference_mixture_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    alpha_key: str = "alpha",
) -> np.ndarray:
    """Naive loop implementation used as a numerical reference in tests."""
    t_std = standardized_time_index(len(x))
    log_two_pi = np.log(2.0 * np.pi)
    n_samples = samples["A"].shape[0]
    out = np.zeros(n_samples)

    for i in range(n_samples):
        alpha = float(samples[alpha_key][i])
        intercept = float(samples["intercept"][i])
        slope = float(samples["slope"][i])
        sigma = float(samples["sigma"][i])
        A = samples["A"][i]
        k = samples["k"][i]
        n = samples["n"][i]
        pis = samples["pis"][i]

        s = np.asarray(adstock_geometric(x, alpha))
        baseline = linear_baseline(intercept, slope, t_std)
        mu = baseline[:, None] + np.asarray(hill_matrix(s, A, k, n))

        sigma_sq = sigma**2
        log_normal = -0.5 * (
            log_two_pi + np.log(sigma_sq) + ((y[:, None] - mu) ** 2) / sigma_sq
        )
        log_probs = np.log(pis + 1e-10) + log_normal
        out[i] = np.logaddexp.reduce(log_probs, axis=1).sum()

    return out


class TestMixtureLogLikelihood:
    """Tests for label-invariant mixture log-likelihood computation."""

    def test_matches_reference_loop(self):
        """Vectorized implementation should match the previous loop-based result."""
        rng = np.random.default_rng(0)
        T = 10
        n_samples = 5
        K = 3

        x = rng.lognormal(mean=1.0, sigma=0.3, size=T).astype(np.float32)
        y = rng.normal(loc=20.0, scale=3.0, size=T).astype(np.float32)
        samples = {
            "alpha": rng.uniform(0.1, 0.8, size=n_samples).astype(np.float32),
            "intercept": rng.normal(18.0, 2.0, size=n_samples).astype(np.float32),
            "slope": rng.normal(0.0, 1.0, size=n_samples).astype(np.float32),
            "sigma": rng.uniform(1.0, 4.0, size=n_samples).astype(np.float32),
            "A": rng.uniform(5.0, 30.0, size=(n_samples, K)).astype(np.float32),
            "k": rng.uniform(1.0, 8.0, size=(n_samples, K)).astype(np.float32),
            "n": rng.uniform(0.8, 2.5, size=(n_samples, K)).astype(np.float32),
            "pis": rng.dirichlet(np.ones(K), size=n_samples).astype(np.float32),
        }

        expected = _reference_mixture_log_likelihood(x, y, samples)
        actual = compute_mixture_log_likelihood(x, y, samples)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_respects_custom_alpha_key(self):
        """Custom alpha_key should produce the same values as the default path."""
        rng = np.random.default_rng(1)
        T = 8
        n_samples = 4
        K = 2

        x = rng.lognormal(mean=1.2, sigma=0.4, size=T).astype(np.float32)
        y = rng.normal(loc=15.0, scale=2.0, size=T).astype(np.float32)
        decay = rng.uniform(0.05, 0.6, size=n_samples).astype(np.float32)
        samples = {
            "decay": decay,
            "intercept": rng.normal(14.0, 1.0, size=n_samples).astype(np.float32),
            "slope": rng.normal(0.0, 0.8, size=n_samples).astype(np.float32),
            "sigma": rng.uniform(0.8, 2.5, size=n_samples).astype(np.float32),
            "A": rng.uniform(4.0, 12.0, size=(n_samples, K)).astype(np.float32),
            "k": rng.uniform(0.7, 5.0, size=(n_samples, K)).astype(np.float32),
            "n": rng.uniform(0.7, 2.0, size=(n_samples, K)).astype(np.float32),
            "pis": rng.dirichlet(np.ones(K), size=n_samples).astype(np.float32),
        }

        expected = _reference_mixture_log_likelihood(x, y, samples, alpha_key="decay")
        actual = compute_mixture_log_likelihood(x, y, samples, alpha_key="decay")

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
