"""Data generation for Hill Mixture MMM experiments.

Provides multiple DGP (Data Generating Process) scenarios for
fair model evaluation:

1. single: Single Hill curve (tests overfitting)
2. mixture_k2: 2-component mixture
3. mixture_k3: 3-component mixture

The benchmark-facing mixture DGPs anchor their half-saturation points to
quantiles of the realized adstocked spend support. This keeps the synthetic
difficulty focused on inference instead of accidental near-ties caused by
curves being evaluated on too narrow a spend range.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .baseline import linear_baseline, standardized_time_index
from .transforms import adstock_geometric, hill, hill_matrix


@dataclass
class DGPConfig:
    """Configuration for data generating process."""

    dgp_type: Literal["single", "mixture_k2", "mixture_k3"]
    T: int = 200
    sigma: float = 3.0
    alpha: float = 0.5
    intercept: float = 50.0
    slope: float = 2.0
    seed: int = 42

    @property
    def name(self) -> str:
        return f"{self.dgp_type}_T{self.T}_seed{self.seed}"


DGP_CONFIGS = {
    "single": DGPConfig(dgp_type="single"),
    "mixture_k2": DGPConfig(dgp_type="mixture_k2"),
    "mixture_k3": DGPConfig(dgp_type="mixture_k3"),
}

A_PRIOR_RANGE_FRACTION = 0.5
RAW_SPEND_LOGNORMAL_MEAN = 1.5
RAW_SPEND_LOGNORMAL_SIGMA = 0.6


def _draw_raw_spend(rng: np.random.Generator, T: int) -> np.ndarray:
    """Draw one synthetic raw spend series."""
    return rng.lognormal(
        mean=RAW_SPEND_LOGNORMAL_MEAN, sigma=RAW_SPEND_LOGNORMAL_SIGMA, size=T
    ).astype(np.float32)


def _support_quantiles(values: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Return selected quantiles on the realized support as float32."""
    return np.asarray(np.quantile(values, quantiles), dtype=np.float32)


def generate_data(config: DGPConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate synthetic MMM data from specified DGP.

    Args:
        config: DGP configuration

    Returns:
        x: (T,) raw spend
        y: (T,) observed response
        meta: Dict with true parameters and intermediate values
    """
    if config.dgp_type == "single":
        return _generate_single_hill(config)
    elif config.dgp_type == "mixture_k2":
        return _generate_mixture(config, K=2)
    elif config.dgp_type == "mixture_k3":
        return _generate_mixture(config, K=3)
    else:
        raise ValueError(f"Unknown DGP type: {config.dgp_type}")


def _generate_single_hill(config: DGPConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate data from a single Hill curve (K=1)."""
    rng = np.random.default_rng(config.seed)
    T = config.T

    x = _draw_raw_spend(rng, T)

    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))

    t_std = standardized_time_index(T)
    baseline = linear_baseline(config.intercept, config.slope, t_std)

    s_median = np.median(s)
    A_true = 30.0
    k_true = s_median
    n_true = 1.5

    effect = np.array(hill(jnp.array(s), A_true, k_true, n_true))
    mu = baseline + effect

    y = rng.normal(loc=mu, scale=config.sigma).astype(np.float32)

    meta = {
        "dgp_type": "single",
        "K_true": 1,
        "pi_true": np.array([1.0]),
        "A_true": np.array([A_true]),
        "k_true": np.array([k_true]),
        "n_true": np.array([n_true]),
        "alpha_true": config.alpha,
        "intercept_true": config.intercept,
        "slope_true": config.slope,
        "sigma_true": config.sigma,
        "s_median": s_median,
        "s_max": np.max(s),
        "s": s,
        "baseline": baseline,
        "mu_true": mu,
        "mu_expected_true": mu,
        "z_true": np.zeros(T, dtype=int),
    }

    return x, y, meta


def _generate_mixture(config: DGPConfig, K: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate data from K-component Hill mixture.

    DGP:
        z_t ~ Categorical(pi)
        y_t ~ Normal(baseline_t + hill(s_t; A[z], k[z], n[z]), sigma)
    """
    rng = np.random.default_rng(config.seed)
    T = config.T

    x = _draw_raw_spend(rng, T)

    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))

    t_std = standardized_time_index(T)
    baseline = linear_baseline(config.intercept, config.slope, t_std)

    s_median = np.median(s)
    k_quantiles = None

    if K == 2:
        pi_true = np.array([0.55, 0.45], dtype=np.float32)
        k_quantiles = [0.25, 0.85]
        k_true = _support_quantiles(s, k_quantiles)
        A_true = np.array([10.0, 50.0], dtype=np.float32)
        n_true = np.array([2.0, 0.8], dtype=np.float32)

    elif K == 3:
        pi_true = np.array([0.40, 0.35, 0.25], dtype=np.float32)
        k_quantiles = [0.15, 0.60, 0.95]
        k_true = _support_quantiles(s, k_quantiles)
        A_true = np.array([12.0, 35.0, 85.0], dtype=np.float32)
        n_true = np.array([0.75, 1.35, 2.1], dtype=np.float32)

    else:
        raise ValueError(f"Unsupported K={K}")

    z_true = rng.choice(K, size=T, p=pi_true)

    hill_mat = np.array(
        hill_matrix(jnp.array(s), jnp.array(A_true), jnp.array(k_true), jnp.array(n_true))
    )
    effects = hill_mat[np.arange(T), z_true]
    expected_effects = np.sum(pi_true[None, :] * hill_mat, axis=1)

    mu = baseline + effects
    mu_expected = baseline + expected_effects
    y = rng.normal(loc=mu, scale=config.sigma).astype(np.float32)

    meta = {
        "dgp_type": f"mixture_k{K}",
        "K_true": K,
        "pi_true": pi_true,
        "A_true": A_true,
        "k_true": k_true,
        "n_true": n_true,
        "k_quantiles_true": None
        if k_quantiles is None
        else np.asarray(k_quantiles, dtype=np.float32),
        "alpha_true": config.alpha,
        "intercept_true": config.intercept,
        "slope_true": config.slope,
        "sigma_true": config.sigma,
        "s_median": s_median,
        "s_max": np.max(s),
        "s": s,
        "baseline": baseline,
        "mu_true": mu,
        "mu_expected_true": mu_expected,
        "z_true": z_true,
        "hill_mat": hill_mat,
    }

    return x, y, meta


def compute_prior_config(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute prior hyperparameters from data.

    Uses empirical Bayes approach to set reasonable priors
    based on data scale.

    Args:
        x: (T,) spend values
        y: (T,) response values

    Returns:
        Dict with prior configuration
    """
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_min = np.min(y)
    y_range = np.max(y) - y_min
    x_median = np.median(x)
    x_max = np.max(x)
    effect_scale = y_range * A_PRIOR_RANGE_FRACTION
    intercept_loc = max(float(y_min), float(y_mean - effect_scale))

    return {
        "intercept_loc": intercept_loc,
        "intercept_scale": float(y_std * 2),
        "slope_scale": float(y_std),
        "A_loc": float(np.log(effect_scale + 1e-6)),
        "A_scale": 0.8,
        "k_base_loc": float(np.log(x_median + 1e-6)),
        "k_scale": 0.7,
        "n_loc": float(np.log(1.5)),
        "n_scale": 0.4,
        "sigma_log_A_loc": 0.0,
        "sigma_log_A_scale": 0.8,
        "sigma_log_n_loc": -0.5,
        "sigma_log_n_scale": 0.8,
        "sigma_scale": float(y_std),
        "x_median": float(x_median),
        "x_max": float(x_max),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
