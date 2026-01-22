"""Data generation for Hill Mixture MMM experiments.

Provides multiple DGP (Data Generating Process) scenarios for
fair model evaluation:

1. single: Single Hill curve (tests overfitting)
2. mixture_k2: 2-component mixture
3. mixture_k3: 3-component mixture (current standard)
4. mixture_k5: 5-component mixture (tests sparse discovery)
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .transforms import adstock_geometric, hill, hill_matrix


@dataclass
class DGPConfig:
    """Configuration for data generating process."""

    dgp_type: Literal["single", "mixture_k2", "mixture_k3", "mixture_k5"]
    T: int = 200
    sigma: float = 3.0
    alpha: float = 0.5  # adstock decay
    intercept: float = 50.0
    slope: float = 2.0
    seed: int = 42

    @property
    def name(self) -> str:
        return f"{self.dgp_type}_T{self.T}_seed{self.seed}"


# Predefined DGP configurations for benchmarking
DGP_CONFIGS = {
    "single": DGPConfig(dgp_type="single"),
    "mixture_k2": DGPConfig(dgp_type="mixture_k2"),
    "mixture_k3": DGPConfig(dgp_type="mixture_k3"),
    "mixture_k5": DGPConfig(dgp_type="mixture_k5"),
}


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
    elif config.dgp_type == "mixture_k5":
        return _generate_mixture(config, K=5)
    else:
        raise ValueError(f"Unknown DGP type: {config.dgp_type}")


def _generate_single_hill(config: DGPConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate data from a single Hill curve (K=1).

    This is the null hypothesis scenario - mixture models
    should NOT significantly outperform single Hill here.
    """
    rng = np.random.default_rng(config.seed)
    T = config.T

    # Generate spend (log-normal, typical marketing data pattern)
    x = rng.lognormal(mean=1.5, sigma=0.6, size=T).astype(np.float32)

    # Adstock transformation
    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))

    # Baseline (intercept + linear trend)
    t = np.arange(T, dtype=np.float32)
    t_std = (t - t.mean()) / (t.std() + 1e-6)
    baseline = config.intercept + config.slope * t_std

    # Single Hill parameters
    s_median = np.median(s)
    A_true = 30.0
    k_true = s_median  # half-saturation at median spend
    n_true = 1.5

    # Compute effect
    effect = np.array(hill(jnp.array(s), A_true, k_true, n_true))
    mu = baseline + effect

    # Generate observations
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
        "z_true": np.zeros(T, dtype=int),  # all belong to component 0
    }

    return x, y, meta


def _generate_mixture(config: DGPConfig, K: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate data from K-component Hill mixture.

    DGP:
        z_t ~ Categorical(pi)  # latent component assignment
        y_t ~ Normal(baseline_t + hill(s_t; A[z], k[z], n[z]), sigma)
    """
    rng = np.random.default_rng(config.seed)
    T = config.T

    # Generate spend
    x = rng.lognormal(mean=1.5, sigma=0.6, size=T).astype(np.float32)

    # Adstock
    s = np.array(adstock_geometric(jnp.array(x), jnp.array(config.alpha)))

    # Baseline
    t = np.arange(T, dtype=np.float32)
    t_std = (t - t.mean()) / (t.std() + 1e-6)
    baseline = config.intercept + config.slope * t_std

    # Component parameters based on K
    s_median = np.median(s)

    if K == 2:
        pi_true = np.array([0.6, 0.4], dtype=np.float32)
        k_true = np.array([s_median * 0.7, s_median * 1.3], dtype=np.float32)
        A_true = np.array([20.0, 40.0], dtype=np.float32)
        n_true = np.array([2.0, 1.2], dtype=np.float32)

    elif K == 3:
        pi_true = np.array([0.40, 0.30, 0.30], dtype=np.float32)
        k_true = np.array([s_median * 0.5, s_median * 1.0, s_median * 1.2], dtype=np.float32)
        A_true = np.array([15.0, 30.0, 60.0], dtype=np.float32)
        n_true = np.array([2.0, 1.5, 1.0], dtype=np.float32)

    elif K == 5:
        pi_true = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float32)
        k_true = np.array(
            [
                s_median * 0.4,
                s_median * 0.7,
                s_median * 1.0,
                s_median * 1.3,
                s_median * 1.6,
            ],
            dtype=np.float32,
        )
        A_true = np.array([10.0, 20.0, 35.0, 50.0, 70.0], dtype=np.float32)
        n_true = np.array([2.5, 2.0, 1.5, 1.2, 1.0], dtype=np.float32)

    else:
        raise ValueError(f"Unsupported K={K}")

    # Latent component assignment
    z_true = rng.choice(K, size=T, p=pi_true)

    # Compute effects based on latent assignment
    hill_mat = np.array(
        hill_matrix(jnp.array(s), jnp.array(A_true), jnp.array(k_true), jnp.array(n_true))
    )
    effects = hill_mat[np.arange(T), z_true]

    # Generate observations
    mu = baseline + effects
    y = rng.normal(loc=mu, scale=config.sigma).astype(np.float32)

    meta = {
        "dgp_type": f"mixture_k{K}",
        "K_true": K,
        "pi_true": pi_true,
        "A_true": A_true,
        "k_true": k_true,
        "n_true": n_true,
        "alpha_true": config.alpha,
        "intercept_true": config.intercept,
        "slope_true": config.slope,
        "sigma_true": config.sigma,
        "s_median": s_median,
        "s_max": np.max(s),
        "s": s,
        "baseline": baseline,
        "mu_true": mu,
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
    y_range = np.max(y) - np.min(y)
    x_median = np.median(x)
    x_max = np.max(x)

    return {
        # Baseline priors
        "intercept_loc": float(y_mean),
        "intercept_scale": float(y_std * 2),
        "slope_scale": float(y_std),
        # A (max effect): expect fraction of y_range
        "A_loc": float(np.log(y_range * 0.3 + 1e-6)),
        "A_scale": 0.8,
        # k (half-saturation): scaled to x
        "k_base_loc": float(np.log(x_median + 1e-6)),
        "k_scale": 0.7,
        # sigma
        "sigma_scale": float(y_std),
        # Reference values
        "x_median": float(x_median),
        "x_max": float(x_max),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
