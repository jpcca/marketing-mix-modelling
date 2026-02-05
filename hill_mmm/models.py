"""NumPyro model definitions for Hill Mixture MMM.

Two models:
1. model_single_hill: Standard single response curve (baseline)
2. model_hill_mixture_hierarchical_reparam: Hierarchical mixture with reparameterization
   - Combines partial pooling (hierarchical priors) with non-centered parameterization
   - Best balance of convergence and predictive accuracy
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam

from .transforms import adstock_geometric, hill, hill_matrix


def model_single_hill(x, y=None, prior_config=None):
    """Single Hill function MMM.

    x -> adstock(x, alpha) -> hill(s, A, k, n) -> y

    Args:
        x: (T,) raw spend
        y: (T,) observed response (None for prior predictive)
        prior_config: Dict with prior hyperparameters
    """
    T = x.shape[0]
    t = jnp.arange(T, dtype=jnp.float32)
    t_std = (t - jnp.mean(t)) / (jnp.std(t) + 1e-6)

    # Default prior config
    if prior_config is None:
        prior_config = {
            "intercept_loc": 50.0,
            "intercept_scale": 20.0,
            "slope_scale": 5.0,
            "A_loc": np.log(30.0),
            "A_scale": 0.8,
            "k_base_loc": np.log(10.0),
            "k_scale": 0.7,
            "sigma_scale": 10.0,
        }

    # Adstock parameter
    alpha = numpyro.sample("alpha", dist.Beta(2, 2))
    s = adstock_geometric(x, alpha)
    numpyro.deterministic("s", s)

    # Baseline (intercept + trend)
    intercept = numpyro.sample(
        "intercept",
        dist.Normal(prior_config["intercept_loc"], prior_config["intercept_scale"]),
    )
    slope = numpyro.sample("slope", dist.Normal(0.0, prior_config["slope_scale"]))
    baseline = intercept + slope * t_std

    # Hill parameters
    s_median = jnp.median(s)
    A = numpyro.sample("A", dist.LogNormal(prior_config["A_loc"], prior_config["A_scale"]))
    k = numpyro.sample("k", dist.LogNormal(jnp.log(s_median + 1e-6), prior_config["k_scale"]))
    n = numpyro.sample("n", dist.LogNormal(jnp.log(1.5), 0.4))
    sigma = numpyro.sample("sigma", dist.HalfNormal(prior_config["sigma_scale"]))

    # Effect and likelihood
    effect = hill(s, A, k, n)
    mu = baseline + effect

    with numpyro.plate("time", T):
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    numpyro.deterministic("mu", mu)
    numpyro.deterministic("effect", effect)


def _model_hill_mixture_hierarchical_reparam_inner(x, y=None, K=3, prior_config=None):
    """Inner model for hierarchical reparameterized Hill Mixture.

    Combines hierarchical priors (partial pooling) with non-centered parameterization.
    Key features:
    1. Hierarchical priors on A, k, n - components share information
    2. Non-centered parameterization for all hierarchical parameters
    3. Stick-breaking for mixture weights
    4. Ordered constraint on k for identifiability
    """
    T = x.shape[0]
    t = jnp.arange(T, dtype=jnp.float32)
    t_std = (t - jnp.mean(t)) / (jnp.std(t) + 1e-6)

    if prior_config is None:
        prior_config = {
            "intercept_loc": 50.0,
            "intercept_scale": 20.0,
            "slope_scale": 5.0,
            "A_loc": np.log(30.0),
            "A_scale": 0.8,
            "k_scale": 0.7,
            "n_scale": 0.4,
            "sigma_scale": 10.0,
        }

    # Adstock parameter
    alpha = numpyro.sample("alpha", dist.Beta(2, 2))
    s = adstock_geometric(x, alpha)
    numpyro.deterministic("s", s)

    # Baseline
    intercept = numpyro.sample(
        "intercept",
        dist.Normal(prior_config["intercept_loc"], prior_config["intercept_scale"]),
    )
    slope = numpyro.sample("slope", dist.Normal(0.0, prior_config["slope_scale"]))
    baseline = intercept + slope * t_std

    # Mixture weights - stick-breaking
    stick_proportions = numpyro.sample("stick_proportions", dist.Beta(1.0, 1.0).expand([K - 1]))
    remaining = jnp.ones(())
    pis_list = []
    for i in range(K - 1):
        pi_i = stick_proportions[i] * remaining
        pis_list.append(pi_i)
        remaining = remaining * (1.0 - stick_proportions[i])
    pis_list.append(remaining)
    pis = jnp.stack(pis_list)
    numpyro.deterministic("pis", pis)

    # ========== HIERARCHICAL PRIORS ==========
    # Hyperpriors for amplitude A (shared across components)
    mu_log_A = numpyro.sample("mu_log_A", dist.Normal(prior_config["A_loc"], 0.5))
    sigma_log_A = numpyro.sample("sigma_log_A", dist.HalfNormal(0.5))

    # Hyperpriors for Hill exponent n (shared across components)
    mu_log_n = numpyro.sample("mu_log_n", dist.Normal(jnp.log(1.5), 0.3))
    sigma_log_n = numpyro.sample("sigma_log_n", dist.HalfNormal(0.3))

    # ========== NON-CENTERED COMPONENT PARAMETERS ==========
    # A: amplitude per component (hierarchical, non-centered)
    log_A_raw = numpyro.sample("log_A_raw", dist.Normal(0, 1).expand([K]))
    log_A = mu_log_A + sigma_log_A * log_A_raw
    A = jnp.exp(log_A)
    numpyro.deterministic("log_A", log_A)
    numpyro.deterministic("A", A)

    # n: Hill exponent per component (hierarchical, non-centered)
    log_n_raw = numpyro.sample("log_n_raw", dist.Normal(0, 1).expand([K]))
    log_n = mu_log_n + sigma_log_n * log_n_raw
    n = jnp.exp(log_n)
    numpyro.deterministic("log_n", log_n)
    numpyro.deterministic("n", n)

    # k: half-saturation with ordering constraint
    # Use cumsum of positive increments for ordering
    s_median = jnp.median(s)

    # Base k centered at median
    log_k_base = numpyro.sample(
        "log_k_base", dist.Normal(jnp.log(s_median + 1e-6), prior_config["k_scale"])
    )
    # Increments (positive via softplus or abs)
    log_k_increments_raw = numpyro.sample(
        "log_k_increments_raw",
        dist.Normal(0, 1).expand([K - 1]),
    )
    # Scale increments to reasonable range
    log_k_increments = jnp.abs(log_k_increments_raw) * prior_config["k_scale"]

    # Build ordered k values
    log_k_values = jnp.concatenate(
        [jnp.array([log_k_base]), log_k_base + jnp.cumsum(log_k_increments)]
    )
    k = jnp.exp(log_k_values)
    numpyro.deterministic("k", k)

    # Observation noise
    sigma = numpyro.sample("sigma", dist.HalfNormal(prior_config["sigma_scale"]))

    # Hill transformation for each component
    hill_mat = hill_matrix(s, A, k, n)  # (T, K)
    mu_components = baseline[:, None] + hill_mat  # (T, K)

    # GMM likelihood
    with numpyro.plate("time", T):
        numpyro.sample(
            "y",
            dist.MixtureSameFamily(dist.Categorical(pis), dist.Normal(mu_components, sigma)),
            obs=y,
        )

    # Deterministic quantities
    mu_expected = baseline + jnp.sum(pis * hill_mat, axis=1)
    numpyro.deterministic("mu_expected", mu_expected)
    numpyro.deterministic("hill_components", hill_mat)


def model_hill_mixture_hierarchical_reparam(x, y=None, K=3, prior_config=None):
    """Hierarchical Hill Mixture Model with reparameterization.

    Combines the best of both worlds:
    1. Hierarchical priors for partial pooling across components (better regularization)
    2. Non-centered parameterization for MCMC convergence
    3. Stick-breaking for mixture weights
    4. Ordered constraint on k for identifiability

    The hierarchical structure shares information across mixture components,
    which can improve predictive performance especially with limited data.
    The non-centered parameterization handles the "funnel" geometry that
    typically causes convergence issues in hierarchical models.

    Args:
        x: (T,) raw spend
        y: (T,) observed response
        K: Number of mixture components
        prior_config: Dict with prior hyperparameters
    """
    # Apply LocScaleReparam to baseline parameters
    # Note: hierarchical parameters use manual non-centering above
    reparam_config = {
        "intercept": LocScaleReparam(centered=0),
        "slope": LocScaleReparam(centered=0),
        "log_k_base": LocScaleReparam(centered=0),
        "mu_log_A": LocScaleReparam(centered=0),
        "mu_log_n": LocScaleReparam(centered=0),
    }

    with reparam(config=reparam_config):
        _model_hill_mixture_hierarchical_reparam_inner(x, y, K, prior_config)
