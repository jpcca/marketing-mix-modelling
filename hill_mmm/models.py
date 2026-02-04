"""NumPyro model definitions for Hill Mixture MMM.

Four models:
1. model_single_hill: Standard single response curve (baseline)
2. model_hill_mixture: GMM-style with fixed K components
3. model_hill_mixture_sparse: Sparse Dirichlet for automatic K selection
4. model_hill_mixture_reparam: Reparameterized mixture for better MCMC convergence
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


def model_hill_mixture(x, y=None, K=3, prior_config=None):
    """Hill Mixture Model - GMM-style with K components.

    Uses ordered constraint on k (half-saturation) via cumsum to ensure
    identifiability and prevent label switching.

    Args:
        x: (T,) raw spend
        y: (T,) observed response
        K: Number of mixture components
        prior_config: Dict with prior hyperparameters
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
            "sigma_scale": 10.0,
        }

    # Adstock
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

    # Mixture weights (symmetric Dirichlet)
    pis = numpyro.sample("pis", dist.Dirichlet(jnp.ones(K)))

    # Hill parameters with identifiability constraint
    s_max = jnp.max(s)

    # k ordered via cumsum of positive deltas
    k_delta = numpyro.sample(
        "k_delta",
        dist.LogNormal(jnp.log(s_max / (K + 1) + 1e-6), prior_config["k_scale"]).expand([K]),
    )
    k = jnp.cumsum(k_delta)
    numpyro.deterministic("k", k)

    # A and n per component
    A = numpyro.sample(
        "A", dist.LogNormal(prior_config["A_loc"], prior_config["A_scale"]).expand([K])
    )
    n = numpyro.sample("n", dist.LogNormal(jnp.log(1.5), 0.4).expand([K]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(prior_config["sigma_scale"]))

    # Hill transformation
    hill_mat = hill_matrix(s, A, k, n)  # (T, K)
    mu_components = baseline[:, None] + hill_mat  # (T, K)

    # GMM likelihood
    with numpyro.plate("time", T):
        numpyro.sample(
            "y",
            dist.MixtureSameFamily(dist.Categorical(pis), dist.Normal(mu_components, sigma)),
            obs=y,
        )

    # Deterministic quantities for analysis
    mu_expected = baseline + jnp.sum(pis * hill_mat, axis=1)
    numpyro.deterministic("mu_expected", mu_expected)
    numpyro.deterministic("hill_components", hill_mat)


def model_hill_mixture_sparse(x, y=None, K=5, prior_config=None):
    """Hill Mixture Model with sparse Dirichlet prior.

    Uses Dirichlet(0.5, ..., 0.5) which encourages sparsity,
    allowing automatic pruning of unnecessary components.

    Args:
        x: (T,) raw spend
        y: (T,) observed response
        K: Maximum number of components (will be pruned)
        prior_config: Dict with prior hyperparameters
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
            "sigma_scale": 10.0,
        }

    # Adstock
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

    # Sparse Dirichlet (concentration < 1 encourages sparsity)
    pis = numpyro.sample("pis", dist.Dirichlet(jnp.ones(K) * 0.5))

    # Hill parameters
    s_max = jnp.max(s)
    k_delta = numpyro.sample(
        "k_delta",
        dist.LogNormal(jnp.log(s_max / (K + 1) + 1e-6), prior_config["k_scale"]).expand([K]),
    )
    k = jnp.cumsum(k_delta)
    numpyro.deterministic("k", k)

    A = numpyro.sample(
        "A", dist.LogNormal(prior_config["A_loc"], prior_config["A_scale"]).expand([K])
    )
    n = numpyro.sample("n", dist.LogNormal(jnp.log(1.5), 0.4).expand([K]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(prior_config["sigma_scale"]))

    # Hill transformation
    hill_mat = hill_matrix(s, A, k, n)
    mu_components = baseline[:, None] + hill_mat

    with numpyro.plate("time", T):
        numpyro.sample(
            "y",
            dist.MixtureSameFamily(dist.Categorical(pis), dist.Normal(mu_components, sigma)),
            obs=y,
        )

    mu_expected = baseline + jnp.sum(pis * hill_mat, axis=1)
    numpyro.deterministic("mu_expected", mu_expected)
    numpyro.deterministic("hill_components", hill_mat)


def _model_hill_mixture_reparam_inner(x, y=None, K=3, prior_config=None):
    """Inner model for reparameterized Hill Mixture.

    Uses non-centered parameterization (NCP) for better MCMC geometry.
    Key improvements:
    1. Tighter priors for identifiability
    2. LocScaleReparam-friendly parameterization
    3. Simpler mixture structure
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
            "A_scale": 0.5,  # Tighter than original 0.8
            "k_scale": 0.5,  # Tighter than original 0.7
            "n_scale": 0.3,  # Tighter scale for n
            "sigma_scale": 10.0,
        }

    # Adstock parameter
    alpha = numpyro.sample("alpha", dist.Beta(2, 2))
    s = adstock_geometric(x, alpha)
    numpyro.deterministic("s", s)

    # Baseline with LocScaleReparam-friendly priors
    intercept = numpyro.sample(
        "intercept",
        dist.Normal(prior_config["intercept_loc"], prior_config["intercept_scale"]),
    )
    slope = numpyro.sample("slope", dist.Normal(0.0, prior_config["slope_scale"]))
    baseline = intercept + slope * t_std

    # Mixture weights - use stick-breaking for better geometry
    # Stick-breaking: more stable than Dirichlet for MCMC
    stick_proportions = numpyro.sample("stick_proportions", dist.Beta(1.0, 1.0).expand([K - 1]))
    # Convert stick-breaking to simplex
    remaining = jnp.ones(())
    pis_list = []
    for i in range(K - 1):
        pi_i = stick_proportions[i] * remaining
        pis_list.append(pi_i)
        remaining = remaining * (1.0 - stick_proportions[i])
    pis_list.append(remaining)
    pis = jnp.stack(pis_list)
    numpyro.deterministic("pis", pis)

    # Hill parameters with ordering constraint
    s_max = jnp.max(s)
    s_median = jnp.median(s)

    # k: Use ordered parameterization via log-space increments
    # First k centered at median, then increments
    log_k_base = numpyro.sample(
        "log_k_base", dist.Normal(jnp.log(s_median + 1e-6), prior_config["k_scale"])
    )
    log_k_increments = numpyro.sample(
        "log_k_increments",
        dist.Normal(jnp.log(s_max / K + 1e-6), prior_config["k_scale"]).expand([K - 1]),
    )
    # Build ordered k values
    log_k_values = jnp.concatenate(
        [jnp.array([log_k_base]), log_k_base + jnp.cumsum(jnp.abs(log_k_increments))]
    )
    k = jnp.exp(log_k_values)
    numpyro.deterministic("k", k)

    # A: amplitude per component (in log space with LocScaleReparam)
    log_A = numpyro.sample(
        "log_A", dist.Normal(prior_config["A_loc"], prior_config["A_scale"]).expand([K])
    )
    A = jnp.exp(log_A)
    numpyro.deterministic("A", A)

    # n: Hill exponent (in log space)
    log_n = numpyro.sample(
        "log_n", dist.Normal(jnp.log(1.5), prior_config.get("n_scale", 0.3)).expand([K])
    )
    n = jnp.exp(log_n)
    numpyro.deterministic("n", n)

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


def model_hill_mixture_reparam(x, y=None, K=3, prior_config=None):
    """Hill Mixture Model with reparameterization for better MCMC convergence.

    Key improvements over model_hill_mixture:
    1. Non-centered parameterization (NCP) via LocScaleReparam
    2. Stick-breaking construction for mixture weights (better than Dirichlet)
    3. Tighter priors for identifiability
    4. Log-space parameterization for positive parameters

    Use this model when the standard model_hill_mixture has convergence issues
    (high R-hat, low ESS, or very small step sizes).

    Args:
        x: (T,) raw spend
        y: (T,) observed response
        K: Number of mixture components
        prior_config: Dict with prior hyperparameters
    """
    # Apply LocScaleReparam to the Normal distributions
    # This decenters the parameterization for better MCMC geometry
    reparam_config = {
        "intercept": LocScaleReparam(centered=0),
        "slope": LocScaleReparam(centered=0),
        "log_k_base": LocScaleReparam(centered=0),
        "log_k_increments": LocScaleReparam(centered=0),
        "log_A": LocScaleReparam(centered=0),
        "log_n": LocScaleReparam(centered=0),
    }

    with reparam(config=reparam_config):
        _model_hill_mixture_reparam_inner(x, y, K, prior_config)
