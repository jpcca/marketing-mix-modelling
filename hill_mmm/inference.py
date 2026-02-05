"""Inference utilities for Hill Mixture MMM.

Handles MCMC execution, posterior predictive, and basic evaluation.
"""

from collections.abc import Callable
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive


def run_inference(
    model_fn: Callable,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    prior_config: dict | None = None,
    **model_kwargs,
) -> MCMC:
    """Run MCMC inference.

    Args:
        model_fn: NumPyro model function
        x: (T,) spend values
        y: (T,) response values
        seed: Random seed
        num_warmup: Warmup iterations per chain
        num_samples: Samples per chain
        num_chains: Number of parallel chains
        prior_config: Prior hyperparameters
        **model_kwargs: Additional model arguments (e.g., K for mixture)

    Returns:
        MCMC object with samples
    """
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(model_fn, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(
        rng_key,
        x=jnp.array(x),
        y=jnp.array(y),
        prior_config=prior_config,
        **model_kwargs,
    )
    return mcmc


def compute_predictions(
    mcmc: MCMC,
    model_fn: Callable,
    x: np.ndarray,
    prior_config: dict | None = None,
    **model_kwargs,
) -> dict[str, np.ndarray]:
    """Compute posterior predictive samples.

    Args:
        mcmc: Fitted MCMC object
        model_fn: NumPyro model function
        x: (T,) spend values for prediction
        prior_config: Prior hyperparameters
        **model_kwargs: Additional model arguments

    Returns:
        Dict with predicted samples for each variable
    """
    samples = mcmc.get_samples()
    pred = Predictive(model_fn, posterior_samples=samples)
    pred_samples = pred(
        jax.random.PRNGKey(99),
        x=jnp.array(x),
        y=None,
        prior_config=prior_config,
        **model_kwargs,
    )
    return {k: np.array(v) for k, v in pred_samples.items()}


def compute_loo(mcmc: MCMC) -> dict[str, Any]:
    """Compute LOO-CV (PSIS-LOO) using ArviZ.

    Returns:
        Dict with elpd_loo, se, p_loo, and pareto_k diagnostics
    """
    idata = az.from_numpyro(mcmc)
    try:
        loo = az.loo(idata, pointwise=True)
        pareto_k = loo.pareto_k.values
        return {
            "elpd_loo": float(loo.elpd_loo),
            "se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "pareto_k_bad": int(np.sum(pareto_k > 0.7)),
            "pareto_k_very_bad": int(np.sum(pareto_k > 1.0)),
            "n_obs": len(pareto_k),
        }
    except Exception as e:
        return {"elpd_loo": np.nan, "error": str(e)}


def compute_waic(mcmc: MCMC) -> dict[str, Any]:
    """Compute WAIC using ArviZ.

    Returns:
        Dict with elpd_waic, se, p_waic
    """
    idata = az.from_numpyro(mcmc)
    try:
        waic = az.waic(idata)
        return {
            "elpd_waic": float(waic.elpd_waic),
            "se": float(waic.se),
            "p_waic": float(waic.p_waic),
        }
    except Exception as e:
        return {"elpd_waic": np.nan, "error": str(e)}


def compute_convergence_diagnostics(mcmc: MCMC) -> dict[str, Any]:
    """Compute R-hat and ESS diagnostics.

    Returns:
        Dict with max_rhat, min_ess_bulk, min_ess_tail, and per-param details
    """
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata, kind="diagnostics")

    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "converged": bool(summary["r_hat"].max() < 1.05),
        "ess_sufficient": bool(summary["ess_bulk"].min() > 400),
    }


def compute_predictive_metrics(y_true: np.ndarray, y_samples: np.ndarray) -> dict[str, float]:
    """Compute RMSE and coverage metrics.

    Args:
        y_true: (T,) true observations
        y_samples: (n_samples, T) posterior predictive samples

    Returns:
        Dict with rmse, coverage_90, mean predictions
    """
    y_pred_mean = y_samples.mean(axis=0)
    q05 = np.quantile(y_samples, 0.05, axis=0)
    q95 = np.quantile(y_samples, 0.95, axis=0)

    rmse = float(np.sqrt(np.mean((y_pred_mean - y_true) ** 2)))
    coverage = float(np.mean((y_true >= q05) & (y_true <= q95)))

    return {
        "rmse": rmse,
        "coverage_90": coverage,
        "y_pred_mean": y_pred_mean,
        "q05": q05,
        "q95": q95,
    }


# =============================================================================
# POST-HOC RELABELING FOR LABEL SWITCHING
# =============================================================================


def relabel_samples_by_k(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Relabel mixture components by sorting on k (half-saturation).

    This function addresses label switching in mixture models by sorting
    components by their k values at each MCMC iteration. This is a post-hoc
    approach that preserves detailed balance (unlike within-MCMC ordering).

    For each sample, components are reordered so that k[0] < k[1] < ... < k[K-1].
    All component-specific parameters (A, k, n, pis) are permuted consistently.

    Args:
        samples: Dict of MCMC samples with keys including 'k', 'A', 'n', 'pis'
                 Each array has shape (n_samples,) or (n_samples, K)

    Returns:
        Dict with relabeled samples. Component-specific parameters are sorted
        by k values. Non-component parameters are passed through unchanged.

    Example:
        >>> mcmc = run_inference(model_hill_mixture_unconstrained, x, y)
        >>> samples = mcmc.get_samples()
        >>> relabeled = relabel_samples_by_k(samples)
        >>> # Now k[:, 0] < k[:, 1] < k[:, 2] for all samples
    """
    # Component-specific parameters to relabel
    component_params = ["k", "A", "n", "pis", "log_k", "log_A", "log_n"]

    # Check if k exists
    if "k" not in samples:
        raise ValueError("samples must contain 'k' for relabeling")

    k = samples["k"]  # Shape: (n_samples, K)
    n_samples, K = k.shape

    # Get sort indices for each sample (ascending order by k)
    sort_indices = np.argsort(k, axis=1)  # Shape: (n_samples, K)

    # Create output dict
    relabeled = {}

    for key, value in samples.items():
        if key in component_params and value.ndim == 2 and value.shape[1] == K:
            # Relabel this parameter using fancy indexing
            # For each sample i, we want value[i, sort_indices[i, :]]
            relabeled[key] = np.take_along_axis(value, sort_indices, axis=1)
        else:
            # Pass through unchanged
            relabeled[key] = value

    return relabeled


def check_label_switching(samples: dict[str, np.ndarray], param: str = "k") -> dict[str, Any]:
    """Diagnose label switching by analyzing component ordering over samples.

    Checks how often the ordering of components (by the specified parameter)
    changes across MCMC samples. High switching rates indicate label switching.

    Args:
        samples: Dict of MCMC samples
        param: Parameter to check ordering on (default: 'k')

    Returns:
        Dict with diagnostics:
        - switching_rate: Fraction of samples where ordering differs from mode
        - n_unique_orderings: Number of distinct orderings observed
        - mode_ordering: Most common component ordering
    """
    if param not in samples:
        raise ValueError(f"samples must contain '{param}'")

    values = samples[param]  # Shape: (n_samples, K)
    n_samples, K = values.shape

    # Get ordering for each sample
    orderings = np.argsort(values, axis=1)  # Shape: (n_samples, K)

    # Convert orderings to tuples for counting
    ordering_tuples = [tuple(o) for o in orderings]

    # Count occurrences
    from collections import Counter

    counts = Counter(ordering_tuples)
    mode_ordering = counts.most_common(1)[0][0]
    mode_count = counts[mode_ordering]

    return {
        "switching_rate": 1.0 - (mode_count / n_samples),
        "n_unique_orderings": len(counts),
        "mode_ordering": mode_ordering,
        "mode_count": mode_count,
        "n_samples": n_samples,
        "top_orderings": counts.most_common(min(5, len(counts))),
    }


# =============================================================================
# LABEL-INVARIANT CONVERGENCE DIAGNOSTICS
# =============================================================================


def compute_mixture_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    samples: dict[str, np.ndarray],
    alpha_key: str = "alpha",
) -> np.ndarray:
    """Compute per-sample log-likelihood for mixture model.

    The log-likelihood is label-invariant: it has the same value regardless
    of how components are labeled. This makes it suitable for convergence
    diagnostics in mixture models.

    Args:
        x: (T,) spend values
        y: (T,) observed response values
        samples: Dict of MCMC samples containing 'A', 'k', 'n', 'pis', 'sigma',
                 'intercept', 'slope', and alpha_key

    Returns:
        (n_samples,) array of total log-likelihood per MCMC sample
    """
    from .transforms import adstock_geometric, hill_matrix

    T = len(x)
    n_samples = samples["A"].shape[0]

    # Standardized time
    t = np.arange(T, dtype=np.float32)
    t_std = (t - np.mean(t)) / (np.std(t) + 1e-6)

    log_likelihoods = np.zeros(n_samples)

    for i in range(n_samples):
        # Get parameters for this sample
        alpha = float(samples[alpha_key][i])
        intercept = float(samples["intercept"][i])
        slope = float(samples["slope"][i])
        A = samples["A"][i]  # (K,)
        k = samples["k"][i]  # (K,)
        n = samples["n"][i]  # (K,)
        pis = samples["pis"][i]  # (K,)
        sigma = float(samples["sigma"][i])

        # Compute adstock
        s = adstock_geometric(jnp.array(x), alpha)
        s = np.array(s)

        # Baseline
        baseline = intercept + slope * t_std

        # Hill components
        hill_mat = hill_matrix(jnp.array(s), jnp.array(A), jnp.array(k), jnp.array(n))
        hill_mat = np.array(hill_mat)  # (T, K)

        # Component means: baseline + hill effect
        mu_components = baseline[:, None] + hill_mat  # (T, K)

        # Mixture log-likelihood: log sum_k pi_k * N(y_t | mu_tk, sigma)
        # = log sum_k exp(log(pi_k) + log N(y_t | mu_tk, sigma))
        from scipy.stats import norm

        log_pis = np.log(pis + 1e-10)
        log_probs = np.zeros((T, len(pis)))
        for k_idx in range(len(pis)):
            log_probs[:, k_idx] = log_pis[k_idx] + norm.logpdf(y, mu_components[:, k_idx], sigma)

        # Log-sum-exp for numerical stability
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        log_likelihood_per_obs = max_log_probs.squeeze() + np.log(
            np.sum(np.exp(log_probs - max_log_probs), axis=1)
        )
        log_likelihoods[i] = np.sum(log_likelihood_per_obs)

    return log_likelihoods


def compute_label_invariant_diagnostics(
    mcmc: MCMC,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "rank",
) -> dict[str, Any]:
    """Compute convergence diagnostics using label-invariant quantities.

    Standard R-hat on component parameters is unreliable for mixture models
    due to label switching. This function computes R-hat on:
    1. Total log-likelihood (label-invariant)
    2. Other scalar parameters (intercept, slope, sigma, alpha)

    Args:
        mcmc: Fitted MCMC object
        x: (T,) spend values
        y: (T,) observed response values
        method: R-hat method - 'rank' (recommended) or 'split'

    Returns:
        Dict with:
        - rhat_log_lik: R-hat on total log-likelihood
        - rhat_scalars: R-hat on scalar parameters
        - converged: Whether all R-hats < 1.01 (stricter threshold)
        - details: Per-parameter diagnostics
    """
    samples = mcmc.get_samples(group_by_chain=True)
    n_chains = samples["sigma"].shape[0]

    # Compute log-likelihood per chain
    log_liks_by_chain = []
    for chain_idx in range(n_chains):
        chain_samples = {k: v[chain_idx] for k, v in samples.items()}
        log_lik = compute_mixture_log_likelihood(x, y, chain_samples)
        log_liks_by_chain.append(log_lik)

    log_liks = np.stack(log_liks_by_chain)  # (n_chains, n_samples_per_chain)

    # Compute R-hat on log-likelihood
    rhat_log_lik = _compute_rhat(log_liks, method=method)

    # Compute R-hat on scalar parameters
    scalar_params = ["intercept", "slope", "sigma", "alpha"]
    rhat_scalars = {}
    for param in scalar_params:
        if param in samples:
            param_values = samples[param]  # (n_chains, n_samples)
            if param_values.ndim == 2:  # Scalar parameter
                rhat_scalars[param] = _compute_rhat(param_values, method=method)

    # Check convergence with stricter threshold (1.01 per Vehtari et al.)
    all_rhats = [rhat_log_lik] + list(rhat_scalars.values())
    max_rhat = max(all_rhats)
    converged = max_rhat < 1.01

    return {
        "rhat_log_lik": rhat_log_lik,
        "rhat_scalars": rhat_scalars,
        "max_rhat": max_rhat,
        "converged": converged,
        "method": method,
        "threshold": 1.01,
    }


def compute_diagnostics_on_relabeled(
    mcmc: MCMC,
    method: str = "rank",
) -> dict[str, Any]:
    """Compute convergence diagnostics on relabeled samples.

    After relabeling by k, component parameters should be comparable
    across chains. This allows standard R-hat to be meaningful.

    Args:
        mcmc: Fitted MCMC object
        method: R-hat method - 'rank' (recommended) or 'split'

    Returns:
        Dict with per-parameter R-hat on relabeled samples
    """
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    n_chains = list(samples_by_chain.values())[0].shape[0]

    # Relabel each chain independently
    relabeled_by_chain = []
    for chain_idx in range(n_chains):
        chain_samples = {k: v[chain_idx] for k, v in samples_by_chain.items()}
        relabeled = relabel_samples_by_k(chain_samples)
        relabeled_by_chain.append(relabeled)

    # Stack back into (n_chains, n_samples, ...) format
    relabeled_samples = {}
    for key in relabeled_by_chain[0].keys():
        relabeled_samples[key] = np.stack([r[key] for r in relabeled_by_chain])

    # Compute R-hat on component parameters
    component_params = ["A", "k", "n", "pis"]
    results = {}

    for param in component_params:
        if param in relabeled_samples:
            values = relabeled_samples[param]  # (n_chains, n_samples, K)
            if values.ndim == 3:
                K = values.shape[2]
                rhats = []
                for k_idx in range(K):
                    rhat = _compute_rhat(values[:, :, k_idx], method=method)
                    rhats.append(rhat)
                results[param] = {
                    "per_component": rhats,
                    "max": max(rhats),
                }

    # Overall convergence
    max_rhat = max(r["max"] for r in results.values())
    converged = max_rhat < 1.01

    return {
        "component_rhats": results,
        "max_rhat": max_rhat,
        "converged": converged,
        "method": method,
        "threshold": 1.01,
    }


def _compute_rhat(values: np.ndarray, method: str = "rank") -> float:
    """Compute R-hat using ArviZ.

    Args:
        values: (n_chains, n_samples) array
        method: 'rank' for rank-normalized R-hat (recommended),
                'split' for split R-hat

    Returns:
        R-hat value
    """
    import xarray as xr

    n_chains, n_samples = values.shape

    # Create xarray DataArray in ArviZ format
    da = xr.DataArray(
        values,
        dims=["chain", "draw"],
        coords={"chain": np.arange(n_chains), "draw": np.arange(n_samples)},
    )

    # Compute R-hat
    if method == "rank":
        rhat = az.rhat(da, method="rank")
    else:
        rhat = az.rhat(da, method="split")

    # az.rhat returns float for single variable, DataArray for multiple
    # Use hasattr to handle both cases safely
    if hasattr(rhat, "values"):
        return float(rhat.values)  # type: ignore[union-attr]
    return float(rhat)


def compute_comprehensive_mixture_diagnostics(
    mcmc: MCMC,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "rank",
) -> dict[str, Any]:
    """Compute comprehensive convergence diagnostics for mixture models.

    This function combines multiple diagnostic approaches:
    1. Standard diagnostics (for reference, may be unreliable)
    2. Label-invariant diagnostics (R-hat on log-likelihood)
    3. Diagnostics on relabeled samples
    4. Label switching detection

    Args:
        mcmc: Fitted MCMC object
        x: (T,) spend values
        y: (T,) observed response values
        method: R-hat method - 'rank' (recommended) or 'split'

    Returns:
        Dict with comprehensive diagnostics and recommendations
    """
    # 1. Standard diagnostics (for reference)
    standard = compute_convergence_diagnostics(mcmc)

    # 2. Label-invariant diagnostics
    label_invariant = compute_label_invariant_diagnostics(mcmc, x, y, method=method)

    # 3. Diagnostics on relabeled samples
    relabeled = compute_diagnostics_on_relabeled(mcmc, method=method)

    # 4. Label switching detection
    samples = mcmc.get_samples()
    switching = check_label_switching(samples, param="k")

    # Overall assessment
    # Use label-invariant log-likelihood R-hat as primary criterion
    primary_converged = label_invariant["converged"]

    # Check if relabeled parameters also converged
    relabeled_converged = relabeled["converged"]

    # High switching rate suggests label switching is occurring
    # (Used in recommendation logic below)
    _significant_switching = switching["switching_rate"] > 0.1

    # Recommendation
    if primary_converged and relabeled_converged:
        status = "converged"
        recommendation = "Chains have converged. Safe to use posterior samples."
    elif primary_converged and not relabeled_converged:
        status = "partial"
        recommendation = (
            "Log-likelihood converged but component parameters show high R-hat. "
            "This may indicate residual label switching. Consider longer chains "
            "or use label-invariant summaries (e.g., mixture density, predictive)."
        )
    else:
        status = "not_converged"
        recommendation = (
            "Log-likelihood R-hat is high, indicating true non-convergence. "
            "Consider: (1) longer warmup/sampling, (2) stronger priors, "
            "(3) reparameterization, or (4) simpler model."
        )

    return {
        "status": status,
        "recommendation": recommendation,
        "standard": standard,
        "label_invariant": label_invariant,
        "relabeled": relabeled,
        "label_switching": switching,
        "summary": {
            "rhat_log_lik": label_invariant["rhat_log_lik"],
            "rhat_relabeled_max": relabeled["max_rhat"],
            "rhat_standard_max": standard["max_rhat"],
            "switching_rate": switching["switching_rate"],
            "primary_converged": primary_converged,
            "relabeled_converged": relabeled_converged,
        },
    }
