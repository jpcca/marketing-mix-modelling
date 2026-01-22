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
