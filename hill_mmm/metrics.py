"""Evaluation metrics for Hill Mixture MMM.

Key metrics:
- effective_k: Number of active mixture components
- parameter_recovery: Check if true params in credible intervals
- delta_loo: Relative improvement over baseline
"""

import numpy as np
from numpyro.infer import MCMC


def compute_effective_k(mcmc: MCMC, threshold: float = 0.05) -> dict[str, float]:
    """Compute effective number of mixture components.

    Counts components with mixture weight > threshold.
    This is the key metric for showing sparse Dirichlet works.

    Args:
        mcmc: Fitted MCMC object (must have 'pis' samples)
        threshold: Minimum weight to count as active

    Returns:
        Dict with mean, std, and per-sample effective K
    """
    samples = mcmc.get_samples()

    if "pis" not in samples:
        # Single Hill model has no mixture weights
        return {
            "effective_k_mean": 1.0,
            "effective_k_std": 0.0,
            "effective_k_samples": np.ones(1),
        }

    pis = np.array(samples["pis"])  # (n_samples, K)
    effective_k = (pis > threshold).sum(axis=-1)  # (n_samples,)

    return {
        "effective_k_mean": float(effective_k.mean()),
        "effective_k_std": float(effective_k.std()),
        "effective_k_samples": effective_k,
    }


def compute_parameter_recovery(mcmc: MCMC, meta: dict, ci_level: float = 0.95) -> dict[str, dict]:
    """Check if true parameters fall within credible intervals.

    For each recoverable parameter, reports:
    - true value
    - posterior mean
    - CI bounds
    - whether true is in CI

    Args:
        mcmc: Fitted MCMC object
        meta: DGP metadata with true parameter values
        ci_level: Credible interval level (default 95%)

    Returns:
        Dict mapping param names to recovery stats
    """
    samples = mcmc.get_samples()
    alpha = (1 - ci_level) / 2
    results = {}

    # Alpha (adstock decay)
    if "alpha" in samples and "alpha_true" in meta:
        alpha_samples = np.array(samples["alpha"])
        true_val = meta["alpha_true"]
        ci_low = np.percentile(alpha_samples, 100 * alpha)
        ci_high = np.percentile(alpha_samples, 100 * (1 - alpha))
        results["alpha"] = {
            "true": true_val,
            "mean": float(alpha_samples.mean()),
            "std": float(alpha_samples.std()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "in_ci": bool(ci_low <= true_val <= ci_high),
        }

    # Sigma (observation noise)
    if "sigma" in samples and "sigma_true" in meta:
        sigma_samples = np.array(samples["sigma"])
        true_val = meta["sigma_true"]
        ci_low = np.percentile(sigma_samples, 100 * alpha)
        ci_high = np.percentile(sigma_samples, 100 * (1 - alpha))
        results["sigma"] = {
            "true": true_val,
            "mean": float(sigma_samples.mean()),
            "std": float(sigma_samples.std()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "in_ci": bool(ci_low <= true_val <= ci_high),
        }

    # Intercept
    if "intercept" in samples and "intercept_true" in meta:
        int_samples = np.array(samples["intercept"])
        true_val = meta["intercept_true"]
        ci_low = np.percentile(int_samples, 100 * alpha)
        ci_high = np.percentile(int_samples, 100 * (1 - alpha))
        results["intercept"] = {
            "true": true_val,
            "mean": float(int_samples.mean()),
            "std": float(int_samples.std()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "in_ci": bool(ci_low <= true_val <= ci_high),
        }

    # For mixture models, check if any component matches true pis
    if "pis" in samples and "pi_true" in meta:
        pis_samples = np.array(samples["pis"])  # (n_samples, K)
        pis_mean = pis_samples.mean(axis=0)
        K_fit = pis_samples.shape[1]
        K_true = len(meta["pi_true"])

        results["pis"] = {
            "K_fit": K_fit,
            "K_true": K_true,
            "pis_mean": pis_mean.tolist(),
            "pi_true": meta["pi_true"].tolist(),
        }

    return results


def compute_delta_loo(loo_model: dict, loo_baseline: dict) -> dict[str, float]:
    """Compute improvement in LOO-CV relative to baseline.

    Positive delta means model is better than baseline.

    Args:
        loo_model: LOO results for model being evaluated
        loo_baseline: LOO results for baseline (single Hill)

    Returns:
        Dict with delta, se, and significance
    """
    if np.isnan(loo_model.get("elpd_loo", np.nan)) or np.isnan(
        loo_baseline.get("elpd_loo", np.nan)
    ):
        return {"delta_loo": np.nan, "se": np.nan, "significant": False}

    delta = loo_model["elpd_loo"] - loo_baseline["elpd_loo"]
    # Approximate SE (conservative)
    se = np.sqrt(loo_model["se"] ** 2 + loo_baseline["se"] ** 2)

    return {
        "delta_loo": float(delta),
        "se": float(se),
        "significant": bool(abs(delta) > 2 * se),  # Roughly 95% CI
    }


def summarize_results(results: dict) -> str:
    """Format benchmark results as a summary table.

    Args:
        results: Dict with evaluation metrics

    Returns:
        Formatted string for printing
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"DGP: {results.get('dgp', 'unknown')}")
    lines.append(f"Model: {results.get('model', 'unknown')}")
    lines.append("=" * 70)

    # Convergence
    conv = results.get("convergence", {})
    lines.append(
        f"Convergence: R-hat={conv.get('max_rhat', np.nan):.3f}, "
        f"ESS={conv.get('min_ess_bulk', np.nan):.0f}"
    )

    # Model comparison
    lines.append(
        f"LOO-CV: {results.get('elpd_loo', np.nan):.1f} (SE={results.get('loo_se', np.nan):.1f})"
    )
    lines.append(
        f"WAIC: {results.get('elpd_waic', np.nan):.1f} (SE={results.get('waic_se', np.nan):.1f})"
    )

    # Predictive
    lines.append(
        f"Train RMSE: {results.get('train_rmse', np.nan):.3f}, "
        f"Test RMSE: {results.get('test_rmse', np.nan):.3f}"
    )
    lines.append(f"90% Coverage: {results.get('coverage_90', np.nan):.1%}")

    # Effective K
    eff_k = results.get("effective_k_mean", np.nan)
    lines.append(f"Effective K: {eff_k:.2f}")

    # Delta LOO
    delta = results.get("delta_loo", np.nan)
    if not np.isnan(delta):
        sig = "*" if results.get("delta_significant", False) else ""
        lines.append(f"Delta LOO vs baseline: {delta:+.1f}{sig}")

    return "\n".join(lines)
