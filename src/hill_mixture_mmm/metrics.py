"""Evaluation metrics for Hill Mixture MMM.

Key metrics:
- effective_k: Number of active mixture components
- parameter_recovery: Check if true params in credible intervals
- latent_recovery: Check recovery of the true latent mean function
- delta_loo: Relative improvement over baseline
- component_recovery: Permutation-invariant component recovery on synthetic data
- across_seed_stability: Seed-to-seed stability of recovered component structure
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb
from typing import Any

import numpy as np
from numpyro.infer import MCMC


def compute_effective_k(mcmc: MCMC, threshold: float = 0.05) -> dict[str, float]:
    """Compute effective number of mixture components.

    Counts components with mixture weight > threshold.

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

    if "slope" in samples and "slope_true" in meta:
        slope_samples = np.array(samples["slope"])
        true_val = meta["slope_true"]
        ci_low = np.percentile(slope_samples, 100 * alpha)
        ci_high = np.percentile(slope_samples, 100 * (1 - alpha))
        results["slope"] = {
            "true": true_val,
            "mean": float(slope_samples.mean()),
            "std": float(slope_samples.std()),
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


def compute_latent_recovery(mu_true: np.ndarray, mu_samples: np.ndarray) -> dict[str, float]:
    """Measure recovery of the noise-free latent mean function.

    Args:
        mu_true: (T,) true latent mean from the DGP
        mu_samples: (n_samples, T) posterior samples of the latent mean

    Returns:
        Dict with MAPE (percentage points), MAE, and 90% interval coverage
    """
    mu_true = np.asarray(mu_true, dtype=np.float64)
    mu_samples = np.asarray(mu_samples, dtype=np.float64)

    if mu_samples.ndim != 2:
        raise ValueError("mu_samples must have shape (n_samples, T)")
    if mu_true.shape[0] != mu_samples.shape[1]:
        raise ValueError("mu_true and mu_samples must align on time dimension")

    mu_mean = mu_samples.mean(axis=0)
    q05 = np.quantile(mu_samples, 0.05, axis=0)
    q95 = np.quantile(mu_samples, 0.95, axis=0)
    denom = np.maximum(np.abs(mu_true), 1e-8)

    return {
        "mape": float(np.mean(np.abs((mu_mean - mu_true) / denom)) * 100.0),
        "mae": float(np.mean(np.abs(mu_mean - mu_true))),
        "coverage_90": float(np.mean((mu_true >= q05) & (mu_true <= q95))),
    }


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


def summarize_component_posterior(
    samples: dict[str, np.ndarray],
    *,
    scale_reference: float = 1.0,
    weight_threshold: float = 0.05,
) -> dict[str, Any] | None:
    """Summarize posterior component parameters for recovery and stability checks."""
    if not {"A", "k", "n"}.issubset(samples):
        return None

    scale_reference = float(max(scale_reference, 1e-6))
    A = np.asarray(samples["A"], dtype=np.float64)
    k = np.asarray(samples["k"], dtype=np.float64)
    n = np.asarray(samples["n"], dtype=np.float64)

    if A.ndim == 1:
        A = A[:, None]
    if k.ndim == 1:
        k = k[:, None]
    if n.ndim == 1:
        n = n[:, None]

    if "pis" in samples:
        pis = np.asarray(samples["pis"], dtype=np.float64)
        if pis.ndim == 1:
            pis = pis[:, None]
    else:
        pis = np.ones((A.shape[0], 1), dtype=np.float64)

    components = []
    for idx in range(A.shape[1]):
        pi_mean = float(pis[:, idx].mean())
        k_mean = float(k[:, idx].mean())
        components.append(
            {
                "index": idx,
                "A_mean": float(A[:, idx].mean()),
                "A_std": float(A[:, idx].std()),
                "k_mean": k_mean,
                "k_std": float(k[:, idx].std()),
                "k_ratio_mean": float(k_mean / scale_reference),
                "k_ratio_std": float(k[:, idx].std() / scale_reference),
                "n_mean": float(n[:, idx].mean()),
                "n_std": float(n[:, idx].std()),
                "pi_mean": pi_mean,
                "pi_std": float(pis[:, idx].std()),
                "active": bool(pi_mean > weight_threshold),
            }
        )

    return {
        "K_total": len(components),
        "K_active": int(sum(component["active"] for component in components)),
        "weight_threshold": float(weight_threshold),
        "scale_reference": scale_reference,
        "components": components,
    }


def _summarize_true_components(
    meta: dict[str, Any], weight_threshold: float = 0.05
) -> dict[str, Any]:
    """Build a component summary from synthetic DGP metadata."""
    scale_reference = float(max(meta.get("s_median", 1.0), 1e-6))
    A_true = np.asarray(meta["A_true"], dtype=np.float64)
    k_true = np.asarray(meta["k_true"], dtype=np.float64)
    n_true = np.asarray(meta["n_true"], dtype=np.float64)
    pi_true = np.asarray(meta["pi_true"], dtype=np.float64)

    components = []
    for idx, (A_i, k_i, n_i, pi_i) in enumerate(zip(A_true, k_true, n_true, pi_true, strict=True)):
        components.append(
            {
                "index": idx,
                "A_mean": float(A_i),
                "A_std": 0.0,
                "k_mean": float(k_i),
                "k_std": 0.0,
                "k_ratio_mean": float(k_i / scale_reference),
                "k_ratio_std": 0.0,
                "n_mean": float(n_i),
                "n_std": 0.0,
                "pi_mean": float(pi_i),
                "pi_std": 0.0,
                "active": bool(pi_i > weight_threshold),
            }
        )

    return {
        "K_total": len(components),
        "K_active": int(sum(component["active"] for component in components)),
        "weight_threshold": float(weight_threshold),
        "scale_reference": scale_reference,
        "components": components,
    }


def _extract_component_summary(payload: dict[str, Any]) -> tuple[dict[str, Any], str, int | None]:
    """Normalize a case summary or component summary payload."""
    if "component_summary" in payload:
        component_summary = payload["component_summary"]
        label = str(payload.get("label", "case"))
        seed = int(payload["seed"]) if "seed" in payload else None
        return component_summary, label, seed

    if "components" in payload and "K_total" in payload:
        label = str(payload.get("label", "case"))
        seed = int(payload["seed"]) if "seed" in payload else None
        return payload, label, seed

    raise ValueError("payload must contain either component_summary or a component summary itself")


def _active_components(component_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active components, falling back to the highest-weight component if needed."""
    components = [dict(component) for component in component_summary.get("components", [])]
    active = [component for component in components if component.get("active", False)]
    if active:
        return active
    if not components:
        return []
    return [max(components, key=lambda component: float(component.get("pi_mean", 0.0)))]


def _component_curve(component: dict[str, Any], u_grid: np.ndarray) -> np.ndarray:
    """Return the normalized Hill effect curve for one component."""
    A_mean = float(component["A_mean"])
    k_ratio = float(max(component["k_ratio_mean"], 1e-6))
    n_mean = float(max(component["n_mean"], 1e-6))
    numerator = np.power(u_grid, n_mean)
    denominator = np.power(k_ratio, n_mean) + numerator + 1e-12
    return A_mean * numerator / denominator


def _pair_metrics(
    reference_component: dict[str, Any],
    candidate_component: dict[str, Any],
    u_grid: np.ndarray,
) -> dict[str, float]:
    """Compute component-level recovery metrics for one matched pair."""
    reference_curve = _component_curve(reference_component, u_grid)
    candidate_curve = _component_curve(candidate_component, u_grid)
    curve_rmse = float(np.sqrt(np.mean((reference_curve - candidate_curve) ** 2)))
    curve_scale = max(
        float(np.max(np.abs(reference_curve))),
        float(np.max(np.abs(candidate_curve))),
        1.0,
    )
    avg_weight = 0.5 * (
        float(reference_component["pi_mean"]) + float(candidate_component["pi_mean"])
    )
    return {
        "curve_rmse": curve_rmse,
        "curve_nrmse": float(curve_rmse / curve_scale),
        "pi_abs_error": float(
            abs(float(reference_component["pi_mean"]) - float(candidate_component["pi_mean"]))
        ),
        "A_rel_error": float(
            abs(float(reference_component["A_mean"]) - float(candidate_component["A_mean"]))
            / max(abs(float(reference_component["A_mean"])), 1e-6)
        ),
        "k_ratio_rel_error": float(
            abs(
                float(reference_component["k_ratio_mean"])
                - float(candidate_component["k_ratio_mean"])
            )
            / max(abs(float(reference_component["k_ratio_mean"])), 1e-6)
        ),
        "n_abs_error": float(
            abs(float(reference_component["n_mean"]) - float(candidate_component["n_mean"]))
        ),
        "avg_weight": float(avg_weight),
    }


def compute_component_set_alignment(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Align two component sets with a permutation-invariant matching rule."""
    reference_summary, reference_label, reference_seed = _extract_component_summary(reference)
    candidate_summary, candidate_label, candidate_seed = _extract_component_summary(candidate)
    reference_components = _active_components(reference_summary)
    candidate_components = _active_components(candidate_summary)

    u_grid = np.linspace(0.0, curve_grid_max, grid_size, dtype=np.float64)
    reference_weight_total = float(sum(component["pi_mean"] for component in reference_components))
    candidate_weight_total = float(sum(component["pi_mean"] for component in candidate_components))

    if not reference_components and not candidate_components:
        return {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_seed": reference_seed,
            "candidate_seed": candidate_seed,
            "reference_active_k": 0,
            "candidate_active_k": 0,
            "matched_components": [],
            "matched_weight": 0.0,
            "unmatched_reference_weight": 0.0,
            "unmatched_candidate_weight": 0.0,
            "weighted_curve_rmse": 0.0,
            "weighted_curve_nrmse": 0.0,
            "mean_curve_rmse": 0.0,
            "mean_curve_nrmse": 0.0,
            "weighted_pi_abs_error": 0.0,
            "mean_pi_abs_error": 0.0,
            "mean_A_rel_error": 0.0,
            "mean_k_ratio_rel_error": 0.0,
            "mean_n_abs_error": 0.0,
            "assignment_cost": 0.0,
        }

    if len(reference_components) <= len(candidate_components):
        assignment_iter = (
            list(zip(range(len(reference_components)), candidate_order, strict=True))
            for candidate_choice in combinations(
                range(len(candidate_components)), len(reference_components)
            )
            for candidate_order in permutations(candidate_choice)
        )
    else:
        assignment_iter = (
            list(zip(reference_order, range(len(candidate_components)), strict=True))
            for reference_choice in combinations(
                range(len(reference_components)), len(candidate_components)
            )
            for reference_order in permutations(reference_choice)
        )

    best_result: dict[str, Any] | None = None
    for pairs in assignment_iter:
        matched_components = []
        weighted_curve_rmse = 0.0
        weighted_curve_nrmse = 0.0
        weighted_pi_abs_error = 0.0
        mean_curve_rmse = []
        mean_curve_nrmse = []
        mean_pi_abs_error = []
        mean_A_rel_error = []
        mean_k_ratio_rel_error = []
        mean_n_abs_error = []
        matched_reference_weight = 0.0
        matched_candidate_weight = 0.0
        matched_avg_weight = 0.0

        for reference_idx, candidate_idx in pairs:
            reference_component = reference_components[reference_idx]
            candidate_component = candidate_components[candidate_idx]
            metrics = _pair_metrics(reference_component, candidate_component, u_grid)
            matched_components.append(
                {
                    "reference_component_index": int(reference_component["index"]),
                    "candidate_component_index": int(candidate_component["index"]),
                    **metrics,
                }
            )
            matched_reference_weight += float(reference_component["pi_mean"])
            matched_candidate_weight += float(candidate_component["pi_mean"])
            matched_avg_weight += metrics["avg_weight"]
            weighted_curve_rmse += metrics["avg_weight"] * metrics["curve_rmse"]
            weighted_curve_nrmse += metrics["avg_weight"] * metrics["curve_nrmse"]
            weighted_pi_abs_error += metrics["avg_weight"] * metrics["pi_abs_error"]
            mean_curve_rmse.append(metrics["curve_rmse"])
            mean_curve_nrmse.append(metrics["curve_nrmse"])
            mean_pi_abs_error.append(metrics["pi_abs_error"])
            mean_A_rel_error.append(metrics["A_rel_error"])
            mean_k_ratio_rel_error.append(metrics["k_ratio_rel_error"])
            mean_n_abs_error.append(metrics["n_abs_error"])

        unmatched_reference_weight = max(reference_weight_total - matched_reference_weight, 0.0)
        unmatched_candidate_weight = max(candidate_weight_total - matched_candidate_weight, 0.0)
        cost = weighted_curve_nrmse + unmatched_reference_weight + unmatched_candidate_weight

        result = {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_seed": reference_seed,
            "candidate_seed": candidate_seed,
            "reference_active_k": len(reference_components),
            "candidate_active_k": len(candidate_components),
            "matched_components": matched_components,
            "matched_weight": float(matched_avg_weight),
            "unmatched_reference_weight": float(unmatched_reference_weight),
            "unmatched_candidate_weight": float(unmatched_candidate_weight),
            "weighted_curve_rmse": float(weighted_curve_rmse / max(matched_avg_weight, 1e-6)),
            "weighted_curve_nrmse": float(weighted_curve_nrmse / max(matched_avg_weight, 1e-6)),
            "mean_curve_rmse": float(np.mean(mean_curve_rmse)) if mean_curve_rmse else 0.0,
            "mean_curve_nrmse": float(np.mean(mean_curve_nrmse)) if mean_curve_nrmse else 0.0,
            "weighted_pi_abs_error": float(weighted_pi_abs_error / max(matched_avg_weight, 1e-6)),
            "mean_pi_abs_error": float(np.mean(mean_pi_abs_error)) if mean_pi_abs_error else 0.0,
            "mean_A_rel_error": float(np.mean(mean_A_rel_error)) if mean_A_rel_error else 0.0,
            "mean_k_ratio_rel_error": float(np.mean(mean_k_ratio_rel_error))
            if mean_k_ratio_rel_error
            else 0.0,
            "mean_n_abs_error": float(np.mean(mean_n_abs_error)) if mean_n_abs_error else 0.0,
            "assignment_cost": float(cost),
        }
        if best_result is None or result["assignment_cost"] < best_result["assignment_cost"]:
            best_result = result

    if best_result is None:
        # One side has active components but the other has none —
        # no valid assignment exists.  Return a zero-metric result.
        return {
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_seed": reference_seed,
            "candidate_seed": candidate_seed,
            "reference_active_k": len(reference_components),
            "candidate_active_k": len(candidate_components),
            "matched_components": [],
            "matched_weight": 0.0,
            "unmatched_reference_weight": float(reference_weight_total),
            "unmatched_candidate_weight": float(candidate_weight_total),
            "weighted_curve_rmse": 0.0,
            "weighted_curve_nrmse": 0.0,
            "mean_curve_rmse": 0.0,
            "mean_curve_nrmse": 0.0,
            "weighted_pi_abs_error": 0.0,
            "mean_pi_abs_error": 0.0,
            "mean_A_rel_error": 0.0,
            "mean_k_ratio_rel_error": 0.0,
            "mean_n_abs_error": 0.0,
            "assignment_cost": float(reference_weight_total + candidate_weight_total),
        }
    return best_result


def compute_permutation_invariant_component_recovery(
    samples: dict[str, np.ndarray],
    meta: dict[str, Any],
    *,
    weight_threshold: float = 0.05,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any] | None:
    """Compare recovered mixture components to the true DGP up to permutation."""
    required = {"A_true", "k_true", "n_true", "pi_true"}
    if not required.issubset(meta):
        return None

    posterior_summary = summarize_component_posterior(
        samples,
        scale_reference=float(meta.get("s_median", 1.0)),
        weight_threshold=weight_threshold,
    )
    if posterior_summary is None:
        return None

    true_summary = _summarize_true_components(meta, weight_threshold=weight_threshold)
    alignment = compute_component_set_alignment(
        {"component_summary": true_summary, "label": "true_components"},
        {"component_summary": posterior_summary, "label": "posterior_components"},
        curve_grid_max=curve_grid_max,
        grid_size=grid_size,
    )
    return {
        "K_true": int(true_summary["K_total"]),
        "K_true_active": int(true_summary["K_active"]),
        "K_posterior": int(posterior_summary["K_total"]),
        "K_posterior_active": int(posterior_summary["K_active"]),
        "effective_k_error": float(abs(posterior_summary["K_active"] - true_summary["K_active"])),
        **alignment,
    }


def compute_across_seed_component_stability(
    summaries: list[dict[str, Any]],
    *,
    curve_grid_max: float = 4.0,
    grid_size: int = 128,
) -> dict[str, Any]:
    """Measure how stably a model recovers components across benchmark seeds."""
    if len(summaries) < 2:
        raise ValueError("need at least two summaries to evaluate across-seed stability")

    normalized = []
    for summary in summaries:
        component_summary, label, seed = _extract_component_summary(summary)
        normalized.append(
            {
                "label": label,
                "seed": seed,
                "component_summary": component_summary,
            }
        )

    pairwise = []
    for left, right in combinations(normalized, 2):
        pairwise.append(
            compute_component_set_alignment(
                left,
                right,
                curve_grid_max=curve_grid_max,
                grid_size=grid_size,
            )
        )

    active_ks = np.asarray(
        [entry["component_summary"]["K_active"] for entry in normalized],
        dtype=np.float64,
    )
    active_k_values, active_k_counts = np.unique(active_ks, return_counts=True)
    mode_idx = int(np.argmax(active_k_counts))

    def _aggregate(name: str) -> dict[str, float]:
        values = np.asarray([pair[name] for pair in pairwise], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
        }

    unmatched_totals = np.asarray(
        [
            pair["unmatched_reference_weight"] + pair["unmatched_candidate_weight"]
            for pair in pairwise
        ],
        dtype=np.float64,
    )

    return {
        "num_seeds": len(normalized),
        "expected_pair_count": comb(len(normalized), 2),
        "pair_count": len(pairwise),
        "seed_labels": [entry["label"] for entry in normalized],
        "seeds": [entry["seed"] for entry in normalized],
        "mode_active_k": int(active_k_values[mode_idx]),
        "active_k_mean": float(active_ks.mean()),
        "active_k_std": float(active_ks.std()),
        "active_k_consistency": float(active_k_counts[mode_idx] / len(active_ks)),
        "weighted_curve_nrmse": _aggregate("weighted_curve_nrmse"),
        "weighted_curve_rmse": _aggregate("weighted_curve_rmse"),
        "weighted_pi_abs_error": _aggregate("weighted_pi_abs_error"),
        "mean_k_ratio_rel_error": _aggregate("mean_k_ratio_rel_error"),
        "mean_A_rel_error": _aggregate("mean_A_rel_error"),
        "mean_n_abs_error": _aggregate("mean_n_abs_error"),
        "unmatched_total_weight": {
            "mean": float(unmatched_totals.mean()),
            "std": float(unmatched_totals.std()),
            "max": float(unmatched_totals.max()),
        },
        "pairwise": pairwise,
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
        f"Train MAPE: {results.get('train_mape', np.nan):.3f}%, "
        f"Test MAPE: {results.get('test_mape', np.nan):.3f}%"
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
