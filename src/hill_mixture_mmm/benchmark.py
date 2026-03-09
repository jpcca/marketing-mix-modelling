"""Reusable benchmark helpers for test-driven model evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baseline import standardized_time_index
from .data import DGPConfig, compute_prior_config, generate_data
from .data_loader import TimeSeriesConfig, load_timeseries
from .inference import (
    compute_comprehensive_mixture_diagnostics,
    compute_convergence_diagnostics,
    compute_loo,
    compute_predictions,
    compute_predictive_metrics,
    compute_waic,
    relabel_samples_by_k,
    run_inference,
)
from .metrics import (
    compute_delta_loo,
    compute_effective_k,
    compute_latent_recovery,
    compute_parameter_recovery,
)
from .models import model_hill_mixture_hierarchical_reparam, model_single_hill
from .transforms import hill


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a benchmarked model."""

    name: str
    fn: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_SPECS: dict[str, ModelSpec] = {
    "single_hill": ModelSpec("single_hill", model_single_hill, {}),
    "mixture_k2": ModelSpec("mixture_k2", model_hill_mixture_hierarchical_reparam, {"K": 2}),
    "mixture_k3": ModelSpec("mixture_k3", model_hill_mixture_hierarchical_reparam, {"K": 3}),
    "mixture_k5": ModelSpec("mixture_k5", model_hill_mixture_hierarchical_reparam, {"K": 5}),
}


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Inference and split settings for a benchmark case."""

    seed: int = 42
    train_ratio: float = 0.75
    num_warmup: int = 300
    num_samples: int = 300
    num_chains: int = 2
    target_accept_prob: float = 0.90
    progress_bar: bool = False
    allow_mixture_retries: bool = True


@dataclass(frozen=True)
class BenchmarkThresholds:
    """Pass/fail thresholds for a single benchmark case."""

    max_rhat: float | None = 1.05
    min_ess_bulk: float | None = 80.0
    max_label_invariant_rhat: float | None = 1.01
    min_test_coverage_90: float = 0.80
    max_test_rmse: float | None = None
    max_test_mu_rmse: float | None = None
    min_test_mu_coverage_90: float | None = None
    require_alpha_in_ci: bool = False
    require_sigma_in_ci: bool = False
    effective_k_bounds: tuple[float, float] | None = None
    max_pareto_k_bad: int | None = None
    max_pareto_k_very_bad: int | None = None


@dataclass(frozen=True)
class ComparisonThresholds:
    """Pass/fail thresholds for comparing two benchmark cases."""

    min_delta_loo: float | None = None
    max_delta_rmse: float | None = None
    max_candidate_rmse_ratio: float | None = None


@dataclass
class BenchmarkCaseResult:
    """Structured benchmark output for one model on one dataset."""

    label: str
    domain: str
    dataset_name: str
    model_name: str
    seed: int
    train_ratio: float
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    dates_train: np.ndarray | None
    dates_test: np.ndarray | None
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    loo: dict[str, Any]
    waic: dict[str, Any]
    convergence: dict[str, Any]
    label_invariant: dict[str, Any] | None
    relabeled: dict[str, Any] | None
    label_switching: dict[str, Any] | None
    converged: bool
    effective_k: dict[str, Any]
    parameter_recovery: dict[str, Any] | None
    latent_train: dict[str, float] | None
    latent_test: dict[str, float] | None
    meta: dict[str, Any] | None
    samples: dict[str, np.ndarray]
    fit_summary: dict[str, Any]


def get_model_spec(model_name: str) -> ModelSpec:
    """Return benchmark model specification."""
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc


def _build_retry_schedule(config: BenchmarkRunConfig, is_mixture_model: bool) -> list[dict[str, Any]]:
    """Build a deterministic retry schedule for mixture fits."""
    base_attempt = [
        {
            "seed_offset": 0,
            "num_warmup": config.num_warmup,
            "num_samples": config.num_samples,
            "target_accept_prob": config.target_accept_prob,
        }
    ]
    if not is_mixture_model or not config.allow_mixture_retries:
        return base_attempt
    return base_attempt + [
        {
            "seed_offset": 1000,
            "num_warmup": max(config.num_warmup, 500),
            "num_samples": max(config.num_samples, 500),
            "target_accept_prob": max(config.target_accept_prob, 0.95),
        },
        {
            "seed_offset": 2000,
            "num_warmup": max(config.num_warmup, 800),
            "num_samples": max(config.num_samples, 800),
            "target_accept_prob": max(config.target_accept_prob, 0.97),
        },
    ]


def _is_effectively_converged(
    convergence: dict[str, Any],
    label_invariant: dict[str, Any] | None,
    label_threshold: float = 1.01,
) -> bool:
    """Evaluate convergence with label-invariant diagnostics when available."""
    if bool(convergence["converged"]):
        return True
    if label_invariant is None:
        return False
    return bool(label_invariant.get("rhat_log_lik", np.inf) <= label_threshold)


def _extract_latent_samples(predictions: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return latent mean samples from posterior predictive output."""
    if "mu_expected" in predictions:
        return np.asarray(predictions["mu_expected"], dtype=np.float32)
    if "mu" in predictions:
        return np.asarray(predictions["mu"], dtype=np.float32)
    return None


def _fit_case(
    model_spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    t_std_train: np.ndarray,
    prior_config: dict[str, Any],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    """Run inference with optional retries and return diagnostics."""
    numpyro.set_host_device_count(max(1, config.num_chains))

    is_mixture_model = "K" in model_spec.kwargs
    retry_schedule = _build_retry_schedule(config, is_mixture_model)

    mcmc = None
    convergence = None
    label_invariant = None
    relabeled = None
    label_switching = None
    effective_convergence = False
    fit_summary = {
        "retry_attempt": 0,
        "inference_seed": config.seed,
        "num_warmup_used": config.num_warmup,
        "num_samples_used": config.num_samples,
        "target_accept_prob_used": config.target_accept_prob,
    }

    for attempt_idx, attempt in enumerate(retry_schedule):
        fit_summary = {
            "retry_attempt": attempt_idx,
            "inference_seed": config.seed + int(attempt["seed_offset"]),
            "num_warmup_used": int(attempt["num_warmup"]),
            "num_samples_used": int(attempt["num_samples"]),
            "target_accept_prob_used": float(attempt["target_accept_prob"]),
        }

        mcmc = run_inference(
            model_spec.fn,
            x_train,
            y_train,
            seed=fit_summary["inference_seed"],
            num_warmup=fit_summary["num_warmup_used"],
            num_samples=fit_summary["num_samples_used"],
            num_chains=config.num_chains,
            prior_config=prior_config,
            t_std=t_std_train,
            target_accept_prob=fit_summary["target_accept_prob_used"],
            progress_bar=config.progress_bar,
            **model_spec.kwargs,
        )
        if is_mixture_model:
            diagnostics = compute_comprehensive_mixture_diagnostics(mcmc, x_train, y_train)
            convergence = diagnostics["standard"]
            label_invariant = diagnostics["label_invariant"]
            relabeled = diagnostics["relabeled"]
            label_switching = diagnostics["label_switching"]
        else:
            convergence = compute_convergence_diagnostics(mcmc)
            label_invariant = None
            relabeled = None
            label_switching = None

        effective_convergence = _is_effectively_converged(convergence, label_invariant)
        if effective_convergence:
            break

    assert mcmc is not None
    assert convergence is not None

    return {
        "mcmc": mcmc,
        "convergence": convergence,
        "label_invariant": label_invariant,
        "relabeled": relabeled,
        "label_switching": label_switching,
        "converged": effective_convergence,
        "fit_summary": fit_summary,
    }


def _run_case_from_series(
    *,
    label: str,
    domain: str,
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | None,
    model_spec: ModelSpec,
    config: BenchmarkRunConfig,
    meta: dict[str, Any] | None = None,
) -> BenchmarkCaseResult:
    """Run one benchmark case from a prepared single-series dataset."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    T = len(y)
    T_train = int(T * config.train_ratio)
    if T_train <= 0 or T_train >= T:
        raise ValueError("train_ratio must leave at least one train and one test observation")

    x_train, y_train = x[:T_train], y[:T_train]
    x_test, y_test = x[T_train:], y[T_train:]
    dates_train = None if dates is None else np.asarray(dates[:T_train])
    dates_test = None if dates is None else np.asarray(dates[T_train:])

    prior_config = compute_prior_config(x_train, y_train)
    t_std_train = np.asarray(standardized_time_index(T)[:T_train], dtype=np.float32)

    fit = _fit_case(model_spec, x_train, y_train, t_std_train, prior_config, config)
    mcmc = fit["mcmc"]

    pred_train = compute_predictions(
        mcmc,
        model_spec.fn,
        x_train,
        prior_config=prior_config,
        total_time=T,
        **model_spec.kwargs,
    )
    pred_test = compute_predictions(
        mcmc,
        model_spec.fn,
        x_test,
        prior_config=prior_config,
        history_x=x_train,
        total_time=T,
        **model_spec.kwargs,
    )

    train_metrics = compute_predictive_metrics(y_train, pred_train["y"])
    test_metrics = compute_predictive_metrics(y_test, pred_test["y"])

    latent_train = None
    latent_test = None
    if meta is not None and "mu_true" in meta:
        train_mu = _extract_latent_samples(pred_train)
        test_mu = _extract_latent_samples(pred_test)
        if train_mu is not None and test_mu is not None:
            latent_train = compute_latent_recovery(meta["mu_true"][:T_train], train_mu)
            latent_test = compute_latent_recovery(meta["mu_true"][T_train:], test_mu)

    parameter_recovery = None
    if meta is not None:
        parameter_recovery = compute_parameter_recovery(mcmc, meta)

    samples = mcmc.get_samples()
    if "pis" in samples and np.asarray(samples["pis"]).ndim == 2:
        samples = relabel_samples_by_k(samples)

    return BenchmarkCaseResult(
        label=label,
        domain=domain,
        dataset_name=dataset_name,
        model_name=model_spec.name,
        seed=config.seed,
        train_ratio=config.train_ratio,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        dates_train=dates_train,
        dates_test=dates_test,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        loo=compute_loo(mcmc),
        waic=compute_waic(mcmc),
        convergence=fit["convergence"],
        label_invariant=fit["label_invariant"],
        relabeled=fit["relabeled"],
        label_switching=fit["label_switching"],
        converged=fit["converged"],
        effective_k=compute_effective_k(mcmc),
        parameter_recovery=parameter_recovery,
        latent_train=latent_train,
        latent_test=latent_test,
        meta=meta,
        samples={k: np.asarray(v) for k, v in samples.items()},
        fit_summary=fit["fit_summary"],
    )


def run_synthetic_benchmark_case(
    *,
    dgp_config: DGPConfig,
    model_name: str,
    config: BenchmarkRunConfig,
    label: str | None = None,
) -> BenchmarkCaseResult:
    """Run one synthetic benchmark case."""
    x, y, meta = generate_data(dgp_config)
    case_label = label or f"synthetic_{dgp_config.dgp_type}_{model_name}"
    return _run_case_from_series(
        label=case_label,
        domain="synthetic",
        dataset_name=dgp_config.dgp_type,
        x=x,
        y=y,
        dates=None,
        model_spec=get_model_spec(model_name),
        config=config,
        meta=meta,
    )


def run_real_benchmark_case(
    *,
    csv_path: str | Path,
    timeseries_config: TimeSeriesConfig,
    model_name: str,
    config: BenchmarkRunConfig,
    label: str | None = None,
) -> BenchmarkCaseResult:
    """Run one real-data benchmark case."""
    loaded = load_timeseries(csv_path, timeseries_config)
    case_label = label or f"real_{timeseries_config.organisation_id}_{model_name}"
    return _run_case_from_series(
        label=case_label,
        domain="real",
        dataset_name=str(timeseries_config.organisation_id),
        x=loaded.x,
        y=loaded.y,
        dates=loaded.dates,
        model_spec=get_model_spec(model_name),
        config=config,
        meta=loaded.meta,
    )


def case_summary(result: BenchmarkCaseResult) -> dict[str, Any]:
    """Return a compact JSON-serializable summary."""
    summary = {
        "label": result.label,
        "domain": result.domain,
        "dataset_name": result.dataset_name,
        "model_name": result.model_name,
        "seed": result.seed,
        "train_size": int(len(result.y_train)),
        "test_size": int(len(result.y_test)),
        "converged": bool(result.converged),
        "convergence": {
            "max_rhat": float(result.convergence["max_rhat"]),
            "min_ess_bulk": float(result.convergence["min_ess_bulk"]),
        },
        "loo": {
            "elpd_loo": float(result.loo.get("elpd_loo", np.nan)),
            "se": float(result.loo.get("se", np.nan)),
            "pareto_k_bad": int(result.loo.get("pareto_k_bad", 0)),
            "pareto_k_very_bad": int(result.loo.get("pareto_k_very_bad", 0)),
        },
        "waic": {
            "elpd_waic": float(result.waic.get("elpd_waic", np.nan)),
            "se": float(result.waic.get("se", np.nan)),
        },
        "train_metrics": {
            "rmse": float(result.train_metrics["rmse"]),
            "coverage_90": float(result.train_metrics["coverage_90"]),
        },
        "test_metrics": {
            "rmse": float(result.test_metrics["rmse"]),
            "coverage_90": float(result.test_metrics["coverage_90"]),
        },
        "effective_k": {
            "mean": float(result.effective_k["effective_k_mean"]),
            "std": float(result.effective_k["effective_k_std"]),
        },
        "fit_summary": dict(result.fit_summary),
    }
    if result.label_invariant is not None:
        summary["label_invariant"] = {
            "rhat_log_lik": float(result.label_invariant["rhat_log_lik"]),
            "threshold": float(result.label_invariant["threshold"]),
        }
    if result.relabeled is not None:
        summary["relabeled"] = {
            "max_rhat": float(result.relabeled["max_rhat"]),
            "threshold": float(result.relabeled["threshold"]),
            "component_rhats": {
                name: {
                    "max": float(metrics["max"]),
                    "per_component": [float(value) for value in metrics["per_component"]],
                }
                for name, metrics in result.relabeled["component_rhats"].items()
            },
        }
    if result.label_switching is not None:
        summary["label_switching"] = {
            "switching_rate": float(result.label_switching["switching_rate"]),
            "n_unique_orderings": int(result.label_switching["n_unique_orderings"]),
            "mode_ordering": [int(value) for value in result.label_switching["mode_ordering"]],
            "mode_count": int(result.label_switching["mode_count"]),
            "top_orderings": [
                {
                    "ordering": [int(value) for value in ordering],
                    "count": int(count),
                }
                for ordering, count in result.label_switching["top_orderings"]
            ],
        }
    if result.parameter_recovery is not None:
        summary["parameter_recovery"] = {
            name: {
                key: value
                for key, value in metrics.items()
                if key in {"true", "mean", "ci_low", "ci_high", "in_ci"}
            }
            for name, metrics in result.parameter_recovery.items()
            if isinstance(metrics, dict)
        }
    if result.latent_train is not None:
        summary["latent_train"] = dict(result.latent_train)
    if result.latent_test is not None:
        summary["latent_test"] = dict(result.latent_test)
    return summary


def compare_case_results(
    baseline: BenchmarkCaseResult,
    candidate: BenchmarkCaseResult,
) -> dict[str, float]:
    """Compute comparison metrics between two benchmark cases."""
    delta = compute_delta_loo(candidate.loo, baseline.loo)
    return {
        "delta_loo": float(delta["delta_loo"]),
        "delta_loo_se": float(delta["se"]),
        "delta_loo_significant": float(bool(delta["significant"])),
        "delta_test_rmse": float(candidate.test_metrics["rmse"] - baseline.test_metrics["rmse"]),
        "candidate_rmse_ratio": float(candidate.test_metrics["rmse"] / baseline.test_metrics["rmse"]),
        "delta_test_coverage_90": float(
            candidate.test_metrics["coverage_90"] - baseline.test_metrics["coverage_90"]
        ),
    }


def assert_case_passes(result: BenchmarkCaseResult, thresholds: BenchmarkThresholds) -> None:
    """Raise an AssertionError if a benchmark case fails its thresholds."""
    errors: list[str] = []

    if not result.converged:
        errors.append("effective convergence check failed")
    if thresholds.max_rhat is not None and float(result.convergence["max_rhat"]) > thresholds.max_rhat:
        errors.append(
            f"max_rhat={result.convergence['max_rhat']:.3f} exceeds {thresholds.max_rhat:.3f}"
        )
    if (
        thresholds.min_ess_bulk is not None
        and float(result.convergence["min_ess_bulk"]) < thresholds.min_ess_bulk
    ):
        errors.append(
            f"min_ess_bulk={result.convergence['min_ess_bulk']:.1f} is below {thresholds.min_ess_bulk:.1f}"
        )
    if thresholds.max_label_invariant_rhat is not None and result.label_invariant is not None:
        rhat_log_lik = float(result.label_invariant["rhat_log_lik"])
        if rhat_log_lik > thresholds.max_label_invariant_rhat:
            errors.append(
                f"rhat_log_lik={rhat_log_lik:.3f} exceeds {thresholds.max_label_invariant_rhat:.3f}"
            )

    loo_value = float(result.loo.get("elpd_loo", np.nan))
    waic_value = float(result.waic.get("elpd_waic", np.nan))
    if not np.isfinite(loo_value):
        errors.append("LOO is not finite")
    if not np.isfinite(waic_value):
        errors.append("WAIC is not finite")

    coverage = float(result.test_metrics["coverage_90"])
    if coverage < thresholds.min_test_coverage_90:
        errors.append(
            f"test_coverage_90={coverage:.3f} is below {thresholds.min_test_coverage_90:.3f}"
        )
    if thresholds.max_test_rmse is not None:
        rmse = float(result.test_metrics["rmse"])
        if rmse > thresholds.max_test_rmse:
            errors.append(f"test_rmse={rmse:.3f} exceeds {thresholds.max_test_rmse:.3f}")

    if thresholds.max_test_mu_rmse is not None:
        if result.latent_test is None:
            errors.append("latent test metrics are unavailable")
        elif float(result.latent_test["rmse"]) > thresholds.max_test_mu_rmse:
            errors.append(
                f"test_mu_rmse={result.latent_test['rmse']:.3f} exceeds "
                f"{thresholds.max_test_mu_rmse:.3f}"
            )
    if thresholds.min_test_mu_coverage_90 is not None:
        if result.latent_test is None:
            errors.append("latent test metrics are unavailable")
        elif float(result.latent_test["coverage_90"]) < thresholds.min_test_mu_coverage_90:
            errors.append(
                f"test_mu_coverage_90={result.latent_test['coverage_90']:.3f} is below "
                f"{thresholds.min_test_mu_coverage_90:.3f}"
            )

    if thresholds.require_alpha_in_ci:
        alpha_in_ci = bool(result.parameter_recovery and result.parameter_recovery["alpha"]["in_ci"])
        if not alpha_in_ci:
            errors.append("alpha_true is not within the posterior interval")
    if thresholds.require_sigma_in_ci:
        sigma_in_ci = bool(result.parameter_recovery and result.parameter_recovery["sigma"]["in_ci"])
        if not sigma_in_ci:
            errors.append("sigma_true is not within the posterior interval")

    if thresholds.effective_k_bounds is not None:
        lower, upper = thresholds.effective_k_bounds
        effective_k = float(result.effective_k["effective_k_mean"])
        if not lower <= effective_k <= upper:
            errors.append(f"effective_k_mean={effective_k:.3f} is outside [{lower:.3f}, {upper:.3f}]")

    if thresholds.max_pareto_k_bad is not None:
        pareto_bad = int(result.loo.get("pareto_k_bad", 0))
        if pareto_bad > thresholds.max_pareto_k_bad:
            errors.append(
                f"pareto_k_bad={pareto_bad} exceeds {thresholds.max_pareto_k_bad}"
            )
    if thresholds.max_pareto_k_very_bad is not None:
        pareto_very_bad = int(result.loo.get("pareto_k_very_bad", 0))
        if pareto_very_bad > thresholds.max_pareto_k_very_bad:
            errors.append(
                f"pareto_k_very_bad={pareto_very_bad} exceeds {thresholds.max_pareto_k_very_bad}"
            )

    if errors:
        diagnostics_lines: list[str] = []
        diagnostics_lines.append(f"- max_rhat={float(result.convergence['max_rhat']):.3f}")
        diagnostics_lines.append(f"- min_ess_bulk={float(result.convergence['min_ess_bulk']):.1f}")
        if result.label_invariant is not None:
            diagnostics_lines.append(
                f"- rhat_log_lik={float(result.label_invariant['rhat_log_lik']):.3f}"
            )
        if result.relabeled is not None:
            diagnostics_lines.append(
                f"- relabeled_max_rhat={float(result.relabeled['max_rhat']):.3f}"
            )
        if result.label_switching is not None:
            diagnostics_lines.append(
                f"- switching_rate={float(result.label_switching['switching_rate']):.3f}"
            )
            diagnostics_lines.append(
                f"- n_unique_orderings={int(result.label_switching['n_unique_orderings'])}"
            )
        message = "\n".join(
            [f"{result.label} failed benchmark thresholds:"]
            + [f"- {e}" for e in errors]
            + ["Diagnostics:"]
            + diagnostics_lines
        )
        raise AssertionError(message)


def assert_comparison_passes(
    baseline: BenchmarkCaseResult,
    candidate: BenchmarkCaseResult,
    thresholds: ComparisonThresholds,
) -> dict[str, float]:
    """Raise if a model comparison fails its thresholds."""
    comparison = compare_case_results(baseline, candidate)
    errors: list[str] = []

    if thresholds.min_delta_loo is not None and comparison["delta_loo"] < thresholds.min_delta_loo:
        errors.append(
            f"delta_loo={comparison['delta_loo']:.3f} is below {thresholds.min_delta_loo:.3f}"
        )
    if thresholds.max_delta_rmse is not None and comparison["delta_test_rmse"] > thresholds.max_delta_rmse:
        errors.append(
            f"delta_test_rmse={comparison['delta_test_rmse']:.3f} exceeds "
            f"{thresholds.max_delta_rmse:.3f}"
        )
    if (
        thresholds.max_candidate_rmse_ratio is not None
        and comparison["candidate_rmse_ratio"] > thresholds.max_candidate_rmse_ratio
    ):
        errors.append(
            f"candidate_rmse_ratio={comparison['candidate_rmse_ratio']:.3f} exceeds "
            f"{thresholds.max_candidate_rmse_ratio:.3f}"
        )

    if errors:
        message = "\n".join(
            [f"{candidate.label} failed comparison thresholds against {baseline.label}:"]
            + [f"- {e}" for e in errors]
        )
        raise AssertionError(message)

    return comparison


def plot_observed_vs_predictive(result: BenchmarkCaseResult, output_path: str | Path) -> Path:
    """Save observed vs posterior predictive plot for one benchmark case."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_idx = np.arange(len(result.y_train))
    test_idx = np.arange(len(result.y_train), len(result.y_train) + len(result.y_test))
    full_idx = np.concatenate([train_idx, test_idx])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvline(train_idx[-1], color="0.5", linestyle="--", linewidth=1, label="Train/Test Split")
    ax.plot(full_idx[: len(result.y_train)], result.y_train, color="black", linewidth=1.5, label="Observed")
    ax.plot(test_idx, result.y_test, color="black", linewidth=1.5)

    train_mean = np.asarray(result.train_metrics["y_pred_mean"])
    test_mean = np.asarray(result.test_metrics["y_pred_mean"])
    train_q05 = np.asarray(result.train_metrics["q05"])
    train_q95 = np.asarray(result.train_metrics["q95"])
    test_q05 = np.asarray(result.test_metrics["q05"])
    test_q95 = np.asarray(result.test_metrics["q95"])

    ax.plot(train_idx, train_mean, color="#1f77b4", linewidth=2, label="Posterior Mean")
    ax.fill_between(train_idx, train_q05, train_q95, color="#1f77b4", alpha=0.15)
    ax.plot(test_idx, test_mean, color="#1f77b4", linewidth=2)
    ax.fill_between(test_idx, test_q05, test_q95, color="#1f77b4", alpha=0.15, label="90% Interval")

    if result.meta is not None and "mu_true" in result.meta:
        ax.plot(full_idx, result.meta["mu_true"], color="#2ca02c", linestyle=":", linewidth=2, label="True Latent Mean")

    ax.set_title(f"{result.label}: Observed vs Posterior Predictive")
    ax.set_xlabel("Time")
    ax.set_ylabel("Response")
    ax.legend(loc="upper left", ncols=3)
    ax.text(
        0.99,
        0.02,
        f"test RMSE={result.test_metrics['rmse']:.2f}, coverage={result.test_metrics['coverage_90']:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_response_curves(result: BenchmarkCaseResult, output_path: str | Path) -> Path:
    """Save posterior response curve plot for one benchmark case."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_full = np.concatenate([result.x_train, result.x_test])
    grid = np.linspace(0.0, max(float(np.max(x_full)) * 1.1, 1.0), 200, dtype=np.float32)
    samples = result.samples

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    if "pis" not in samples:
        A = np.asarray(samples["A"], dtype=np.float32)
        k = np.asarray(samples["k"], dtype=np.float32)
        n = np.asarray(samples["n"], dtype=np.float32)
        curves = np.asarray([hill(grid, A_i, k_i, n_i) for A_i, k_i, n_i in zip(A, k, n)], dtype=np.float32)
    else:
        A = np.asarray(samples["A"], dtype=np.float32)
        k = np.asarray(samples["k"], dtype=np.float32)
        n = np.asarray(samples["n"], dtype=np.float32)
        pis = np.asarray(samples["pis"], dtype=np.float32)
        component_curves = np.stack(
            [A[:, i : i + 1] * (grid[None, :] ** n[:, i : i + 1]) / (k[:, i : i + 1] ** n[:, i : i + 1] + grid[None, :] ** n[:, i : i + 1] + 1e-12) for i in range(A.shape[1])],
            axis=-1,
        )
        curves = np.sum(component_curves * pis[:, None, :], axis=-1)

        mean_A = A.mean(axis=0)
        mean_k = k.mean(axis=0)
        mean_n = n.mean(axis=0)
        mean_pis = pis.mean(axis=0)
        for idx, (A_i, k_i, n_i, pi_i) in enumerate(zip(mean_A, mean_k, mean_n, mean_pis, strict=True)):
            component = np.asarray(hill(grid, A_i, k_i, n_i), dtype=np.float32)
            ax.plot(grid, component, linestyle="--", linewidth=1.0, alpha=0.6, label=f"Posterior Component {idx + 1} ({pi_i:.2f})")

    curve_mean = curves.mean(axis=0)
    curve_q05 = np.quantile(curves, 0.05, axis=0)
    curve_q95 = np.quantile(curves, 0.95, axis=0)

    ax.plot(grid, curve_mean, color="#d62728", linewidth=2.0, label="Posterior Mean Effect")
    ax.fill_between(grid, curve_q05, curve_q95, color="#d62728", alpha=0.15, label="90% Interval")

    if result.meta is not None and {"A_true", "k_true", "n_true", "pi_true"}.issubset(result.meta):
        A_true = np.asarray(result.meta["A_true"], dtype=np.float32)
        k_true = np.asarray(result.meta["k_true"], dtype=np.float32)
        n_true = np.asarray(result.meta["n_true"], dtype=np.float32)
        pi_true = np.asarray(result.meta["pi_true"], dtype=np.float32)
        true_components = np.stack(
            [np.asarray(hill(grid, A_i, k_i, n_i), dtype=np.float32) for A_i, k_i, n_i in zip(A_true, k_true, n_true, strict=True)],
            axis=0,
        )
        true_curve = np.sum(pi_true[:, None] * true_components, axis=0)
        ax.plot(grid, true_curve, color="#2ca02c", linewidth=2.0, linestyle=":", label="True Expected Effect")

    ax.set_title(f"{result.label}: Response Curves")
    ax.set_xlabel("Spend")
    ax.set_ylabel("Incremental Effect")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_case_comparison(
    results: Sequence[BenchmarkCaseResult],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save comparison plot across benchmark cases."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [result.model_name for result in results]
    loo_values = [float(result.loo.get("elpd_loo", np.nan)) for result in results]
    rmse_values = [float(result.test_metrics["rmse"]) for result in results]
    coverage_values = [float(result.test_metrics["coverage_90"]) for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    metrics = [
        ("ELPD-LOO", loo_values, "#1f77b4"),
        ("Test RMSE", rmse_values, "#ff7f0e"),
        ("Test Coverage 90%", coverage_values, "#2ca02c"),
    ]

    for ax, (metric_name, values, color) in zip(axes, metrics, strict=True):
        ax.bar(labels, values, color=color, alpha=0.85)
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=20)
        if "Coverage" in metric_name:
            ax.axhline(0.90, color="0.4", linestyle="--", linewidth=1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def resolve_case_artifact_dir(output_root: str | Path, result: BenchmarkCaseResult) -> Path:
    """Return the artifact directory for one benchmark case."""
    return Path(output_root) / result.domain / result.model_name


def resolve_comparison_artifact_dir(output_root: str | Path, domain: str) -> Path:
    """Return the artifact directory for comparison plots within one domain."""
    return Path(output_root) / domain / "_comparisons"


def save_case_artifacts(result: BenchmarkCaseResult, output_root: str | Path) -> dict[str, Path]:
    """Write a summary JSON and default plots for one benchmark case."""
    output_dir = resolve_case_artifact_dir(output_root, result)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{result.label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(case_summary(result), fh, indent=2)

    predictive_path = plot_observed_vs_predictive(result, output_dir / f"{result.label}_predictive.png")
    response_path = plot_response_curves(result, output_dir / f"{result.label}_response.png")

    return {
        "summary": summary_path,
        "predictive": predictive_path,
        "response": response_path,
    }
