from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path


def _ensure_xla_host_device_count(device_count: int) -> None:
    """Append a host-device-count flag without clobbering unrelated XLA flags."""
    host_device_flag = f"--xla_force_host_platform_device_count={device_count}"
    tokens = os.environ.get("XLA_FLAGS", "").split()
    if host_device_flag not in tokens:
        tokens.append(host_device_flag)
        os.environ["XLA_FLAGS"] = " ".join(tokens)


_ensure_xla_host_device_count(4)

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hill_mixture_mmm.benchmark import (
    BenchmarkThresholds,
    assert_case_passes,
    run_prepared_synthetic_benchmark_case,
    save_case_artifacts,
)
from hill_mixture_mmm.controlled_tv_profile import (
    SMOKE_PROFILE_IDS,
    TV_PROFILE_LIBRARY,
    build_controlled_tv_profile_config,
    build_controlled_tv_profile_run_config,
)
from hill_mixture_mmm.data import generate_controlled_k_spacing_data
from hill_mixture_mmm.metrics import (
    compute_component_curve_tv_separation,
    compute_similarity_adjusted_effective_count,
    compute_shannon_effective_count,
)


SMOKE_MODELS = ["single_hill", "mixture_k2", "mixture_k3"]
FULL_MODELS = ["single_hill", "mixture_k2", "mixture_k3"]
SMOKE_SEEDS = [0]
FULL_SEEDS = [0, 1, 2, 3, 4]

MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
MODEL_COLORS = {
    "single_hill": "#1f77b4",
    "mixture_k2": "#9467bd",
    "mixture_k3": "#ff7f0e",
}
K_TRUE_MARKERS = {1: "o", 2: "s", 3: "^"}
K_TRUE_LINESTYLES = {2: "--", 3: "-"}

_RELAXED_BASE = BenchmarkThresholds(
    max_rhat=None,
    min_ess_bulk_per_chain=None,
    min_ess_tail_per_chain=None,
    max_divergences=None,
    min_bfmi=None,
    max_tree_depth_hits=None,
    require_finite_loo_waic=True,
    require_finite_predictive_metrics=True,
)

_DATASET_RE = re.compile(r"^controlled_tvprofile_k(?P<k_true>\d+)_(?P<profile_id>.+)$")


def _require_full_controlled_tv_profile_benchmark() -> None:
    enabled = os.getenv("HILL_MMM_RUN_FULL_CONTROLLED_TV_PROFILE_BENCHMARK", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        pytest.skip(
            "full controlled TV-profile benchmark is opt-in; set "
            "HILL_MMM_RUN_FULL_CONTROLLED_TV_PROFILE_BENCHMARK=1 to run it"
        )


def _smoke_profile_cases() -> list[tuple[int, dict[str, object]]]:
    cases: list[tuple[int, dict[str, object]]] = []
    for k_true, profile_ids in SMOKE_PROFILE_IDS.items():
        profiles = {str(profile["profile_id"]): profile for profile in TV_PROFILE_LIBRARY[k_true]}
        for profile_id in profile_ids:
            cases.append((k_true, profiles[profile_id]))
    return cases


def _full_profile_cases() -> list[tuple[int, dict[str, object]]]:
    return [
        (k_true, profile)
        for k_true, profiles in TV_PROFILE_LIBRARY.items()
        for profile in profiles
    ]


def _controlled_thresholds() -> BenchmarkThresholds:
    return replace(
        _RELAXED_BASE,
        require_reportable_diagnostics=True,
        require_truth_metrics=True,
    )


def _dataset_name(k_true: int, profile_id: str) -> str:
    return f"controlled_tvprofile_k{k_true}_{profile_id}"


def _label(k_true: int, profile_id: str, model_name: str, seed: int) -> str:
    return f"{_dataset_name(k_true, profile_id)}_{model_name}_seed{seed}"


def _case_summary_path(
    benchmark_output_root: Path,
    *,
    k_true: int,
    profile_id: str,
    model_name: str,
    seed: int,
) -> Path:
    return (
        benchmark_output_root
        / "synthetic"
        / model_name
        / f"{_label(k_true, profile_id, model_name, seed)}_summary.json"
    )


def _run_and_assert_controlled_case(
    *,
    k_true: int,
    profile: dict[str, object],
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
    quick: bool,
) -> None:
    profile_id = str(profile["profile_id"])
    config = build_controlled_tv_profile_config(k_true=k_true, seed=seed, profile=profile)
    x, y, meta = generate_controlled_k_spacing_data(config)
    result = run_prepared_synthetic_benchmark_case(
        dataset_name=_dataset_name(k_true, profile_id),
        x=x,
        y=y,
        meta=meta,
        model_name=model_name,
        config=build_controlled_tv_profile_run_config(
            model_name,
            seed,
            quick=quick,
        ),
        label=_label(k_true, profile_id, model_name, seed),
    )

    artifacts = save_case_artifacts(result, benchmark_output_root)
    assert_case_passes(result, _controlled_thresholds())
    for path in artifacts.values():
        assert path.exists(), f"Expected controlled TV-profile artifact at {path}"


def _load_selected_metric_rows(summary_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        match = _DATASET_RE.match(str(summary["dataset_name"]))
        assert match, f"Unexpected controlled dataset_name: {summary['dataset_name']}"
        k_true = int(match.group("k_true"))
        profile_id = match.group("profile_id")
        component_summary = summary.get("component_summary")
        true_component_summary = summary.get("true_component_summary")
        true_separation = (
            float(compute_component_curve_tv_separation(true_component_summary)["mean_pairwise_tv"])
            if true_component_summary is not None
            else 0.0
        )
        if component_summary is None:
            active_component_count = 1.0
            similarity_adjusted_count = 1.0
            shannon_count = 1.0
        else:
            active_component_count = float(component_summary["K_active"])
            similarity_adjusted_count = float(
                compute_similarity_adjusted_effective_count(component_summary)["effective_count"]
            )
            shannon_count = float(compute_shannon_effective_count(component_summary)["effective_count"])
        rows.append(
            {
                "seed": int(summary["seed"]),
                "K_true": k_true,
                "profile_id": profile_id,
                "model": str(summary["model_name"]),
                "true_separation": true_separation,
                "strict_converged": bool(summary["converged"]),
                "converged": bool(summary["benchmark_pass"]),
                "publication_status": str(summary["publication_status"]),
                "sampler_status": str(summary["diagnostic_status"]["sampler_status"]),
                "mixing_status": str(summary["diagnostic_status"]["mixing_status"]),
                "interpretation_status": str(summary["diagnostic_status"]["interpretation_status"]),
                "active_component_count": active_component_count,
                "similarity_adjusted_count": similarity_adjusted_count,
                "shannon_count": shannon_count,
            }
        )
    return pd.DataFrame(rows).sort_values(["K_true", "true_separation", "seed", "model"]).reset_index(drop=True)


def _plot_selected_metrics(df: pd.DataFrame, *, output_path: Path) -> None:
    metric_specs = [
        ("active_component_count", "Active K (pi > 0.05)"),
        ("similarity_adjusted_count", "Similarity-Adjusted Count"),
        ("shannon_count", "Hill q=1 (weights only)"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs) + 1, figsize=(16.5, 4.8), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    seed_values = sorted(pd.unique(df["seed"]))
    seed_offsets = {
        int(seed): (idx - (len(seed_values) - 1) / 2) * 0.006 for idx, seed in enumerate(seed_values)
    }

    for axis_idx, (metric_key, title) in enumerate(metric_specs):
        ax = axes_flat[axis_idx]
        for target in [1, 2, 3]:
            ax.axhline(target, color="0.9", linestyle="--", linewidth=0.8, zorder=0)
        for model_name in FULL_MODELS:
            model_panel = df[df["model"] == model_name]
            for k_true in sorted(pd.unique(model_panel["K_true"])):
                panel = model_panel[model_panel["K_true"] == k_true]
                if panel.empty:
                    continue
                x = panel["true_separation"].to_numpy(dtype=float) + panel["seed"].map(seed_offsets).to_numpy(dtype=float)
                ax.scatter(
                    x,
                    panel[metric_key],
                    color=MODEL_COLORS[model_name],
                    marker=K_TRUE_MARKERS[int(k_true)],
                    s=42,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )
                means = (
                    panel.groupby("profile_id", as_index=False)
                    .agg(true_separation=("true_separation", "mean"), y=(metric_key, "mean"))
                    .sort_values("true_separation")
                )
                ax.plot(
                    means["true_separation"],
                    means["y"],
                    color=MODEL_COLORS[model_name],
                    linewidth=1.6,
                    alpha=0.95,
                    linestyle=K_TRUE_LINESTYLES.get(int(k_true), "-"),
                )
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0.8, 3.2)
        ax.set_xlabel("True Component Separation (TV)")
        ax.grid(True, alpha=0.22)

    legend_ax = axes_flat[-1]
    legend_ax.axis("off")
    model_handles = [
        plt.Line2D([], [], color=MODEL_COLORS[m], linewidth=1.8, label=MODEL_LABELS[m])
        for m in FULL_MODELS
    ]
    marker_handles = [
        plt.Line2D([], [], linestyle="None", marker=K_TRUE_MARKERS[k], markersize=7, markerfacecolor="0.35", markeredgecolor="white", label=f"Data K={k}")
        for k in sorted(pd.unique(df["K_true"]))
    ]
    line_handles = [
        plt.Line2D([], [], color="0.35", linewidth=1.8, linestyle=K_TRUE_LINESTYLES[k], label=f"Line: Data K={k}")
        for k in sorted(k for k in pd.unique(df["K_true"]) if int(k) in K_TRUE_LINESTYLES)
    ]
    legend_ax.legend(handles=model_handles + marker_handles + line_handles, loc="center", frameon=False)
    axes_flat[0].set_ylabel("Estimated Count")

    fig.suptitle("Controlled TV-Profile Benchmark", y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_selected_metric_artifacts(
    *,
    benchmark_output_root: Path,
    cases: list[tuple[int, dict[str, object]]],
    seeds: list[int],
    models: list[str],
    artifact_name: str,
) -> None:
    summary_paths = [
        _case_summary_path(
            benchmark_output_root,
            k_true=k_true,
            profile_id=str(profile["profile_id"]),
            model_name=model_name,
            seed=seed,
        )
        for k_true, profile in cases
        for model_name in models
        for seed in seeds
    ]
    for summary_path in summary_paths:
        assert summary_path.exists(), f"Expected controlled benchmark summary at {summary_path}"

    df = _load_selected_metric_rows(summary_paths)
    summary = (
        df.groupby(["K_true", "profile_id", "model"], as_index=False)
        .agg(
            true_separation=("true_separation", "mean"),
            active_component_count=("active_component_count", "mean"),
            similarity_adjusted_count=("similarity_adjusted_count", "mean"),
            shannon_count=("shannon_count", "mean"),
            converged_rate=("converged", "mean"),
            strict_converged_rate=("strict_converged", "mean"),
            publication_pass_rate=("publication_status", lambda s: float((s == "Pass").mean())),
            publication_fail_rate=("publication_status", lambda s: float((s == "Fail").mean())),
        )
        .sort_values(["K_true", "true_separation", "model"])
        .reset_index(drop=True)
    )

    artifact_dir = benchmark_output_root / "controlled_tv_profile" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = artifact_dir / "selected_metric_results.csv"
    summary_csv = artifact_dir / "selected_metric_summary.csv"
    plot_path = artifact_dir / "selected_metric_comparison.png"
    metadata_path = artifact_dir / "metadata.json"

    df.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_selected_metrics(df, output_path=plot_path)
    metadata_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "models": models,
                "metrics": [
                    "active_component_count",
                    "similarity_adjusted_count",
                    "shannon_count",
                ],
                "cases": [{"K_true": int(k_true), "profile_id": str(profile["profile_id"])} for k_true, profile in cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert raw_csv.exists()
    assert summary_csv.exists()
    assert plot_path.exists()
    assert metadata_path.exists()
    assert len(df) == len(cases) * len(models) * len(seeds)


@pytest.mark.slow
@pytest.mark.benchmark_smoke
@pytest.mark.parametrize("seed", SMOKE_SEEDS)
@pytest.mark.parametrize("model_name", SMOKE_MODELS)
@pytest.mark.parametrize(("k_true", "profile"), _smoke_profile_cases())
def test_controlled_tv_profile_smoke_matrix(
    k_true: int,
    profile: dict[str, object],
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _run_and_assert_controlled_case(
        k_true=k_true,
        profile=profile,
        model_name=model_name,
        seed=seed,
        benchmark_output_root=benchmark_output_root,
        quick=True,
    )


@pytest.mark.slow
@pytest.mark.benchmark_smoke
def test_controlled_tv_profile_smoke_selected_metric_artifacts(
    benchmark_output_root: Path,
) -> None:
    _write_selected_metric_artifacts(
        benchmark_output_root=benchmark_output_root,
        cases=_smoke_profile_cases(),
        seeds=SMOKE_SEEDS,
        models=SMOKE_MODELS,
        artifact_name="smoke",
    )


@pytest.mark.slow
@pytest.mark.benchmark_full
@pytest.mark.parametrize("seed", FULL_SEEDS)
@pytest.mark.parametrize("model_name", FULL_MODELS)
@pytest.mark.parametrize(("k_true", "profile"), _full_profile_cases())
def test_controlled_tv_profile_full_matrix(
    k_true: int,
    profile: dict[str, object],
    model_name: str,
    seed: int,
    benchmark_output_root: Path,
) -> None:
    _require_full_controlled_tv_profile_benchmark()
    _run_and_assert_controlled_case(
        k_true=k_true,
        profile=profile,
        model_name=model_name,
        seed=seed,
        benchmark_output_root=benchmark_output_root,
        quick=False,
    )


@pytest.mark.slow
@pytest.mark.benchmark_full
def test_controlled_tv_profile_full_selected_metric_artifacts(
    benchmark_output_root: Path,
) -> None:
    _require_full_controlled_tv_profile_benchmark()
    _write_selected_metric_artifacts(
        benchmark_output_root=benchmark_output_root,
        cases=_full_profile_cases(),
        seeds=FULL_SEEDS,
        models=FULL_MODELS,
        artifact_name="full",
    )
