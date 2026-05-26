"""Compute posterior cosine separation for resolvability and real-data fits.

The resolvability paper section reports the mean pairwise cosine distance
between the *true* Hill component curves on synthetic DGPs. For real data
there is no ground truth, so we recompute the same quantity from each fit's
posterior ``component_summary``. To keep the metric comparable across fits,
we use a fit-specific saturation grid that covers the active components'
half-saturation range, removing the dependence on ``scale_reference``.

The script writes two CSVs under ``results/``:

* ``posterior_separation_real.csv`` — per-fit posterior cosine separation
  and Shannon effective component count for the real-data benchmark.
* ``posterior_separation_resolvability.csv`` — same quantities for the
  90 resolvability fits, with the true cosine separation retained as
  a reference column.

It then renders ``paper/figures/fig_real_resolvability_overlay.png``,
overlaying the real organisations on the resolvability scatter to
visualise where field data lands relative to the resolvability threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hill_mixture_mmm.metrics import compute_component_curve_cosine_separation

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
RESULTS_ROOT = REPO_ROOT / "results"

REAL_DIRS = {
    "single_hill": FIGURES_ROOT / "real" / "single_hill",
    "mixture_k2": FIGURES_ROOT / "real" / "mixture_k2",
    "mixture_k3": FIGURES_ROOT / "real" / "mixture_k3",
}

RESOLV_DIRS = {
    "mixture_k2": FIGURES_ROOT / "synthetic" / "mixture_k2",
    "mixture_k3": FIGURES_ROOT / "synthetic" / "mixture_k3",
}

DATASET_LABELS = {
    "beauty_fitness": "Beauty & Fitness",
    "home_garden": "Home & Garden",
    "toys_hobbies": "Toys & Hobbies",
}
MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture K=2",
    "mixture_k3": "Mixture K=3",
}

# Use the same grid resolution as the resolvability study, but stretch
# curve_grid_max so that the largest active k_ratio sits at u=4 of the
# normalised range. This makes the cosine separation a property of the
# curve shape and independent of how scale_reference was set.
GRID_K_MULT = 4.0
GRID_SIZE = 128
THRESHOLD_LINE = 0.10  # paper-level resolvability threshold


def _adaptive_curve_grid_max(component_summary: dict) -> float:
    components = component_summary.get("components", [])
    active_k = [
        float(c.get("k_ratio_mean", 0.0))
        for c in components
        if c.get("active", False) and float(c.get("pi_mean", 0.0)) > 0.0
    ]
    if not active_k:
        return 4.0
    return float(max(active_k) * GRID_K_MULT)


def _posterior_cosine(component_summary: dict) -> float:
    grid_max = _adaptive_curve_grid_max(component_summary)
    out = compute_component_curve_cosine_separation(
        component_summary,
        curve_grid_max=grid_max,
        grid_size=GRID_SIZE,
    )
    return float(out.get("mean_pairwise_cosine", 0.0))


def _shannon_count(component_summary: dict) -> float:
    pis = np.asarray(
        [
            float(c.get("pi_mean", 0.0))
            for c in component_summary.get("components", [])
        ],
        dtype=np.float64,
    )
    pis = pis / pis.sum() if pis.sum() > 0 else pis
    entropy = -np.nansum(pis * np.log(np.where(pis > 0, pis, 1.0)))
    return float(np.exp(entropy))


def _publication_pass(summary: dict) -> bool:
    return str(summary.get("publication_status", "")).lower() == "pass"


def _dataset_from_label(label: str) -> str:
    # label is like ``real_beauty_fitness_mixture_k3_seed0`` — peel off
    # the leading ``real_`` and trailing ``_<model>_seed<n>``.
    parts = label.split("_")
    for end in range(len(parts) - 1, 0, -1):
        if parts[end].startswith("seed"):
            return "_".join(parts[1 : end - 2])
    return "unknown"


def _load_real_records() -> pd.DataFrame:
    rows = []
    for model, directory in REAL_DIRS.items():
        if model == "single_hill":
            continue
        for path in sorted(directory.glob("real_*_summary.json")):
            with path.open() as fh:
                data = json.load(fh)
            comp = data.get("component_summary") or {}
            if not comp.get("components"):
                continue
            rows.append(
                {
                    "dataset": _dataset_from_label(data["label"]),
                    "model": data["model_name"],
                    "seed": int(data["seed"]),
                    "posterior_cosine": _posterior_cosine(comp),
                    "shannon_count": _shannon_count(comp),
                    "K_active": int(comp.get("K_active", 0)),
                    "publication_pass": _publication_pass(data),
                }
            )
    return pd.DataFrame(rows)


def _load_resolvability_records() -> pd.DataFrame:
    rows = []
    for model, directory in RESOLV_DIRS.items():
        for path in sorted(directory.glob("resolvability_*_summary.json")):
            with path.open() as fh:
                data = json.load(fh)
            comp = data.get("component_summary") or {}
            if not comp.get("components"):
                continue
            label = data["label"]
            tokens = label.split("_")
            K_true = int(tokens[1][1:]) if tokens[1].startswith("k") else None
            profile = "_".join(tokens[2:-2])
            rows.append(
                {
                    "label": label,
                    "model": data["model_name"],
                    "seed": int(data["seed"]),
                    "K_true": K_true,
                    "profile_id": profile,
                    "posterior_cosine": _posterior_cosine(comp),
                    "shannon_count": _shannon_count(comp),
                    "K_active": int(comp.get("K_active", 0)),
                    "publication_pass": _publication_pass(data),
                }
            )
    return pd.DataFrame(rows)


def _attach_true_separation(df: pd.DataFrame) -> pd.DataFrame:
    csv_path = (
        FIGURES_ROOT
        / "component_resolvability"
        / "full"
        / "selected_metric_results.csv"
    )
    true_df = pd.read_csv(csv_path)
    merged = df.merge(
        true_df[["seed", "K_true", "profile_id", "model", "true_cosine_separation"]],
        on=["seed", "K_true", "profile_id", "model"],
        how="left",
    )
    return merged


def _render_overlay(real_df: pd.DataFrame, resolv_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Synthetic resolvability cloud as background.
    converged = resolv_df["publication_pass"]
    ax.scatter(
        resolv_df.loc[converged, "posterior_cosine"],
        resolv_df.loc[converged, "shannon_count"],
        s=22,
        c="#bdbdbd",
        alpha=0.65,
        edgecolors="none",
        label="Synthetic resolvability fits (pub-pass)",
    )
    ax.scatter(
        resolv_df.loc[~converged, "posterior_cosine"],
        resolv_df.loc[~converged, "shannon_count"],
        s=22,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=0.9,
        label="Synthetic resolvability fits (pub-fail)",
    )

    # Real data — averaged per (dataset, model) over seeds.
    agg = (
        real_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            posterior_cosine_mean=("posterior_cosine", "mean"),
            posterior_cosine_std=("posterior_cosine", "std"),
            shannon_mean=("shannon_count", "mean"),
            shannon_std=("shannon_count", "std"),
            pub_pass_rate=("publication_pass", "mean"),
        )
        .fillna(0.0)
    )

    marker_for = {"mixture_k2": "s", "mixture_k3": "^"}
    color_for = {
        "beauty_fitness": "#1f77b4",
        "home_garden": "#2ca02c",
        "toys_hobbies": "#9467bd",
    }
    for _, row in agg.iterrows():
        ax.errorbar(
            row["posterior_cosine_mean"],
            row["shannon_mean"],
            xerr=row["posterior_cosine_std"],
            yerr=row["shannon_std"],
            fmt=marker_for.get(row["model"], "o"),
            color=color_for.get(row["dataset"], "black"),
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=0.8,
            elinewidth=1.0,
            capsize=3,
            label=f"{DATASET_LABELS[row['dataset']]} · {MODEL_LABELS[row['model']]}",
        )

    ax.axvline(THRESHOLD_LINE, color="#444444", linestyle="--", linewidth=1.0)
    ax.text(
        THRESHOLD_LINE + 0.005,
        ax.get_ylim()[1] * 0.96 if ax.get_ylim()[1] > 0 else 0.96,
        "Resolvability\nthreshold (~0.1)",
        fontsize=9,
        color="#444444",
        va="top",
    )

    ax.set_xlabel("Posterior mean pairwise cosine distance between active Hill components")
    ax.set_ylabel("Posterior Shannon effective component count $^1\\!D$")
    ax.set_title("Real-data fits sit at low posterior separation, below the resolvability transition")
    ax.set_xlim(left=-0.01)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # Compact legend — collapse duplicates.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    cleaned = [(h, l) for h, l in zip(handles, labels) if (l not in seen and not seen.add(l))]
    ax.legend(
        [h for h, _ in cleaned],
        [l for _, l in cleaned],
        fontsize=8,
        loc="lower right",
        framealpha=0.92,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    real_df = _load_real_records()
    resolv_df = _attach_true_separation(_load_resolvability_records())

    real_path = RESULTS_ROOT / "posterior_separation_real.csv"
    resolv_path = RESULTS_ROOT / "posterior_separation_resolvability.csv"
    real_df.to_csv(real_path, index=False)
    resolv_df.to_csv(resolv_path, index=False)
    print(f"Wrote {real_path}")
    print(f"Wrote {resolv_path}")

    fig_path = FIGURES_ROOT / "fig_real_resolvability_overlay.png"
    _render_overlay(real_df, resolv_df, fig_path)
    print(f"Wrote {fig_path}")

    summary = (
        real_df.groupby(["dataset", "model"])
        .agg(
            posterior_cosine_mean=("posterior_cosine", "mean"),
            posterior_cosine_std=("posterior_cosine", "std"),
            shannon_mean=("shannon_count", "mean"),
            pub_pass_rate=("publication_pass", "mean"),
        )
        .round(3)
    )
    print("\nReal-data posterior separation summary (mean across seeds):")
    print(summary)


if __name__ == "__main__":
    main()
