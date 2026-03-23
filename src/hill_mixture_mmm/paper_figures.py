"""Generate selected paper figures from synthetic benchmark results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DGP_ORDER = ["single", "mixture_k2", "mixture_k3"]
DGP_K_TRUE = {"single": 1, "mixture_k2": 2, "mixture_k3": 3}
DGP_LABELS = {
    "single": "Single (K=1)",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3"]
MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
COLORS = {
    "single_hill": "#1f77b4",
    "mixture_k2": "#9467bd",
    "mixture_k3": "#ff7f0e",
}
DEFAULT_FIGURE_IDS = ("fig0", "fig1", "fig2", "fig3", "fig4")
RHAT_TEST_PASS_MAX = 1.05

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _safe_float(value: Any) -> float | None:
    """Return a float when possible, else None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_metric(summary: dict[str, Any], section: str, key: str) -> float | None:
    """Read a numeric metric from a nested summary dictionary."""
    payload = summary.get(section)
    if not isinstance(payload, dict):
        return None
    return _safe_float(payload.get(key))


def _normalize_bool(series: pd.Series) -> pd.Series:
    """Convert boolean-like benchmark columns into strict bools."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def _compute_rhat_test_pass(df: pd.DataFrame) -> pd.Series:
    """Return whether each row passes the R-hat benchmark threshold.

    The full synthetic benchmark uses `max_rhat <= 1.05` for the single-Hill
    model and `label_invariant_max_rhat <= 1.05` for mixture models.
    When a raw CSV does not include label-invariant diagnostics, fall back to
    the available `max_rhat` column so figure generation still works.
    """
    if "rhat_test_pass" in df.columns:
        return _normalize_bool(df["rhat_test_pass"])

    standard_rhat = pd.to_numeric(df.get("max_rhat"), errors="coerce")
    if "model" not in df.columns:
        return standard_rhat.le(RHAT_TEST_PASS_MAX).fillna(False)

    mixture_mask = df["model"].astype(str) != "single_hill"
    if "label_invariant_max_rhat" in df.columns:
        label_rhat = pd.to_numeric(df["label_invariant_max_rhat"], errors="coerce")
        rhat_used = standard_rhat.where(~mixture_mask, label_rhat)
    else:
        rhat_used = standard_rhat
    return rhat_used.le(RHAT_TEST_PASS_MAX).fillna(False)


def _summary_to_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Project a seed-level synthetic summary JSON into one raw-results record."""
    dgp_name = str(summary["dataset_name"])
    label_invariant = summary.get("label_invariant") or {}
    relabeled = summary.get("relabeled") or {}
    publication_status = summary.get("publication_status")
    interpretation_status = summary.get("interpretation_status")
    benchmark_pass = summary.get("benchmark_pass")
    if benchmark_pass is None and publication_status is not None:
        benchmark_pass = str(publication_status).lower() != "fail"

    return {
        "dgp": dgp_name,
        "K_true": DGP_K_TRUE.get(dgp_name),
        "model": summary["model_name"],
        "seed": int(summary["seed"]),
        "converged": bool(summary["converged"]),
        "publication_status": publication_status,
        "interpretation_status": interpretation_status,
        "benchmark_pass": bool(benchmark_pass) if benchmark_pass is not None else bool(summary["converged"]),
        "max_rhat": _nested_metric(summary, "convergence", "max_rhat"),
        "min_ess_bulk": _nested_metric(summary, "convergence", "min_ess_bulk"),
        "min_ess_tail": _nested_metric(summary, "convergence", "min_ess_tail"),
        "label_invariant_max_rhat": _safe_float(label_invariant.get("max_rhat")),
        "rhat_log_lik": _safe_float(label_invariant.get("rhat_log_lik")),
        "relabeled_max_rhat": _safe_float(relabeled.get("max_rhat")),
        "num_divergences": _nested_metric(summary, "hmc_diagnostics", "num_divergences"),
        "min_bfmi": _nested_metric(summary, "hmc_diagnostics", "min_bfmi"),
        "tree_depth_hits": _nested_metric(summary, "hmc_diagnostics", "tree_depth_hits"),
        "elpd_loo": _nested_metric(summary, "loo", "elpd_loo"),
        "pareto_k_bad": _nested_metric(summary, "loo", "pareto_k_bad"),
        "pareto_k_very_bad": _nested_metric(summary, "loo", "pareto_k_very_bad"),
        "train_mape": _nested_metric(summary, "train_metrics", "mape"),
        "test_mape": _nested_metric(summary, "test_metrics", "mape"),
        "train_coverage_90": _nested_metric(summary, "train_metrics", "coverage_90"),
        "test_coverage_90": _nested_metric(summary, "test_metrics", "coverage_90"),
        "effective_k_mean": _nested_metric(summary, "effective_k", "mean"),
        "effective_k_std": _nested_metric(summary, "effective_k", "std"),
    }


def load_synthetic_results_from_artifacts(
    artifact_root: str | Path,
    *,
    summary_paths: Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    """Load seed-level synthetic benchmark rows from saved case summaries."""
    artifact_root = Path(artifact_root)
    if summary_paths is None:
        paths = sorted(artifact_root.glob("synthetic/*/*_seed*_summary.json"))
    else:
        paths = [Path(path) for path in summary_paths]

    rows_by_case: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("domain") != "synthetic":
            continue
        record = _summary_to_record(summary)
        key = (record["dgp"], record["model"], record["seed"])
        rows_by_case[key] = record

    if not rows_by_case:
        raise ValueError(f"No synthetic seed-level summaries found under {artifact_root}")

    df = pd.DataFrame(rows_by_case.values())
    if "converged" in df.columns:
        df["converged"] = _normalize_bool(df["converged"])
    if "benchmark_pass" in df.columns:
        df["benchmark_pass"] = _normalize_bool(df["benchmark_pass"])
    df["rhat_test_pass"] = _compute_rhat_test_pass(df)
    return df


def load_synthetic_results(
    *,
    results_csv: str | Path | None = None,
    artifact_root: str | Path | None = None,
    summary_paths: Sequence[str | Path] | None = None,
) -> pd.DataFrame:
    """Load synthetic benchmark rows from either a raw CSV or saved summaries."""
    if results_csv is not None:
        df = pd.read_csv(results_csv)
        if "converged" in df.columns:
            df["converged"] = _normalize_bool(df["converged"])
        if "benchmark_pass" in df.columns:
            df["benchmark_pass"] = _normalize_bool(df["benchmark_pass"])
        df["rhat_test_pass"] = _compute_rhat_test_pass(df)
        return df
    if artifact_root is None:
        raise ValueError("Pass either results_csv or artifact_root")
    return load_synthetic_results_from_artifacts(artifact_root, summary_paths=summary_paths)


def _metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate one metric to mean/std per DGP-model cell."""
    metric_frame = (
        df.groupby(["dgp", "model"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
        .fillna({"std": 0.0})
    )
    return metric_frame


def _lookup_metric(metric_frame: pd.DataFrame, dgp_name: str, model_name: str) -> tuple[float, float]:
    """Return mean/std for one DGP-model cell."""
    row = metric_frame[
        (metric_frame["dgp"] == dgp_name) & (metric_frame["model"] == model_name)
    ]
    if row.empty:
        return np.nan, np.nan
    return float(row["mean"].iloc[0]), float(row["std"].iloc[0])


def generate_graphical_model_figure(output_dir: str | Path) -> Path:
    """Render Figure 0: conceptual overview of the Hill mixture model."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    edge = "#1f2937"
    text = "#111827"
    panel_fill = "#f8fafc"
    latent_fill = "#ffffff"
    observed_fill = "#d1d5db"

    def add_panel(x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18,rounding_size=0.15",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=panel_fill,
        )
        ax.add_patch(patch)
        ax.text(x + 0.2, y + h - 0.25, title, ha="left", va="top", fontsize=12, color=text, weight="bold")
        ax.text(x + 0.2, y + h - 0.6, subtitle, ha="left", va="top", fontsize=9, color="#4b5563")

    def add_node(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        observed: bool = False,
        fontsize: int = 14,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=observed_fill if observed else latent_fill,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=text)

    def add_arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=edge,
            shrinkA=8,
            shrinkB=8,
        )
        ax.add_patch(arrow)

    add_panel(0.7, 2.0, 4.2, 3.5, "Temporal structure", r"$t = 1, \ldots, T$")
    add_panel(5.4, 2.0, 3.4, 3.5, "Segment structure", r"$k = 1, \ldots, K$")
    add_panel(9.3, 2.0, 3.7, 3.5, "Observation model", "Mixture over segment responses")

    add_node(1.0, 4.35, 0.9, 0.6, r"$\alpha$")
    add_node(1.0, 3.2, 1.0, 0.65, r"$x_t$", observed=True)
    add_node(2.45, 3.2, 1.0, 0.65, r"$s_t$")
    add_node(3.75, 4.35, 1.25, 0.6, r"$\mu_0,\beta$")
    add_node(3.95, 2.75, 0.9, 0.65, r"$b_t$")

    add_node(6.05, 4.35, 1.1, 0.6, r"$\theta_k$")
    add_node(5.9, 3.0, 1.4, 0.75, r"$h_{t,k}$")

    add_node(9.8, 4.35, 1.0, 0.6, r"$\pi_k$")
    add_node(11.35, 4.35, 1.0, 0.6, r"$\sigma$")
    add_node(10.5, 3.0, 1.25, 0.75, r"$y_t$", observed=True)

    add_arrow((1.45, 4.35), (2.95, 3.55))
    add_arrow((2.0, 3.52), (2.45, 3.52))
    add_arrow((3.45, 3.52), (5.9, 3.38))
    add_arrow((4.38, 4.35), (4.38, 3.35))
    add_arrow((4.85, 3.05), (10.5, 3.3))
    add_arrow((6.6, 4.35), (6.6, 3.75))
    add_arrow((7.3, 3.38), (10.5, 3.38))
    add_arrow((10.3, 4.35), (11.0, 3.75))
    add_arrow((11.85, 4.35), (11.25, 3.75))

    ax.text(1.45, 2.55, "spend", ha="center", fontsize=10, color="#374151")
    ax.text(2.95, 2.55, "geometric adstock", ha="center", fontsize=10, color="#374151")
    ax.text(4.4, 2.1, "linear baseline", ha="center", fontsize=10, color="#374151")
    ax.text(6.6, 2.55, r"segment parameters $(A_k, k_k, n_k)$", ha="center", fontsize=10, color="#374151")
    ax.text(6.6, 2.22, "Hill response for segment k", ha="center", fontsize=10, color="#374151")
    ax.text(10.3, 2.55, "stick-breaking weights", ha="center", fontsize=10, color="#374151")
    ax.text(11.85, 2.55, "Gaussian noise", ha="center", fontsize=10, color="#374151")
    ax.text(11.1, 2.2, r"$y_t \sim \sum_k \pi_k \,\mathcal{N}(b_t + h_{t,k}, \sigma^2)$", ha="center", fontsize=10, color=text)

    fig.tight_layout()
    output_path = output_dir / "fig0_graphical_model.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_elpd_comparison_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 1: ELPD-LOO comparison across DGPs and models."""
    output_dir = Path(output_dir)
    metric_frame = _metric_frame(df, "elpd_loo")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DGP_ORDER))
    width = min(0.28, 0.8 / max(len(MODEL_ORDER), 1))

    for idx, model_name in enumerate(MODEL_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for dgp_name in DGP_ORDER:
            mean, std = _lookup_metric(metric_frame, dgp_name, model_name)
            means.append(mean)
            stds.append(std)

        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Data Generating Process")
    ax.set_ylabel("ELPD-LOO")
    ax.set_title("Model Comparison: Expected Log Pointwise Predictive Density")
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])
    ax.legend(title="Model")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.annotate(
        "Error bars: ±1 std across random seeds",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    output_path = output_dir / "fig1_elpd_comparison.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_mape_comparison_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 2: holdout test MAPE comparison across DGPs and models."""
    output_dir = Path(output_dir)
    metric_frame = _metric_frame(df, "test_mape")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DGP_ORDER))
    width = min(0.28, 0.8 / max(len(MODEL_ORDER), 1))

    max_height = 0.0
    for idx, model_name in enumerate(MODEL_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for dgp_name in DGP_ORDER:
            mean, std = _lookup_metric(metric_frame, dgp_name, model_name)
            means.append(mean)
            stds.append(std)
            if not np.isnan(mean):
                max_height = max(max_height, mean + (0.0 if np.isnan(std) else std))

        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model_name],
            color=COLORS[model_name],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Data Generating Process")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Model Comparison: Holdout Test MAPE (Lower Is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])
    ax.legend(title="Model")
    ax.set_ylim(0, max(1.0, max_height * 1.15))
    ax.annotate(
        "Lower is better; error bars: ±1 std across random seeds",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    output_path = output_dir / "fig2_mape_comparison.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_convergence_heatmap_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 4: R-hat pass-rate heatmap by DGP and model."""
    output_dir = Path(output_dir)
    convergence = _compute_rhat_test_pass(df)
    conv_rates = (
        df.assign(converged=convergence)
        .groupby(["dgp", "model"])["converged"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(index=DGP_ORDER, columns=MODEL_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(conv_rates.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("R-hat Pass Rate")

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticks(np.arange(len(DGP_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[model_name] for model_name in MODEL_ORDER])
    ax.set_yticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER])

    for row_idx, dgp_name in enumerate(DGP_ORDER):
        for col_idx, model_name in enumerate(MODEL_ORDER):
            value = float(conv_rates.loc[dgp_name, model_name])
            text_color = "white" if value < 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Data Generating Process")
    ax.set_title("R-hat Threshold Pass Rate by DGP and Model")

    plt.tight_layout()
    output_path = output_dir / "fig4_convergence_heatmap.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_coverage_figure(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Render Figure 3: train/test 90% interval coverage."""
    output_dir = Path(output_dir)
    coverage = (
        df.groupby(["dgp", "model"], as_index=False)
        .agg(
            train_cov_mean=("train_coverage_90", "mean"),
            train_cov_std=("train_coverage_90", "std"),
            test_cov_mean=("test_coverage_90", "mean"),
            test_cov_std=("test_coverage_90", "std"),
        )
        .fillna({"train_cov_std": 0.0, "test_cov_std": 0.0})
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    dgp_spacing = 0.3
    n_models = len(MODEL_ORDER)
    width = min(0.28, 0.8 / max(n_models, 1))

    for axis_index, (prefix, title) in enumerate([("train", "Training Set"), ("test", "Test Set")]):
        ax = axes[axis_index]
        x_positions = np.array(
            [idx * (n_models * width + dgp_spacing) for idx in range(len(DGP_ORDER))]
        )

        for model_index, model_name in enumerate(MODEL_ORDER):
            means: list[float] = []
            stds: list[float] = []
            for dgp_name in DGP_ORDER:
                row = coverage[
                    (coverage["dgp"] == dgp_name) & (coverage["model"] == model_name)
                ]
                if row.empty:
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                means.append(float(row[f"{prefix}_cov_mean"].iloc[0]))
                stds.append(float(row[f"{prefix}_cov_std"].iloc[0]))

            offset = (model_index - (n_models - 1) / 2) * width
            ax.bar(
                x_positions + offset,
                means,
                width,
                yerr=stds,
                label=MODEL_LABELS[model_name],
                color=COLORS[model_name],
                capsize=3,
                alpha=0.85,
            )

        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=2, label="Nominal (90%)")
        ax.set_xlabel("Data Generating Process")
        ax.set_ylabel("Coverage Rate")
        ax.set_title(f"90% Prediction Interval Coverage ({title})")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([DGP_LABELS[dgp_name] for dgp_name in DGP_ORDER], rotation=15)
        ax.set_ylim(0.5, 1.05)

        if axis_index == 0:
            ax.legend(title="Model", loc="lower left", fontsize=8)

    fig.text(
        0.5,
        -0.02,
        "Error bars: ±1 std across random seeds",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    output_path = output_dir / "fig3_coverage_comparison.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_publication_figures(
    *,
    output_dir: str | Path,
    results_csv: str | Path | None = None,
    artifact_root: str | Path | None = None,
    summary_paths: Sequence[str | Path] | None = None,
    figure_ids: Sequence[str] = DEFAULT_FIGURE_IDS,
) -> dict[str, Path]:
    """Generate the selected benchmark paper figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = tuple(dict.fromkeys(figure_ids))
    unknown = sorted(set(requested) - set(DEFAULT_FIGURE_IDS))
    if unknown:
        raise ValueError(f"Unknown figure ids: {', '.join(unknown)}")

    generated: dict[str, Path] = {}
    if "fig0" in requested:
        generated["fig0"] = generate_graphical_model_figure(output_dir)

    data_figures = [figure_id for figure_id in requested if figure_id != "fig0"]
    if not data_figures:
        return generated

    df = load_synthetic_results(
        results_csv=results_csv,
        artifact_root=artifact_root,
        summary_paths=summary_paths,
    )

    generators = {
        "fig1": generate_elpd_comparison_figure,
        "fig2": generate_mape_comparison_figure,
        "fig3": generate_coverage_figure,
        "fig4": generate_convergence_heatmap_figure,
    }
    for figure_id in data_figures:
        generated[figure_id] = generators[figure_id](df, output_dir)

    return generated


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory to receive fig0/1/2/3/4 outputs.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Raw synthetic benchmark CSV to visualize.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Benchmark artifact root containing synthetic/*/*_seed*_summary.json.",
    )
    parser.add_argument(
        "--figure",
        dest="figure_ids",
        action="append",
        choices=DEFAULT_FIGURE_IDS,
        help="Generate only the selected figure id. Repeat to request multiple figures.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    generated = generate_publication_figures(
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        artifact_root=args.artifact_root,
        figure_ids=args.figure_ids or DEFAULT_FIGURE_IDS,
    )
    for figure_id, path in generated.items():
        print(f"{figure_id}: {path}")


if __name__ == "__main__":
    main()
