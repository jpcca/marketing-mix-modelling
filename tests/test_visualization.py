"""Visualization tests for paper figures.

Generates publication-quality figures for the paper and saves them to paper/figures/.
Each test creates figures for experimental results only.

Figure categories:
1. Experimental results (ELPD comparison, convergence, effective K recovery)
2. Model fit visualization (response curves, prediction vs actual)
3. MCMC trace plots for all DGP × Model combinations
"""

from pathlib import Path
from typing import Literal

import daft
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hill_mmm import (
    DGPConfig,
    compute_prior_config,
    generate_data,
    hill,
    model_hill_mixture_hierarchical_reparam,
)
from hill_mmm.inference import run_inference

# Aliases for backward compatibility (all mixture models now use unified hierarchical model)
model_hill_mixture = model_hill_mixture_hierarchical_reparam
model_hill_mixture_k2 = model_hill_mixture_hierarchical_reparam  # K=2 passed at call time
model_hill_mixture_sparse = model_hill_mixture_hierarchical_reparam  # K=5 passed at call time

# =============================================================================
# Configuration
# =============================================================================

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark"
RESULTS_SUMMARY_CSV = RESULTS_DIR / "synthetic_20260210_152336_summary.csv"
RESULTS_CSV = RESULTS_DIR / "synthetic_20260210_152336.csv"

# DGP ordering for plots
DGP_ORDER = ["single", "mixture_k2", "mixture_k3", "mixture_k5"]
DGP_K_TRUE = {"single": 1, "mixture_k2": 2, "mixture_k3": 3, "mixture_k5": 5}
DGP_LABELS = {
    "single": "Single (K=1)",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
    "mixture_k5": "Mixture (K=5)",
}
MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3", "mixture_k5"]
MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
    "mixture_k5": "Mixture (K=5)",
}

# Color schemes for consistent styling
COLORS = {
    "single_hill": "#1f77b4",
    "mixture_k2": "#9467bd",  # Purple for K=2
    "mixture_k3": "#ff7f0e",
    "mixture_k5": "#2ca02c",
}
COLORS_LIST = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Matplotlib style settings for publication
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


# =============================================================================
# Helper Functions
# =============================================================================


def load_results_summary() -> pd.DataFrame:
    """Load multi-index results summary CSV.

    The summary CSV has a multi-index header format:
    - Rows 0-1: Column multi-index (metric names, mean/std)
    - Row 2: Index names (dgp, K_true, model) - needs to be skipped
    - Row 3+: Data
    """
    df = pd.read_csv(RESULTS_SUMMARY_CSV, header=[0, 1], index_col=[0, 1, 2], skiprows=[2])
    df.index.names = ["dgp", "K_true", "model"]
    return df


def load_results() -> pd.DataFrame:
    """Load raw results CSV."""
    return pd.read_csv(RESULTS_CSV)


# =============================================================================
# Figure 0: Graphical Model (Plate Notation)
# =============================================================================


def test_graphical_model(output_dir: Path) -> None:
    """Figure 0: Bayesian graphical model with plate notation.

    Creates a publication-quality graphical model showing the hierarchical
    structure of the Hill Mixture Model using standard plate notation.
    """
    # Create PGM with appropriate aspect ratio - wider spacing
    pgm = daft.PGM(dpi=150)

    # Spacing constants for better layout
    # Y levels: priors=5, segment=3, time=1.5, outcome=0
    # X spacing: increased from 1 to 1.5 between nodes
    Y_PRIOR = 5
    Y_SEGMENT = 3
    Y_TIME = 1.5
    Y_OUTCOME = 0

    # --- Top level priors (shared parameters) - wider spacing ---
    pgm.add_node("alpha", r"$\alpha$", 1, Y_PRIOR)
    pgm.add_node("mu0", r"$\mu_0$", 2.5, Y_PRIOR)  # type: ignore[arg-type]
    pgm.add_node("beta", r"$\beta$", 4, Y_PRIOR)
    pgm.add_node("sigma", r"$\sigma$", 5.5, Y_PRIOR)  # type: ignore[arg-type]
    pgm.add_node("pi", r"$\boldsymbol{\pi}$", 7, Y_PRIOR)

    # --- Segment-specific parameters (Plate K) - wider spacing ---
    # Note: daft's type stubs incorrectly declare x/y as int, but float works at runtime
    pgm.add_node("A_k", r"$A_k$", 7, Y_SEGMENT)  # type: ignore[arg-type]
    pgm.add_node("lambda_k", r"$\lambda_k$", 8.5, Y_SEGMENT)  # type: ignore[arg-type]
    pgm.add_node("n_k", r"$n_k$", 10, Y_SEGMENT)  # type: ignore[arg-type]

    # --- Time-varying nodes (Plate T) - wider spacing ---
    pgm.add_node("x_t", r"$x_t$", 1, Y_TIME, observed=True)  # type: ignore[arg-type]
    pgm.add_node("s_t", r"$s_t$", 2.5, Y_TIME)  # type: ignore[arg-type]
    pgm.add_node("f_k_t", r"$f_k(s_t)$", 5.5, Y_TIME)  # type: ignore[arg-type]
    pgm.add_node("y_t", r"$y_t$", 5.5, Y_OUTCOME, observed=True)  # type: ignore[arg-type]

    # --- Edges ---
    # Adstock transformation
    pgm.add_edge("x_t", "s_t")
    pgm.add_edge("alpha", "s_t")

    # Hill function
    pgm.add_edge("s_t", "f_k_t")
    pgm.add_edge("A_k", "f_k_t")
    pgm.add_edge("lambda_k", "f_k_t")
    pgm.add_edge("n_k", "f_k_t")

    # Observation model
    pgm.add_edge("f_k_t", "y_t")
    pgm.add_edge("mu0", "y_t")
    pgm.add_edge("beta", "y_t")
    pgm.add_edge("sigma", "y_t")
    pgm.add_edge("pi", "y_t")

    # --- Plates - adjusted for new coordinates ---
    pgm.add_plate(
        [6.3, 2.4, 4.4, 1.2],
        label=r"$k = 1, \ldots, K$" + "\n(Latent Segments)",
        shift=-0.1,  # type: ignore[arg-type]
    )
    pgm.add_plate(
        [0.3, -0.6, 6.0, 2.8],
        label=r"$t = 1, \ldots, T$" + "\n(Time Periods)",
        shift=-0.1,  # type: ignore[arg-type]
    )

    # --- Text annotations for nodes - positioned to avoid overlaps ---
    # Top-level priors (above nodes)
    pgm.add_text(1, Y_PRIOR + 0.7, "Adstock\nDecay", fontsize=8)  # type: ignore[arg-type]
    pgm.add_text(2.5, Y_PRIOR + 0.7, "Baseline", fontsize=8)  # type: ignore[arg-type]
    pgm.add_text(4, Y_PRIOR + 0.7, "Trend", fontsize=8)  # type: ignore[arg-type]
    pgm.add_text(5.5, Y_PRIOR + 0.7, "Noise", fontsize=8)  # type: ignore[arg-type]
    pgm.add_text(7, Y_PRIOR + 0.7, "Mixture\nWeights", fontsize=8)  # type: ignore[arg-type]

    # Segment-specific parameters (above nodes)
    pgm.add_text(7, Y_SEGMENT + 0.7, "Max\nEffect", fontsize=7)  # type: ignore[arg-type]
    pgm.add_text(8.5, Y_SEGMENT + 0.7, "Half-\nSaturation", fontsize=7)  # type: ignore[arg-type]
    pgm.add_text(10, Y_SEGMENT + 0.7, "Steepness", fontsize=7)  # type: ignore[arg-type]

    # Time-varying nodes (below nodes to avoid edges)
    pgm.add_text(1, Y_TIME - 0.7, "Spend", fontsize=8)  # type: ignore[arg-type]
    pgm.add_text(2.5, Y_TIME - 0.7, "Adstocked\nSpend", fontsize=7)  # type: ignore[arg-type]
    pgm.add_text(6.5, Y_TIME, "Hill\nResponse", fontsize=7)  # type: ignore[arg-type]
    pgm.add_text(5.5, Y_OUTCOME - 0.7, "Observed\nOutcome", fontsize=8)  # type: ignore[arg-type]

    # Render and save
    pgm.render()
    plt.savefig(output_dir / "fig0_graphical_model.png", dpi=300, bbox_inches="tight")
    plt.close("all")


# =============================================================================
# Figure 1: ELPD-LOO Comparison (from results)
# =============================================================================


def test_elpd_comparison(output_dir: Path) -> None:
    """Figure 1: ELPD-LOO comparison across DGPs and models.

    Creates a grouped bar chart comparing ELPD-LOO values for 3 models
    across 4 DGP scenarios with error bars.
    """
    df = load_results_summary()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(DGP_ORDER))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for dgp in DGP_ORDER:
            k_true = DGP_K_TRUE[dgp]
            try:
                row = df.loc[(dgp, k_true, model)]
                means.append(row[("elpd_loo", "mean")])
                stds.append(row[("elpd_loo", "std")])
            except KeyError:
                means.append(np.nan)
                stds.append(np.nan)

        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            capsize=3,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Data Generating Process")
    ax.set_ylabel("ELPD-LOO")
    ax.set_title("Model Comparison: Expected Log Pointwise Predictive Density")
    ax.set_xticks(x)
    ax.set_xticklabels([DGP_LABELS[dgp] for dgp in DGP_ORDER])
    ax.legend(title="Model")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_elpd_comparison.png")
    plt.close("all")


# =============================================================================
# Figure 2: ELPD Delta (Improvement over Single Hill)
# =============================================================================


def test_elpd_delta(output_dir: Path) -> None:
    """Figure 2: ELPD improvement over Single Hill baseline.

    Forest plot showing the difference in ELPD-LOO relative to Single Hill.
    """
    df = load_results_summary()

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = 0
    y_ticks = []
    y_labels = []

    for dgp in DGP_ORDER:
        k_true = DGP_K_TRUE[dgp]

        # Get baseline (single_hill)
        try:
            baseline = df.loc[(dgp, k_true, "single_hill")][("elpd_loo", "mean")]
        except KeyError:
            continue

        for model in ["mixture_k3", "sparse_k5"]:
            try:
                row = df.loc[(dgp, k_true, model)]
                delta = row[("elpd_loo", "mean")] - baseline
                # Approximate combined std (simplified)
                std = row[("elpd_loo", "std")]

                color = COLORS[model]
                ax.barh(y_pos, delta, xerr=std, color=color, alpha=0.8, capsize=3, height=0.6)

                y_ticks.append(y_pos)
                y_labels.append(f"{DGP_LABELS[dgp]}\n{MODEL_LABELS[model]}")
                y_pos += 1
            except KeyError:
                pass

        y_pos += 0.5  # Gap between DGPs

    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("ΔELPD-LOO (vs Single Hill)")
    ax.set_title("ELPD Improvement Over Single Hill Baseline")

    # Add significance annotation
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color="green")
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color="red")
    ax.text(ax.get_xlim()[1] * 0.7, y_pos - 1, "Better", fontsize=10, color="green")
    ax.text(ax.get_xlim()[0] * 0.7, y_pos - 1, "Worse", fontsize=10, color="red")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_elpd_delta.png")
    plt.close("all")


# =============================================================================
# Figure 3: Convergence Heatmap
# =============================================================================


def test_convergence_heatmap(output_dir: Path) -> None:
    """Figure 3: Convergence rate heatmap (DGP x Model).

    Creates a heatmap showing convergence rates (proportion of seeds
    with converged=True) for each DGP-model combination.
    """
    df = load_results()

    # Calculate convergence rate per DGP-model
    conv_rates = df.groupby(["dgp", "model"])["converged"].mean().unstack(fill_value=0)

    # Reorder rows and columns
    conv_rates = conv_rates.reindex(index=DGP_ORDER, columns=MODEL_ORDER)  # type: ignore[call-arg]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Create heatmap using imshow
    im = ax.imshow(conv_rates.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Convergence Rate")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticks(np.arange(len(DGP_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_yticklabels([DGP_LABELS[dgp] for dgp in DGP_ORDER])

    # Add text annotations
    for i in range(len(DGP_ORDER)):
        for j in range(len(MODEL_ORDER)):
            val = conv_rates.values[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=text_color, fontsize=12)

    ax.set_xlabel("Model")
    ax.set_ylabel("Data Generating Process")
    ax.set_title("MCMC Convergence Rate by DGP and Model")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_convergence_heatmap.png")
    plt.close("all")


def test_convergence_rate_threshold() -> None:
    """Test that all DGP-model combinations meet minimum convergence threshold.

    Asserts that every cell in the convergence heatmap has at least 60%
    convergence rate. Fails if any DGP-model combination falls below this.
    Note: Some model-DGP combinations (especially sparse models with single DGP)
    have known convergence challenges, hence the 60% threshold.
    """
    MIN_CONVERGENCE_RATE = 0.60

    df = load_results()

    # Calculate convergence rate per DGP-model combination
    conv_rates = df.groupby(["dgp", "model"])["converged"].mean().unstack(fill_value=0)
    conv_rates = conv_rates.reindex(index=DGP_ORDER, columns=MODEL_ORDER)  # type: ignore[call-arg]

    # Check each cell
    failures = []
    for dgp in DGP_ORDER:
        for model in MODEL_ORDER:
            rate = conv_rates.loc[dgp, model]
            if rate < MIN_CONVERGENCE_RATE:
                failures.append(
                    f"  {DGP_LABELS[dgp]} + {MODEL_LABELS[model]}: {rate:.1%} < {MIN_CONVERGENCE_RATE:.0%}"
                )

    if failures:
        failure_msg = "\n".join(failures)
        pytest.fail(f"Convergence rate below {MIN_CONVERGENCE_RATE:.0%} threshold:\n{failure_msg}")


# =============================================================================
# Figure 4: Effective K Recovery
# =============================================================================


def test_effective_k_recovery(output_dir: Path) -> None:
    """Figure 4: Effective K recovery plot.

    Shows how well each model recovers the true number of components K.
    X-axis: True K, Y-axis: Estimated effective K with error bars.
    Includes diagonal line for perfect recovery.
    """
    df = load_results_summary()

    fig, ax = plt.subplots(figsize=(7, 6))

    k_true_values = [1, 2, 3, 5]
    markers = ["o", "s", "^", "D"]  # 4 markers for 4 models

    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for dgp in DGP_ORDER:
            k_true = DGP_K_TRUE[dgp]
            try:
                row = df.loc[(dgp, k_true, model)]
                means.append(row[("effective_k_mean", "mean")])
                stds.append(row[("effective_k_mean", "std")])
            except KeyError:
                means.append(np.nan)
                stds.append(np.nan)

        ax.errorbar(
            k_true_values,
            means,
            yerr=stds,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            marker=markers[i],
            markersize=10,
            capsize=4,
            linewidth=2,
        )

    # Diagonal line for perfect recovery
    ax.plot([0, 6], [0, 6], "k--", alpha=0.5, label="Perfect Recovery", linewidth=1)

    # Shaded region for model capacity
    ax.axhline(y=3, color="#ff7f0e", linestyle=":", alpha=0.5)
    ax.axhline(y=5, color="#2ca02c", linestyle=":", alpha=0.5)
    ax.text(5.5, 3.1, "K=3 model", fontsize=8, color="#ff7f0e")
    ax.text(5.5, 5.1, "K=5 model", fontsize=8, color="#2ca02c")

    ax.set_xlabel("True K")
    ax.set_ylabel("Effective K (mean ± std)")
    ax.set_title("Recovery of True Number of Components")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xticks(k_true_values)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_effective_k_recovery.png")
    plt.close("all")


# =============================================================================
# Figure 5: Coverage Plot
# =============================================================================


def test_coverage_plot(output_dir: Path) -> None:
    """Figure 5: 90% prediction interval coverage plot.

    Shows train and test coverage for each DGP-model combination,
    with a horizontal line at the nominal 0.9 level.
    """
    df = load_results()

    # Aggregate coverage by DGP and model
    coverage = (
        df.groupby(["dgp", "model"])
        .agg(
            train_cov_mean=("train_coverage_90", "mean"),
            train_cov_std=("train_coverage_90", "std"),
            test_cov_mean=("test_coverage_90", "mean"),
            test_cov_std=("test_coverage_90", "std"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (cov_type, title) in enumerate([("train", "Training Set"), ("test", "Test Set")]):
        ax = axes[ax_idx]
        x = np.arange(len(DGP_ORDER))
        width = 0.25

        for i, model in enumerate(MODEL_ORDER):
            means = []
            stds = []
            for dgp in DGP_ORDER:
                row = coverage[(coverage["dgp"] == dgp) & (coverage["model"] == model)]
                if len(row) > 0:
                    means.append(row[f"{cov_type}_cov_mean"].iloc[0])  # type: ignore[union-attr]
                    stds.append(row[f"{cov_type}_cov_std"].iloc[0])  # type: ignore[union-attr]
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            offset = (i - 1) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=MODEL_LABELS[model],
                color=COLORS[model],
                capsize=3,
                alpha=0.85,
            )

        # Target line at 0.9
        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=2, label="Nominal (90%)")

        ax.set_xlabel("Data Generating Process")
        ax.set_ylabel("Coverage Rate")
        ax.set_title(f"90% Prediction Interval Coverage ({title})")
        ax.set_xticks(x)
        ax.set_xticklabels([DGP_LABELS[dgp] for dgp in DGP_ORDER], rotation=15)
        ax.set_ylim(0.5, 1.05)

        if ax_idx == 0:
            ax.legend(title="Model", loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_coverage.png")
    plt.close("all")


# =============================================================================
# Figure 6: Response Curves - True vs Estimated (Slow - requires MCMC)
# =============================================================================

DGPType = Literal["single", "mixture_k2", "mixture_k3", "mixture_k5"]


def test_response_curves_comparison(output_dir: Path) -> None:
    """Figure 6: Response curves comparison.

    Compares estimated response curves from MCMC samples with
    true underlying curves from the data generating process.
    """
    # Generate data with mixture DGP
    config = DGPConfig(dgp_type="mixture_k3", T=150, seed=42)
    x, y, meta = generate_data(config)

    # Get prior config
    prior_config = compute_prior_config(x, y)

    # Run MCMC (matching run_benchmark.py defaults)
    mcmc = run_inference(
        model_fn=model_hill_mixture,
        x=x,
        y=y,
        seed=42,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        prior_config=prior_config,
        K=3,
    )

    samples = mcmc.get_samples()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # X values for plotting curves
    x_plot = np.linspace(0, meta["s_max"] * 1.2, 200)

    # Panel A: True curves
    ax = axes[0]
    for j in range(meta["K_true"]):
        y_true = hill(jnp.array(x_plot), meta["A_true"][j], meta["k_true"][j], meta["n_true"][j])
        ax.plot(
            x_plot,
            np.array(y_true),
            color=COLORS_LIST[j],
            linewidth=2.5,
            label=f"True K={j + 1} (π={meta['pi_true'][j]:.2f})",
        )
    # Plot effect only (subtract baseline) and color by true component assignment
    y_effect = y - meta["baseline"]
    z_true = meta["z_true"]
    for j in range(meta["K_true"]):
        mask = z_true == j
        ax.scatter(
            meta["s"][mask],
            y_effect[mask],
            alpha=0.4,
            s=20,
            color=COLORS_LIST[j],
            edgecolors="none",
        )
    ax.set_xlabel("Adstocked Spend (s)")
    ax.set_ylabel("Hill Effect")
    ax.set_title("(a) True Response Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Estimated curves with uncertainty
    ax = axes[1]
    A_samples = np.array(samples.get("A", samples.get("A_raw", None)))
    k_samples = np.array(samples.get("k", samples.get("k_raw", None)))
    n_samples = np.array(samples.get("n", samples.get("n_raw", None)))

    if A_samples is not None:
        K_model = A_samples.shape[1] if A_samples.ndim > 1 else 1

        for j in range(K_model):
            if A_samples.ndim > 1:
                # Sample posterior curves
                curves = []
                for idx in range(min(100, len(A_samples))):
                    y_sample = hill(
                        jnp.array(x_plot),
                        A_samples[idx, j],
                        k_samples[idx, j],
                        n_samples[idx, j],
                    )
                    curves.append(np.array(y_sample))
                curves = np.array(curves)

                # Plot mean and credible interval
                mean_curve = curves.mean(axis=0)
                lower = np.percentile(curves, 5, axis=0)
                upper = np.percentile(curves, 95, axis=0)

                ax.fill_between(x_plot, lower, upper, color=COLORS_LIST[j], alpha=0.2)
                ax.plot(
                    x_plot,
                    mean_curve,
                    color=COLORS_LIST[j],
                    linewidth=2,
                    label=f"Estimated K={j + 1}",
                )

    # Plot effect only (subtract baseline) - same scale as curves
    y_effect = y - meta["baseline"]
    ax.scatter(meta["s"], y_effect, alpha=0.3, s=15, color="gray")
    ax.set_xlabel("Adstocked Spend (s)")
    ax.set_ylabel("Hill Effect")
    ax.set_title("(b) Estimated Response Curves (with 90% CI)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig6_response_curves.png")
    plt.close("all")


# =============================================================================
# Figure 7: Mixture Weights Visualization (Slow - requires MCMC)
# =============================================================================


def test_mixture_weights_visualization(output_dir: Path) -> None:
    """Figure 7: Mixture weight recovery visualization.

    Shows posterior distribution of mixture weights for different DGPs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    dgp_configs: list[tuple[DGPType, str]] = [
        ("single", "(a) True K=1"),
        ("mixture_k2", "(b) True K=2"),
        ("mixture_k3", "(c) True K=3"),
        ("mixture_k5", "(d) True K=5"),
    ]

    for ax, (dgp_type, title) in zip(axes, dgp_configs):
        config = DGPConfig(dgp_type=dgp_type, T=100, seed=42)
        x, y, meta = generate_data(config)
        prior_config = compute_prior_config(x, y)

        # Run K=5 mixture model (MCMC params match run_benchmark.py)
        mcmc = run_inference(
            model_fn=model_hill_mixture,
            x=x,
            y=y,
            seed=42,
            num_warmup=1000,
            num_samples=2000,
            num_chains=4,
            prior_config=prior_config,
            K=5,
        )

        samples = mcmc.get_samples()
        pis = np.array(samples["pis"])

        # Box plot of mixture weights
        bp = ax.boxplot(
            pis,
            positions=range(1, 6),
            patch_artist=True,
            widths=0.6,
        )
        for patch, color in zip(bp["boxes"], COLORS_LIST):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add true weights if mixture
        if meta["K_true"] > 1:
            for k, pi_true in enumerate(meta["pi_true"]):
                ax.scatter(
                    k + 1,
                    pi_true,
                    color="red",
                    marker="*",
                    s=150,
                    zorder=5,
                    label="True π" if k == 0 else "",
                )

        ax.set_xlabel("Component")
        ax.set_ylabel("Mixture Weight (π)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1 / 5, color="gray", linestyle="--", alpha=0.5)
        if meta["K_true"] > 1:
            ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_mixture_weights.png")
    plt.close("all")


# =============================================================================
# Figure 8: Prediction vs Actual (Slow - requires MCMC)
# =============================================================================


def test_prediction_vs_actual(output_dir: Path) -> None:
    """Figure 8: Prediction vs actual comparison.

    Shows time series of predictions with uncertainty bands.
    """
    config = DGPConfig(dgp_type="mixture_k3", T=150, seed=42)
    x, y, meta = generate_data(config)
    prior_config = compute_prior_config(x, y)

    # Train/test split
    train_size = 100
    x_train, y_train = x[:train_size], y[:train_size]
    # Note: test data available for future use
    _x_test, _y_test = x[train_size:], y[train_size:]

    # Run MCMC on training data (parameters match run_benchmark.py)
    mcmc = run_inference(
        model_fn=model_hill_mixture_sparse,
        x=x_train,
        y=y_train,
        seed=42,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        prior_config=prior_config,
        K=5,
    )

    samples = mcmc.get_samples()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Get mu_expected samples
    if "mu_expected" in samples:
        mu_samples = np.array(samples["mu_expected"])

        # Panel A: Training fit
        ax = axes[0]
        mu_mean = mu_samples.mean(axis=0)
        mu_lower = np.percentile(mu_samples, 5, axis=0)
        mu_upper = np.percentile(mu_samples, 95, axis=0)

        t_train = np.arange(train_size)
        ax.fill_between(t_train, mu_lower, mu_upper, color="#3498db", alpha=0.3, label="90% CI")
        ax.plot(t_train, mu_mean, color="#3498db", linewidth=2, label="Predicted Mean")
        ax.scatter(t_train, y_train, color="black", s=20, alpha=0.6, label="Observed", zorder=5)
        ax.plot(t_train, meta["mu_true"][:train_size], "--", color="red", label="True μ")

        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Response (y)")
        ax.set_title("(a) Training Set Fit")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Panel B: Residual analysis (inside same condition to ensure mu_mean is bound)
        ax = axes[1]
        residuals = y_train - mu_mean
        ax.scatter(mu_mean, residuals, alpha=0.5, color="#3498db", s=30)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Mean")
        ax.set_ylabel("Residual (y - ŷ)")
        ax.set_title("(b) Residual Plot")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig8_prediction_vs_actual.png")
    plt.close("all")


# =============================================================================
# Summary Figure: All Key Results
# =============================================================================


def test_summary_figure(output_dir: Path) -> None:
    """Summary figure combining key results for paper overview.

    Creates a 2x2 panel figure with:
    - ELPD comparison
    - Convergence heatmap
    - Effective K recovery
    - Coverage
    """
    fig = plt.figure(figsize=(14, 12))

    # Load data
    df_summary = load_results_summary()
    df_raw = load_results()

    # Panel A: ELPD
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(DGP_ORDER))
    width = 0.25
    for i, model in enumerate(MODEL_ORDER):
        means = []
        for dgp in DGP_ORDER:
            k_true = DGP_K_TRUE[dgp]
            try:
                means.append(df_summary.loc[(dgp, k_true, model)][("elpd_loo", "mean")])
            except KeyError:
                means.append(np.nan)
        ax1.bar(
            x + (i - 1) * width,
            means,
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            alpha=0.85,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([DGP_LABELS[d] for d in DGP_ORDER], fontsize=9)
    ax1.set_ylabel("ELPD-LOO")
    ax1.set_title("(a) Model Comparison: ELPD-LOO")
    ax1.legend(fontsize=8)

    # Panel B: Convergence
    ax2 = fig.add_subplot(2, 2, 2)
    conv_rates = df_raw.groupby(["dgp", "model"])["converged"].mean().unstack(fill_value=0)
    conv_rates = conv_rates.reindex(index=DGP_ORDER, columns=MODEL_ORDER)  # type: ignore[call-arg]
    ax2.imshow(conv_rates.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(np.arange(len(MODEL_ORDER)))
    ax2.set_yticks(np.arange(len(DGP_ORDER)))
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax2.set_yticklabels([DGP_LABELS[d] for d in DGP_ORDER], fontsize=9)
    for i in range(len(DGP_ORDER)):
        for j in range(len(MODEL_ORDER)):
            ax2.text(
                j,
                i,
                f"{conv_rates.values[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if conv_rates.values[i, j] < 0.5 else "black",
            )
    ax2.set_title("(b) Convergence Rate")

    # Panel C: Effective K
    ax3 = fig.add_subplot(2, 2, 3)
    k_true_values = [1, 2, 3, 5]
    markers = ["o", "s", "^", "D"]
    for i, model in enumerate(MODEL_ORDER):
        means = []
        for dgp in DGP_ORDER:
            k_true = DGP_K_TRUE[dgp]
            try:
                means.append(df_summary.loc[(dgp, k_true, model)][("effective_k_mean", "mean")])
            except KeyError:
                means.append(np.nan)
        ax3.plot(
            k_true_values,
            means,
            marker=markers[i],
            label=MODEL_LABELS[model],
            color=COLORS[model],
            linewidth=2,
            markersize=8,
        )
    ax3.plot([0, 6], [0, 6], "k--", alpha=0.5, label="Perfect")
    ax3.set_xlabel("True K")
    ax3.set_ylabel("Effective K")
    ax3.set_title("(c) Component Recovery")
    ax3.set_xlim(0, 6)
    ax3.set_ylim(0, 6)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D: Test Coverage
    ax4 = fig.add_subplot(2, 2, 4)
    coverage = df_raw.groupby(["dgp", "model"])["test_coverage_90"].mean().unstack(fill_value=0)
    coverage = coverage.reindex(index=DGP_ORDER, columns=MODEL_ORDER)  # type: ignore[call-arg]
    for i, model in enumerate(MODEL_ORDER):
        ax4.bar(
            x + (i - 1) * width,
            np.asarray(coverage[model].values),
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            alpha=0.85,
        )
    ax4.axhline(y=0.9, color="red", linestyle="--", label="Nominal")
    ax4.set_xticks(x)
    ax4.set_xticklabels([DGP_LABELS[d] for d in DGP_ORDER], fontsize=9)
    ax4.set_ylabel("Coverage")
    ax4.set_title("(d) 90% PI Coverage (Test Set)")
    ax4.set_ylim(0.5, 1.05)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_summary.png")
    plt.close("all")
