"""Build paper-ready figures from real-data benchmark artifacts.

Produces two figures referenced in the real-data section:

  1. ``paper/figures/fig_real_response_panel.png``: 1x3 panel of
     posterior response curves for one representative organisation
     (home_garden, seed 0) across the three models.

  2. ``paper/figures/fig_real_elpd_bar.png``: grouped bar chart of
     mean test ELPD-LOO per (organisation, model) with seed-level
     standard-deviation error bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
REAL_ROOT = FIGURES_ROOT / "real"
RAW_CSV = REPO_ROOT / "results" / "real_benchmark_raw.csv"

# Pick home_garden as the representative organisation — best
# calibrated across all three models and clean K=2 vs K=3 contrast.
PANEL_ORG = "home_garden"
PANEL_SEED = 0
PANEL_MODELS = [
    ("single_hill", "Single Hill"),
    ("mixture_k2", "Mixture (K=2)"),
    ("mixture_k3", "Mixture (K=3)"),
]

# Models ordered for the bar chart.
BAR_MODEL_ORDER = ["single_hill", "mixture_k2", "mixture_k3"]
BAR_MODEL_LABELS = {
    "single_hill": "Single Hill",
    "mixture_k2": "Mixture (K=2)",
    "mixture_k3": "Mixture (K=3)",
}
BAR_DATASET_ORDER = ["beauty_fitness", "home_garden", "toys_hobbies"]
BAR_DATASET_LABELS = {
    "beauty_fitness": "Beauty & Fitness",
    "home_garden": "Home & Garden",
    "toys_hobbies": "Toys & Hobbies",
}


def _crop_title(img: np.ndarray, frac: float = 0.07) -> np.ndarray:
    """Remove the top ``frac`` of the image (auto-generated title)."""
    h = img.shape[0]
    top = int(h * frac)
    return img[top:, :, :]


def build_response_panel() -> Path:
    out = FIGURES_ROOT / "fig_real_response_panel.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (model, label) in zip(axes, PANEL_MODELS):
        png = (
            REAL_ROOT
            / model
            / f"real_{PANEL_ORG}_{model}_seed{PANEL_SEED}_response.png"
        )
        if not png.exists():
            ax.text(0.5, 0.5, f"missing: {png.name}", ha="center", va="center")
            ax.axis("off")
            continue
        img = mpimg.imread(png)
        img = _crop_title(img)
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle(
        f"Posterior response curves — {BAR_DATASET_LABELS[PANEL_ORG]} (seed {PANEL_SEED})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_elpd_bar() -> Path:
    out = FIGURES_ROOT / "fig_real_elpd_bar.png"
    df = pd.read_csv(RAW_CSV)
    agg = (
        df.groupby(["dataset_label", "model"])["elpd_loo"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n_models = len(BAR_MODEL_ORDER)
    width = 0.25
    x = np.arange(len(BAR_DATASET_ORDER))
    colors = ["#4878D0", "#A66BB1", "#EE854A"]

    for i, model in enumerate(BAR_MODEL_ORDER):
        means = []
        stds = []
        for dataset in BAR_DATASET_ORDER:
            row = agg[(agg["dataset_label"] == dataset) & (agg["model"] == model)]
            means.append(row["mean"].iloc[0] if len(row) else np.nan)
            stds.append(row["std"].iloc[0] if len(row) else np.nan)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=BAR_MODEL_LABELS[model],
            color=colors[i],
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BAR_DATASET_LABELS[d] for d in BAR_DATASET_ORDER])
    ax.set_ylabel("ELPD-LOO")
    ax.set_title("Real-data predictive density across organisations and models")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(
        0.01,
        -0.18,
        "Higher is better; error bars: ±1 std across three seeds.",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    panel = build_response_panel()
    bar = build_elpd_bar()
    print(f"Wrote {panel.relative_to(REPO_ROOT)}")
    print(f"Wrote {bar.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
