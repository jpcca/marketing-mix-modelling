#!/usr/bin/env python3
"""Data visualization script for Conjura MMM dataset.

This script generates exploratory visualizations to understand the dataset structure,
distributions, and relationships between variables.

Usage:
    python data/visualize_data.py [--output-dir OUTPUT_DIR]

Outputs PNG files to data/figures/ by default.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configuration
SPEND_COLUMNS = [
    "google_paid_search_spend",
    "google_shopping_spend",
    "google_pmax_spend",
    "google_display_spend",
    "google_video_spend",
    "meta_facebook_spend",
    "meta_instagram_spend",
    "meta_other_spend",
    "tiktok_spend",
]

TARGET_COLUMNS = [
    "first_purchases",
    "all_purchases",
    "first_purchases_original_price",
    "all_purchases_original_price",
]

TRAFFIC_COLUMNS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
]


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the Conjura MMM dataset."""
    df = pd.read_csv(data_path)
    df["date_day"] = pd.to_datetime(df["date_day"])
    return df


def plot_data_overview(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot data overview: missing values and basic statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Missing values heatmap
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) > 0:
        ax = axes[0]
        missing_pct.head(20).plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Missing %")
        ax.set_title("Top 20 Columns with Missing Values")
        ax.invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes[0].set_title("Missing Values")

    # Data shape and basic info
    ax = axes[1]
    info_text = (
        f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n\n"
        f"Date Range: {df['date_day'].min().date()} to {df['date_day'].max().date()}\n\n"
        f"Unique Timeseries: {df['mmm_timeseries_id'].nunique()}\n"
        f"Unique Organizations: {df['organisation_id'].nunique()}\n"
        f"Unique Territories: {df['territory_name'].nunique()}\n\n"
        f"Verticals: {df['organisation_vertical'].nunique()}\n"
        f"Currencies: {df['currency_code'].nunique()}"
    )
    ax.text(0.1, 0.5, info_text, fontsize=12, family="monospace", va="center")
    ax.axis("off")
    ax.set_title("Dataset Overview")

    plt.tight_layout()
    plt.savefig(output_dir / "01_data_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_data_overview.png")


def plot_spend_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot distribution of advertising spend across channels."""
    # Calculate total spend per channel
    spend_totals = df[SPEND_COLUMNS].sum().sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Total spend by channel
    ax = axes[0]
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(spend_totals)))
    spend_totals.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Total Spend")
    ax.set_title("Total Advertising Spend by Channel")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x / 1e6:.1f}M"))

    # Spend distribution (log scale)
    ax = axes[1]
    spend_data = df[SPEND_COLUMNS].replace(0, np.nan).melt(var_name="Channel", value_name="Spend")
    spend_data = spend_data.dropna()
    sns.boxplot(data=spend_data, x="Spend", y="Channel", ax=ax, orient="h")
    ax.set_xscale("log")
    ax.set_xlabel("Daily Spend (log scale)")
    ax.set_title("Daily Spend Distribution by Channel")

    plt.tight_layout()
    plt.savefig(output_dir / "02_spend_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_spend_distribution.png")


def plot_time_series_sample(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot time series for a sample timeseries."""
    # Pick first timeseries
    sample_ts = df["mmm_timeseries_id"].iloc[0]
    sample_df = df[df["mmm_timeseries_id"] == sample_ts].sort_values(by="date_day")  # pyright: ignore[reportCallIssue]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Purchases over time
    ax = axes[0]
    ax.plot(sample_df["date_day"], sample_df["first_purchases"], label="First Purchases", alpha=0.8)
    ax.plot(sample_df["date_day"], sample_df["all_purchases"], label="All Purchases", alpha=0.8)
    ax.set_ylabel("Purchases")
    ax.set_title(f"Sample Timeseries: {sample_ts[:16]}...")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total spend over time
    ax = axes[1]
    total_spend = sample_df[SPEND_COLUMNS].sum(axis=1)
    ax.fill_between(sample_df["date_day"], total_spend, alpha=0.7, label="Total Spend")
    ax.set_ylabel("Total Daily Spend")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spend breakdown by platform
    ax = axes[2]
    google_spend = sample_df[[c for c in SPEND_COLUMNS if "google" in c]].sum(axis=1)
    meta_spend = sample_df[[c for c in SPEND_COLUMNS if "meta" in c]].sum(axis=1)
    tiktok_spend = sample_df["tiktok_spend"].fillna(0)

    ax.stackplot(
        sample_df["date_day"],
        google_spend,
        meta_spend,
        tiktok_spend,
        labels=["Google", "Meta", "TikTok"],
        alpha=0.7,
    )
    ax.set_ylabel("Spend by Platform")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "03_time_series_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_time_series_sample.png")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation matrix between spend and target variables."""
    # Select columns for correlation
    corr_cols = SPEND_COLUMNS + TARGET_COLUMNS
    corr_df = df[corr_cols].copy()

    # Shorten column names for display
    rename_map = {
        "google_paid_search_spend": "G_Search",
        "google_shopping_spend": "G_Shopping",
        "google_pmax_spend": "G_PMax",
        "google_display_spend": "G_Display",
        "google_video_spend": "G_Video",
        "meta_facebook_spend": "M_Facebook",
        "meta_instagram_spend": "M_Instagram",
        "meta_other_spend": "M_Other",
        "tiktok_spend": "TikTok",
        "first_purchases": "1st_Purch",
        "all_purchases": "All_Purch",
        "first_purchases_original_price": "1st_Revenue",
        "all_purchases_original_price": "All_Revenue",
    }
    corr_df = corr_df.rename(columns=rename_map)  # type: ignore[arg-type]

    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlation Matrix: Spend vs Target Variables")

    plt.tight_layout()
    plt.savefig(output_dir / "04_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_correlation_matrix.png")


def plot_vertical_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot breakdown by organization vertical."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Timeseries count by vertical
    ax = axes[0]
    vertical_counts_series = df.groupby("organisation_vertical")["mmm_timeseries_id"].nunique()
    vertical_counts = vertical_counts_series.sort_values(ascending=True)  # pyright: ignore[reportCallIssue]
    vertical_counts.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of Timeseries")
    ax.set_title("Timeseries Count by Vertical")

    # Average daily spend by vertical
    ax = axes[1]
    total_spend_col = df[SPEND_COLUMNS].sum(axis=1)
    df_temp = df.copy()
    df_temp["total_spend"] = total_spend_col
    avg_spend_series = df_temp.groupby("organisation_vertical")["total_spend"].mean()
    avg_spend = avg_spend_series.sort_values(ascending=True)  # pyright: ignore[reportCallIssue]
    avg_spend.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_xlabel("Average Daily Spend")
    ax.set_title("Average Daily Spend by Vertical")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig(output_dir / "05_vertical_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_vertical_breakdown.png")


def plot_channel_usage(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot channel usage patterns across timeseries."""
    # Calculate which channels are used (non-zero spend) per timeseries
    channel_usage = df.groupby("mmm_timeseries_id")[SPEND_COLUMNS].apply(lambda x: (x > 0).any())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Channel usage rate
    ax = axes[0]
    usage_rate_series = channel_usage.mean()
    usage_rate = usage_rate_series.sort_values(ascending=True) * 100  # pyright: ignore[reportAttributeAccessIssue]
    colors = [
        "#4285F4" if "google" in c else "#0866FF" if "meta" in c else "#000000"
        for c in usage_rate.index
    ]
    usage_rate.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("% of Timeseries Using Channel")
    ax.set_title("Channel Usage Rate")
    ax.axvline(x=50, color="red", linestyle="--", alpha=0.5)

    # Number of channels used per timeseries
    ax = axes[1]
    channels_per_ts = channel_usage.sum(axis=1)
    channels_per_ts.hist(bins=range(0, 11), ax=ax, color="teal", edgecolor="white", rwidth=0.8)
    ax.set_xlabel("Number of Channels Used")
    ax.set_ylabel("Number of Timeseries")
    ax.set_title("Distribution of Channel Count per Timeseries")
    ax.set_xticks(range(0, 10))

    plt.tight_layout()
    plt.savefig(output_dir / "06_channel_usage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 06_channel_usage.png")


def plot_traffic_sources(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot organic traffic source distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Total traffic by source
    ax = axes[0]
    traffic_totals = df[TRAFFIC_COLUMNS].sum().sort_values(ascending=True)
    traffic_totals.plot(kind="barh", ax=ax, color="forestgreen")
    ax.set_xlabel("Total Clicks")
    ax.set_title("Total Traffic by Source")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1e6:.1f}M"))

    # Traffic composition (pie chart)
    ax = axes[1]
    traffic_totals.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
    ax.set_ylabel("")
    ax.set_title("Traffic Source Composition")

    plt.tight_layout()
    plt.savefig(output_dir / "07_traffic_sources.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_traffic_sources.png")


def plot_spend_vs_purchases(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot relationship between spend and purchases."""
    # Aggregate by timeseries
    agg_df = df.groupby("mmm_timeseries_id").agg(
        {
            "first_purchases": "sum",
            "all_purchases": "sum",
            **{col: "sum" for col in SPEND_COLUMNS},
        }
    )
    agg_df = pd.DataFrame(agg_df)  # Ensure DataFrame type for pyright
    agg_df["total_spend"] = agg_df[SPEND_COLUMNS].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Total spend vs first purchases
    ax = axes[0]
    ax.scatter(agg_df["total_spend"], agg_df["first_purchases"], alpha=0.5, s=30)
    ax.set_xlabel("Total Spend")
    ax.set_ylabel("Total First Purchases")
    ax.set_title("Total Spend vs First Purchases (by Timeseries)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Total spend vs all purchases
    ax = axes[1]
    ax.scatter(agg_df["total_spend"], agg_df["all_purchases"], alpha=0.5, s=30, color="orange")
    ax.set_xlabel("Total Spend")
    ax.set_ylabel("Total All Purchases")
    ax.set_title("Total Spend vs All Purchases (by Timeseries)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "08_spend_vs_purchases.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 08_spend_vs_purchases.png")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize Conjura MMM dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/conjura_mmm_data.csv"),
        help="Path to the CSV data file",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    print(f"Loaded {len(df):,} rows")

    print("\nGenerating visualizations...")
    plot_data_overview(df, args.output_dir)
    plot_spend_distribution(df, args.output_dir)
    plot_time_series_sample(df, args.output_dir)
    plot_correlation_matrix(df, args.output_dir)
    plot_vertical_breakdown(df, args.output_dir)
    plot_channel_usage(df, args.output_dir)
    plot_traffic_sources(df, args.output_dir)
    plot_spend_vs_purchases(df, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
