"""Data loader for Conjura MMM dataset.

Provides utilities to load and preprocess real marketing mix data
for model validation. Each organisation is treated as an independent
time series (no cross-organisation data sharing).

Usage:
    from hill_mmm.data_loader import list_timeseries, load_timeseries

    # Discover available time series
    ts_info = list_timeseries("data/conjura_mmm_data.csv")

    # Load a specific organisation's data
    data = load_timeseries(
        "data/conjura_mmm_data.csv",
        TimeSeriesConfig(organisation_id="org_123")
    )
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# All spend columns in the Conjura dataset
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

# Target variable options
TARGET_COLUMNS = [
    "first_purchases",
    "first_purchases_units",
    "first_purchases_original_price",
    "all_purchases",
    "all_purchases_units",
    "all_purchases_original_price",
]


@dataclass
class TimeSeriesConfig:
    """Configuration for loading a specific time series.

    Attributes:
        organisation_id: Organisation to load (required for load_timeseries)
        territory: Territory filter (default: "All Territories")
        target_col: Target variable column name
        spend_cols: Spend columns to include (None = auto-detect non-zero)
        aggregate_spend: If True, sum all spend into single x (Phase 1)
        min_nonzero_ratio: Minimum ratio of non-zero days to include a channel
        min_series_length: Minimum time series length required
    """

    organisation_id: str | None = None
    territory: str = "All Territories"
    target_col: str = "all_purchases"
    spend_cols: list[str] | None = None
    aggregate_spend: bool = True
    min_nonzero_ratio: float = 0.1
    min_series_length: int = 100


@dataclass
class LoadedData:
    """Loaded and preprocessed time series data.

    Attributes:
        x: (T,) aggregated spend or (T, C) multi-channel spend
        y: (T,) target response values
        dates: (T,) date index as numpy datetime64
        meta: Metadata about the loaded series
    """

    x: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    meta: dict = field(default_factory=dict)


def list_timeseries(
    csv_path: str | Path,
    min_length: int = 100,
) -> pd.DataFrame:
    """List available time series with summary statistics.

    Scans the dataset to identify unique organisation/territory combinations
    and their key characteristics.

    Args:
        csv_path: Path to conjura_mmm_data.csv
        min_length: Minimum series length to include

    Returns:
        DataFrame with columns:
            - organisation_id
            - territory_name
            - organisation_vertical
            - n_days: Number of observations
            - date_min, date_max: Date range
            - total_spend: Sum of all ad spend
            - total_target: Sum of all_purchases
            - n_active_channels: Number of channels with >10% non-zero days
            - active_channels: List of active channel names
    """
    df = pd.read_csv(csv_path, parse_dates=["date_day"])

    results = []

    for key, group in df.groupby(["organisation_id", "territory_name"]):
        org_id, territory = key[0], key[1]  # type: ignore[index]
        n_days = len(group)

        if n_days < min_length:
            continue

        # Calculate active channels
        active_channels = []
        for col in SPEND_COLUMNS:
            if col in group.columns:
                nonzero_ratio = (group[col].fillna(0) > 0).mean()
                if nonzero_ratio > 0.1:
                    active_channels.append(col)

        # Sum spend and target
        spend_cols_present = [c for c in SPEND_COLUMNS if c in group.columns]
        total_spend = group[spend_cols_present].fillna(0).sum().sum()
        total_target = group["all_purchases"].sum() if "all_purchases" in group.columns else 0

        results.append(
            {
                "organisation_id": org_id,
                "territory_name": territory,
                "organisation_vertical": group["organisation_vertical"].iloc[0],
                "n_days": n_days,
                "date_min": group["date_day"].min(),
                "date_max": group["date_day"].max(),
                "total_spend": total_spend,
                "total_target": total_target,
                "n_active_channels": len(active_channels),
                "active_channels": active_channels,
            }
        )

    result_df = pd.DataFrame(results)
    if len(result_df) == 0:
        return result_df
    return result_df.sort_values(by="n_days", ascending=False).reset_index(drop=True)


def get_active_channels(
    df: pd.DataFrame,
    min_nonzero_ratio: float = 0.1,
) -> list[str]:
    """Identify channels with sufficient non-zero data.

    Args:
        df: DataFrame with spend columns
        min_nonzero_ratio: Minimum ratio of non-zero observations

    Returns:
        List of column names with sufficient data
    """
    active = []
    for col in SPEND_COLUMNS:
        if col not in df.columns:
            continue
        nonzero_ratio = (df[col].fillna(0) > 0).mean()
        if nonzero_ratio >= min_nonzero_ratio:
            active.append(col)
    return active


def load_timeseries(
    csv_path: str | Path,
    config: TimeSeriesConfig,
) -> LoadedData:
    """Load a single organisation's time series.

    Args:
        csv_path: Path to conjura_mmm_data.csv
        config: Configuration for loading

    Returns:
        LoadedData with x, y, dates, and metadata

    Raises:
        ValueError: If organisation_id not specified or not found
        ValueError: If series too short or no active channels
    """
    if config.organisation_id is None:
        raise ValueError("organisation_id must be specified in config")

    df = pd.read_csv(csv_path, parse_dates=["date_day"])

    # Filter to specific organisation and territory
    mask = (df["organisation_id"] == config.organisation_id) & (
        df["territory_name"] == config.territory
    )
    org_df = df[mask].copy()

    if len(org_df) == 0:
        raise ValueError(
            f"No data found for organisation_id={config.organisation_id}, "
            f"territory={config.territory}"
        )

    # Sort by date and ensure consecutive
    org_df = org_df.sort_values(by="date_day").reset_index(drop=True)  # type: ignore[call-overload]

    if len(org_df) < config.min_series_length:
        raise ValueError(
            f"Series too short: {len(org_df)} days < {config.min_series_length} minimum"
        )

    # Determine spend columns to use
    if config.spend_cols is not None:
        spend_cols = config.spend_cols
    else:
        spend_cols = get_active_channels(org_df, config.min_nonzero_ratio)

    if len(spend_cols) == 0:
        raise ValueError(
            f"No active spend channels found for organisation {config.organisation_id}"
        )

    # Extract data
    dates = org_df["date_day"].values
    y = org_df[config.target_col].fillna(0).values.astype(np.float32)

    # Extract spend
    spend_df = org_df[spend_cols].fillna(0)

    if config.aggregate_spend:
        # Phase 1: Sum all channels into single x
        x = spend_df.sum(axis=1).values.astype(np.float32)
    else:
        # Phase 2+: Multi-channel x
        x = spend_df.values.astype(np.float32)

    # Build metadata
    meta = {
        "organisation_id": config.organisation_id,
        "territory": config.territory,
        "organisation_vertical": org_df["organisation_vertical"].iloc[0],
        "currency_code": org_df["currency_code"].iloc[0],
        "target_col": config.target_col,
        "spend_cols": spend_cols,
        "n_channels": len(spend_cols),
        "aggregated": config.aggregate_spend,
        "T": len(org_df),
        "date_min": str(dates[0])[:10],
        "date_max": str(dates[-1])[:10],
        "total_spend": float(spend_df.sum().sum()),
        "total_target": float(y.sum()),
        "spend_nonzero_ratio": float((x > 0).mean()) if config.aggregate_spend else None,
    }

    return LoadedData(x=x, y=y, dates=np.asarray(dates), meta=meta)


def load_real_data(csv_path: str | Path) -> pd.DataFrame:
    """Load real data in experiment-ready format.

    Returns a DataFrame with standardized columns for paper experiments:
    - organization_id: Organization identifier
    - spend: Aggregated daily ad spend
    - revenue: Target metric (all_purchases)
    - date: Date of observation

    Args:
        csv_path: Path to conjura_mmm_data.csv

    Returns:
        DataFrame with columns [organization_id, date, spend, revenue]
    """
    df = pd.read_csv(csv_path, parse_dates=["date_day"])

    # Use "All Territories" for aggregated view
    df = df[df["territory_name"] == "All Territories"].copy()

    # Aggregate spend across all channels
    spend_cols = [c for c in SPEND_COLUMNS if c in df.columns]
    df["spend"] = df[spend_cols].fillna(0).sum(axis=1)  # type: ignore[union-attr]

    # Use all_purchases as target
    df["revenue"] = df["all_purchases"].fillna(0)  # type: ignore[union-attr]

    # Rename and select columns
    result = df[["organisation_id", "date_day", "spend", "revenue"]].copy()
    result = result.rename(  # type: ignore[union-attr]
        columns={
            "organisation_id": "organization_id",
            "date_day": "date",
        }
    )

    # Sort by org and date
    result = result.sort_values(["organization_id", "date"]).reset_index(drop=True)

    return result


def select_representative_timeseries(
    df: pd.DataFrame,
    n_timeseries: int = 5,
    selection_criteria: str = "most_data",
    min_length: int = 200,
) -> pd.DataFrame:
    """Select representative time series from loaded data.

    Args:
        df: DataFrame from load_real_data()
        n_timeseries: Number of organizations to select
        selection_criteria: "most_data" or "stratified"
        min_length: Minimum observations required

    Returns:
        DataFrame filtered to selected organizations
    """
    # Count observations per organization
    org_counts = df.groupby("organization_id").size().reset_index(name="n_obs")  # type: ignore[call-overload]
    org_counts = org_counts[org_counts["n_obs"] >= min_length]

    if len(org_counts) == 0:
        raise ValueError(f"No organizations with >= {min_length} observations")

    if selection_criteria == "most_data":
        # Select top N by observation count
        selected_orgs = org_counts.nlargest(n_timeseries, "n_obs")["organization_id"].tolist()  # type: ignore[call-overload]
    else:
        # Random selection
        selected_orgs = org_counts.sample(min(n_timeseries, len(org_counts)))[
            "organization_id"
        ].tolist()

    result_df: pd.DataFrame = df[df["organization_id"].isin(selected_orgs)].copy()  # type: ignore[assignment]
    return result_df
