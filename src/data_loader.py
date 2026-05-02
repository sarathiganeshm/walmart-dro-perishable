"""
M5 Walmart data loader — FOODS_3, CA_1, weekly aggregation.
Loads sales_train_evaluation.csv, calendar.csv, sell_prices.csv from data/.
Filters dept_id='FOODS_3', store_id='CA_1', days d_1577 to d_1941.
Aggregates to weekly demand. Returns train (weeks 1–42) and holdout (weeks 43–52).
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"


def load_m5_weekly(data_dir=DATA_DIR):
    """Load M5 files and return weekly demand series for FOODS_3, CA_1, d_1577-d_1941."""
    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"

    # Check files exist; if not, raise informative error
    for p in [sales_path, calendar_path, prices_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required data file not found: {p}\n"
                "Download M5 files from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
                "and place in the data/ directory."
            )

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)

    # Filter to FOODS_3, CA_1
    mask = (sales["dept_id"] == "FOODS_3") & (sales["store_id"] == "CA_1")
    sales_filtered = sales[mask].copy()

    # Column days d_1577 to d_1941
    day_cols = [f"d_{i}" for i in range(1577, 1942)]
    # Keep only columns that exist (safety check)
    day_cols = [c for c in day_cols if c in sales_filtered.columns]

    demand_matrix = sales_filtered[day_cols].values  # shape: (items, days)
    # Aggregate across all items in FOODS_3/CA_1 to get total daily demand
    daily_demand = demand_matrix.sum(axis=0)  # shape: (365,)

    # Convert to weekly by grouping every 7 days
    n_days = len(daily_demand)
    n_weeks = n_days // 7
    weekly_demand = np.array([daily_demand[i*7:(i+1)*7].sum() for i in range(n_weeks)])

    return weekly_demand


def split_train_holdout(weekly_demand, train_end=42):
    """Split weekly demand into train (weeks 1-42) and holdout (weeks 43-52)."""
    train_w = weekly_demand[:train_end]
    holdout_w = weekly_demand[train_end:train_end + 10]
    return train_w, holdout_w


def compute_normality_tests(train_w):
    """
    Run Shapiro-Wilk and D'Agostino normality tests on training demand.
    Returns dict with test statistics, p-values, skew, and kurtosis.
    """
    sw_stat, sw_p = stats.shapiro(train_w)
    da_stat, da_p = stats.normaltest(train_w)
    skewness = float(stats.skew(train_w))
    kurt = float(stats.kurtosis(train_w))
    return {
        "shapiro_stat": sw_stat,
        "shapiro_p": sw_p,
        "dagostino_stat": da_stat,
        "dagostino_p": da_p,
        "skew": skewness,
        "kurtosis": kurt,
        "mean": float(np.mean(train_w)),
        "std": float(np.std(train_w, ddof=1)),
        "min": float(np.min(train_w)),
        "max": float(np.max(train_w)),
    }


def load_data(data_dir=DATA_DIR):
    """Main entry point: load, split, and return train/holdout with stats."""
    weekly = load_m5_weekly(data_dir)
    train_w, holdout_w = split_train_holdout(weekly)
    stats_dict = compute_normality_tests(train_w)
    return train_w, holdout_w, stats_dict
