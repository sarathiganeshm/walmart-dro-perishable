"""
Normality diagnostics for weekly demand — saves CSV report to results/tables/.
Wraps scipy Shapiro-Wilk and D'Agostino tests from data_loader.
"""

import csv
from pathlib import Path
from src.data_loader import compute_normality_tests

TABLES_DIR = Path(__file__).parent.parent / "results" / "tables"


def save_normality_report(train_w, path=None):
    """Run normality tests on train_w and save results to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stats = compute_normality_tests(train_w)
    path = path or TABLES_DIR / "normality_tests.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Statistic", "Value"])
        for k, v in stats.items():
            writer.writerow([k, f"{v:.6f}"])
    return stats, str(path)
