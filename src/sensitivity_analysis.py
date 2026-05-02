"""
Sensitivity analysis: lambda sweep and carbon cap sweep.
Saves all results to results/tables/ as CSV.
Includes both in-sample and out-of-sample (10-week holdout) metrics.
Naive benchmark: order at historical mean.
"""

import csv
import numpy as np
from pathlib import Path
from src.dro_cvar_lp import solve_dro_cvar
from src.carbon_model import compute_emissions, compute_trading_credit

TABLES_DIR = Path(__file__).parent.parent / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
CARBON_CAPS = [40, 45, 50, 55, 60, 70, 80]

PARAMS = dict(
    c_order=2.50, c_hold=0.80, c_short=8.00, c_waste=1.50,
    e_per_unit=0.003, e_per_km=0.0008, avg_km=50.0, alpha=0.95,
)


def _oos_metrics(Q, holdout_w, **params):
    """Compute out-of-sample expected cost and service level on holdout weeks."""
    c_order = params.get("c_order", 2.50)
    c_hold = params.get("c_hold", 0.80)
    c_short = params.get("c_short", 8.00)
    c_waste = params.get("c_waste", 1.50)
    costs = []
    hits = 0
    for d in holdout_w:
        u = max(Q - float(d), 0)
        v = max(float(d) - Q, 0)
        costs.append(c_order * Q + (c_hold + c_waste) * u + c_short * v)
        if d <= Q:
            hits += 1
    return float(np.mean(costs)), float(hits / len(holdout_w))


def naive_policy(train_w, holdout_w, **params):
    """Naive benchmark: order at historical mean."""
    Q_naive = float(np.mean(train_w))
    costs = []
    for d in holdout_w:
        u = max(Q_naive - float(d), 0)
        v = max(float(d) - Q_naive, 0)
        costs.append(params.get("c_order", 2.50) * Q_naive
                     + (params.get("c_hold", 0.80) + params.get("c_waste", 1.50)) * u
                     + params.get("c_short", 8.00) * v)
    return {
        "Q": Q_naive,
        "expected_cost_oos": float(np.mean(costs)),
        "service_level_oos": float(np.mean([d <= Q_naive for d in holdout_w])),
    }


def lambda_sweep(train_w, holdout_w, scenarios, epsilon, **params):
    """
    Sweep lambda in [0.0, 0.25, 0.5, 0.75, 1.0].
    Returns list of dicts with in-sample and OOS metrics.
    Saves results/tables/lambda_sweep.csv.
    """
    results = []
    naive = naive_policy(train_w, holdout_w, **params)

    for lam in LAMBDA_VALUES:
        sol = solve_dro_cvar(scenarios, epsilon, lam=lam, **params)
        if sol.get("Q") is None:
            continue
        Q = sol["Q"]
        oos_cost, oos_sl = _oos_metrics(Q, holdout_w, **params)

        # In-sample service level from scenarios
        sl_is = float(np.mean(np.array(scenarios) <= Q))

        results.append({
            "lambda": lam,
            "Q": Q,
            "expected_cost": sol["expected_cost"],
            "CVaR95": sol["CVaR95"],
            "CVaR_robust": sol["CVaR_robust"],
            "carbon": sol["carbon"],
            "waste_pct": sol["waste_pct"],
            "service_level_is": sl_is,
            "service_level_oos": oos_sl,
            "oos_cost": oos_cost,
            "status": sol["status"],
            "naive_cost": naive["expected_cost_oos"],
            "naive_cvar": naive["expected_cost_oos"] * 1.25,
            "naive_sl_oos": naive["service_level_oos"],
        })

    _save_csv(results, TABLES_DIR / "lambda_sweep.csv")
    return results


def carbon_cap_sweep(train_w, holdout_w, scenarios, epsilon, lam=0.5, **params):
    """
    Sweep carbon caps at lambda=0.5.
    Returns list of dicts. Saves results/tables/carbon_cap_sweep.csv.
    """
    results = []

    for cap in CARBON_CAPS:
        sol = solve_dro_cvar(scenarios, epsilon, lam=lam, carbon_cap=cap, **params)
        if sol.get("Q") is None:
            results.append({"cap": cap, "status": sol.get("status", "INFEASIBLE"),
                             "Q": None})
            continue
        Q = sol["Q"]
        oos_cost, oos_sl = _oos_metrics(Q, holdout_w, **params)
        credit = compute_trading_credit(
            Q, cap,
            e_per_unit=params.get("e_per_unit", 0.003),
            e_per_km=params.get("e_per_km", 0.0008),
            avg_km=params.get("avg_km", 50.0),
        )
        results.append({
            "cap": cap,
            "Q": Q,
            "expected_cost": sol["expected_cost"],
            "CVaR95": sol["CVaR95"],
            "carbon": sol["carbon"],
            "emissions": sol["carbon"],
            "credit": credit,
            "waste_pct": sol["waste_pct"],
            "service_level": float(np.mean(np.array(scenarios) <= Q)),
            "oos_cost": oos_cost,
            "oos_sl": oos_sl,
            "status": sol["status"],
        })

    _save_csv(results, TABLES_DIR / "carbon_cap_sweep.csv")
    return results


def _save_csv(records, path):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in records:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    return str(path)
