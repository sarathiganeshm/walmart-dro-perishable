"""
DRO-CVaR inventory optimisation via PuLP + CBC.

Implements Rockafellar-Uryasev CVaR linearisation with Mohajerin Esfahani &
Kuhn (2018) type-1 Wasserstein robustification (as cited in Kim & Chung 2024).

Formulation:
  Decision variables: Q, tau, u_i, v_i, eta_i for i=1..N
  Constraints (each scenario i):
    u_i >= Q - d_i          (overstock)
    v_i >= d_i - Q          (understock)
    eta_i >= c_order*Q + (c_hold+c_waste)*u_i + c_short*v_i - tau   (CVaR slack)
  Optional carbon constraint:
    (e_per_unit + e_per_km*avg_km/1000) * Q <= carbon_cap

  Objective:
    E_cost    = (1/N) * sum_i [c_order*Q + (c_hold+c_waste)*u_i + c_short*v_i]
    CVaR_term = tau + (1/(N*(1-alpha)))*sum_i eta_i + epsilon * L_lipschitz
    Minimise (1-lam)*E_cost + lam*CVaR_term

References:
  Rockafellar, R.T. & Uryasev, S. (2000). J. Risk, 2(3), 21-41.
  Mohajerin Esfahani, P. & Kuhn, D. (2018). Math. Programming, 171(1-2), 115-166.
  Kim, Y.G. & Chung, B.D. (2024). Omega, 127.
  Gao, R., Chen, X. & Kleywegt, A.J. (2024). Operations Research, 72(3).
"""

import numpy as np
import pulp

# Cost parameters
PARAMS = {
    "c_order": 2.50,
    "c_hold": 0.80,
    "c_short": 8.00,
    "c_waste": 1.50,
    "c_carbon": 15.0,
    "e_per_unit": 0.003,
    "e_per_km": 0.0008,
    "avg_km": 50.0,
    "alpha": 0.95,
}


def solve_dro_cvar(
    scenarios,
    epsilon,
    lam=0.5,
    alpha=0.95,
    c_order=2.50,
    c_hold=0.80,
    c_short=8.00,
    c_waste=1.50,
    carbon_cap=None,
    e_per_unit=0.003,
    e_per_km=0.0008,
    avg_km=50.0,
    solver="cbc",
):
    """
    Solve the DRO-CVaR newsvendor LP.

    Rockafellar-Uryasev (2000) CVaR linearisation with Wasserstein
    robustification from Mohajerin Esfahani & Kuhn (2018), as applied
    in Kim & Chung (2024) and Gao et al. (2024).

    Parameters
    ----------
    scenarios : np.ndarray, shape (N,)
        Simulated demand scenarios.
    epsilon : float
        Wasserstein ball radius.
    lam : float
        Risk weight in [0,1]. lam=0 → risk-neutral; lam=1 → pure CVaR.
    alpha : float
        CVaR confidence level (default 0.95).
    c_order, c_hold, c_short, c_waste : float
        Unit cost parameters.
    carbon_cap : float or None
        Optional carbon cap in tonnes. None = unconstrained.
    e_per_unit, e_per_km, avg_km : float
        Emission factors for carbon constraint.
    solver : str
        'cbc' (default), 'gurobi', or 'cplex'.

    Returns
    -------
    dict with keys:
      Q, tau, expected_cost, CVaR95, CVaR_robust, carbon,
      waste_pct, service_level, status
    """
    scenarios = np.asarray(scenarios, dtype=float)
    N = len(scenarios)
    L_lipschitz = max(c_hold + c_waste, c_short)

    # Bounds for Q
    Q_lb = 0.6 * float(np.min(scenarios))
    Q_ub = 1.4 * float(np.max(scenarios))

    prob = pulp.LpProblem("DRO_CVaR_Inventory", pulp.LpMinimize)

    # Decision variables
    Q = pulp.LpVariable("Q", lowBound=Q_lb, upBound=Q_ub)
    tau = pulp.LpVariable("tau")  # VaR (free)
    u = [pulp.LpVariable(f"u_{i}", lowBound=0) for i in range(N)]
    v = [pulp.LpVariable(f"v_{i}", lowBound=0) for i in range(N)]
    eta = [pulp.LpVariable(f"eta_{i}", lowBound=0) for i in range(N)]

    # Scenario constraints
    for i in range(N):
        d_i = float(scenarios[i])
        prob += u[i] >= Q - d_i, f"overstock_{i}"
        prob += v[i] >= d_i - Q, f"understock_{i}"
        cost_i = c_order * Q + (c_hold + c_waste) * u[i] + c_short * v[i]
        prob += eta[i] >= cost_i - tau, f"cvar_slack_{i}"

    # Optional carbon constraint
    emission_factor = e_per_unit + e_per_km * avg_km / 1000.0
    if carbon_cap is not None:
        prob += emission_factor * Q <= carbon_cap, "carbon_cap"

    # Objective components
    E_cost = (1.0 / N) * pulp.lpSum(
        c_order * Q + (c_hold + c_waste) * u[i] + c_short * v[i]
        for i in range(N)
    )
    CVaR_term = (
        tau
        + (1.0 / (N * (1.0 - alpha))) * pulp.lpSum(eta[i] for i in range(N))
        + epsilon * L_lipschitz
    )
    prob += (1.0 - lam) * E_cost + lam * CVaR_term

    # Solver selection with graceful fallback
    def _get_solver():
        if solver == "gurobi":
            try:
                return pulp.GUROBI_CMD(msg=False)
            except Exception:
                return pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        elif solver == "cplex":
            try:
                return pulp.CPLEX_CMD(msg=False)
            except Exception:
                return pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=60)

    prob.solve(_get_solver())

    status = pulp.LpStatus[prob.status]
    if prob.status != 1:
        return {"status": status, "Q": None}

    Q_val = float(pulp.value(Q))
    tau_val = float(pulp.value(tau))

    # Compute per-scenario costs
    costs = np.array([
        c_order * Q_val
        + (c_hold + c_waste) * max(Q_val - float(scenarios[i]), 0)
        + c_short * max(float(scenarios[i]) - Q_val, 0)
        for i in range(N)
    ])

    expected_cost = float(np.mean(costs))
    CVaR95 = float(np.percentile(costs, alpha * 100))
    CVaR_robust = CVaR95 + epsilon * L_lipschitz

    # Carbon emissions
    carbon = emission_factor * Q_val

    # Waste % = fraction of scenarios where demand < Q (over-ordered)
    waste_count = np.sum(scenarios < Q_val)
    waste_pct = float(waste_count / N * 100)

    # Service level = fraction of scenarios demand <= Q (not stocked out)
    service_level = float(np.mean(scenarios <= Q_val))

    return {
        "Q": Q_val,
        "tau": tau_val,
        "expected_cost": expected_cost,
        "CVaR95": CVaR95,
        "CVaR_robust": CVaR_robust,
        "carbon": carbon,
        "waste_pct": waste_pct,
        "service_level": service_level,
        "status": status,
    }
