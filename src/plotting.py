"""
B&W journal-quality figures for Wasserstein DRO-CVaR inventory paper.
All figures strictly monochrome: black, white, lightgrey, dimgrey only.
Hatching and line styles distinguish series. 300 dpi PNG output.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "grid.color": "#bbbbbb",
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def _apply_rc():
    plt.rcParams.update(RC)


def plot_methodology_flowchart(save_path=None):
    """
    fig1: Pure schematic flowchart — 6 stages with FancyBboxPatch nodes,
    numbered circles, and FancyArrowPatch connections.
    Stages: Input Data -> MC Scenarios -> Wasserstein -> CVaR -> LP Model -> Outputs
    """
    _apply_rc()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        (0.5, "1", "Input Data\n(M5 Walmart\nFOODS_3/CA_1)"),
        (2.0, "2", "Monte Carlo\nScenarios\n(N=500, Bootstrap)"),
        (3.5, "3", "Wasserstein\nRadius\n(ε calibration)"),
        (5.0, "4", "CVaR\nLinearisation\n(α=0.95)"),
        (6.5, "5", "DRO-CVaR\nLP (PuLP+CBC)"),
        (8.0, "6", "Outputs\n(Q*, Cost,\nService Level)"),
    ]

    box_w, box_h = 1.1, 1.6
    cy = 2.0

    for (cx, num, label) in stages:
        # Box
        fancy = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.08",
            linewidth=1.0,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(fancy)
        # Number circle
        circle = plt.Circle((cx - box_w / 2 + 0.18, cy + box_h / 2 - 0.18),
                             0.16, color="black", zorder=5)
        ax.add_patch(circle)
        ax.text(cx - box_w / 2 + 0.18, cy + box_h / 2 - 0.18, num,
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", zorder=6)
        # Label
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.5,
                linespacing=1.4)

    # Arrows between boxes
    for i in range(len(stages) - 1):
        x_start = stages[i][0] + box_w / 2
        x_end = stages[i + 1][0] - box_w / 2
        arrow = FancyArrowPatch(
            (x_start, cy), (x_end, cy),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="black",
        )
        ax.add_patch(arrow)

    # Carbon branch arrow from LP box
    lp_cx = stages[4][0]
    ax.annotate(
        "Carbon\nConstraint",
        xy=(lp_cx, cy - box_h / 2),
        xytext=(lp_cx, cy - box_h / 2 - 0.7),
        fontsize=7,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="dimgrey", lw=0.8),
        color="dimgrey",
    )

    ax.set_title(
        "Figure 1: DRO-CVaR Methodology Flowchart",
        fontsize=10, fontweight="bold", pad=6,
    )

    save_path = save_path or FIGURES_DIR / "fig1_methodology_flowchart.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_demand_analysis(train_w, scenarios, norm_stats, save_path=None):
    """
    fig2: 2-panel demand analysis.
    (a) Histogram: training data (hatch='//') vs scenarios (lightgrey fill),
        normal-fit dashed line, mean and P95 vertical lines.
    (b) Q-Q plot: open black circles, reference line, stats text box.
    """
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Histogram
    ax = axes[0]
    bins = np.linspace(
        min(np.min(train_w), np.min(scenarios)),
        max(np.max(train_w), np.max(scenarios)),
        25,
    )
    ax.hist(scenarios, bins=bins, color="lightgrey", edgecolor="black",
            linewidth=0.5, label="Scenarios (N=500)", alpha=0.85)
    ax.hist(train_w, bins=bins, color="white", edgecolor="black",
            hatch="//", linewidth=0.6, label="Training data", alpha=0.9)

    # Normal fit
    mu, sigma = float(np.mean(train_w)), float(np.std(train_w, ddof=1))
    x_fit = np.linspace(bins[0], bins[-1], 200)
    # Scale pdf to histogram counts
    bin_width = bins[1] - bins[0]
    pdf_scaled = stats.norm.pdf(x_fit, mu, sigma) * len(scenarios) * bin_width
    ax.plot(x_fit, pdf_scaled, "k--", linewidth=1.2, label="Normal fit")

    # Mean and P95 lines
    ax.axvline(mu, color="black", linestyle="-", linewidth=1.0, label=f"Mean={mu:.0f}")
    p95 = float(np.percentile(train_w, 95))
    ax.axvline(p95, color="dimgrey", linestyle=":", linewidth=1.0, label=f"P95={p95:.0f}")

    ax.set_xlabel("Weekly Demand (units)")
    ax.set_ylabel("Frequency")
    ax.set_title("(a) Demand Distribution", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, linewidth=0.4)
    ax.set_axisbelow(True)

    # Panel (b): Q-Q plot
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(train_w, dist="norm")
    ax2.plot(osm, osr, "o", markersize=5, markerfacecolor="white",
             markeredgecolor="black", markeredgewidth=0.8, label="Observations")
    q_line = np.array([osm[0], osm[-1]])
    ax2.plot(q_line, slope * q_line + intercept, "k-", linewidth=1.0, label="Reference line")

    stats_text = (
        f"Shapiro-Wilk: p={norm_stats['shapiro_p']:.3f}\n"
        f"D'Agostino: p={norm_stats['dagostino_p']:.3f}\n"
        f"Skewness: {norm_stats['skew']:.3f}\n"
        f"Kurtosis: {norm_stats['kurtosis']:.3f}"
    )
    ax2.text(
        0.05, 0.97, stats_text,
        transform=ax2.transAxes,
        fontsize=7.5,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.6),
    )

    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("(b) Normal Q-Q Plot", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(True, linewidth=0.4)
    ax2.set_axisbelow(True)

    fig.suptitle("Figure 2: Weekly Demand Analysis — FOODS_3, CA_1 (Weeks 1–42)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = save_path or FIGURES_DIR / "fig2_demand_analysis.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_efficient_frontier(lambda_results, save_path=None):
    """
    fig3: 2-panel efficient frontier.
    lambda_results: list of dicts with keys lambda, expected_cost, CVaR95,
                    service_level_is (in-sample), service_level_oos (out-of-sample).
    (a) Scatter of (E[cost], CVaR95) for each lambda + naive policy.
    (b) Grouped bars: in-sample SL (white, hatch='..') vs OOS SL (lightgrey, hatch='//')
    """
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    markers = ["o", "s", "D", "^", "v"]

    lambdas = [r["lambda"] for r in lambda_results]
    exp_costs = [r["expected_cost"] for r in lambda_results]
    cvar_vals = [r["CVaR95"] for r in lambda_results]
    sl_is = [r.get("service_level_is", r.get("service_level", 0)) for r in lambda_results]
    sl_oos = [r.get("service_level_oos", r.get("service_level", 0)) for r in lambda_results]
    naive_cost = lambda_results[0].get("naive_cost", exp_costs[0])
    naive_cvar = lambda_results[0].get("naive_cvar", cvar_vals[0])

    # Panel (a): Efficient frontier scatter
    ax = axes[0]
    ax.plot(exp_costs, cvar_vals, "k-", linewidth=0.8, zorder=1)
    for i, (ec, cv, lam, mk) in enumerate(zip(exp_costs, cvar_vals, lambdas, markers)):
        ax.plot(ec, cv, mk, color="black", markersize=7,
                markerfacecolor="white" if i % 2 == 0 else "black",
                markeredgecolor="black", markeredgewidth=0.8,
                label=f"λ={lam:.2f}", zorder=2)
    # Naive policy
    ax.plot(naive_cost, naive_cvar, "kx", markersize=10, markeredgewidth=1.5,
            label="Naive (mean)", zorder=3)
    ax.set_xlabel("Expected Cost (£/week)")
    ax.set_ylabel("CVaR₉₅ (£/week)")
    ax.set_title("(a) Risk–Cost Efficient Frontier", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linewidth=0.4)
    ax.set_axisbelow(True)

    # Panel (b): Grouped service level bars
    ax2 = axes[1]
    x = np.arange(len(lambdas))
    w = 0.35
    bars_is = ax2.bar(x - w / 2, [s * 100 for s in sl_is], w,
                      color="white", edgecolor="black", linewidth=0.7,
                      hatch="..", label="In-sample SL (%)")
    bars_oos = ax2.bar(x + w / 2, [s * 100 for s in sl_oos], w,
                       color="lightgrey", edgecolor="black", linewidth=0.7,
                       hatch="//", label="OOS SL (%)")
    ax2.axhline(95, color="black", linestyle="--", linewidth=0.9, label="95% target")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"λ={l:.2f}" for l in lambdas], fontsize=8)
    ax2.set_ylabel("Service Level (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("(b) In-Sample vs Out-of-Sample Service Level", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(True, axis="y", linewidth=0.4)
    ax2.set_axisbelow(True)

    fig.suptitle("Figure 3: Efficient Frontier — Risk Weight λ Sweep",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = save_path or FIGURES_DIR / "fig3_efficient_frontier.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_cost_distributions(scenarios, rn_result, dro_result, save_path=None):
    """
    fig4: 2-panel cost distributions.
    rn_result, dro_result: dicts with keys Q, CVaR95, tau (VaR).
    (a) Overlaid histograms RN (white, hatch='\\\\') vs DRO-CVaR (dimgrey).
    (b) Sorted scenario costs as percentile plot; tail shaded lightgrey.
    """
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Compute per-scenario costs for RN and DRO
    def scenario_costs(Q, scenarios, c_order=2.50, c_hold=0.80, c_short=8.00, c_waste=1.50):
        Q = float(Q)
        costs = []
        for d in scenarios:
            u = max(Q - float(d), 0)
            v = max(float(d) - Q, 0)
            costs.append(c_order * Q + (c_hold + c_waste) * u + c_short * v)
        return np.array(costs)

    costs_rn = scenario_costs(rn_result["Q"], scenarios)
    costs_dro = scenario_costs(dro_result["Q"], scenarios)

    # Panel (a): Overlaid histograms
    ax = axes[0]
    all_costs = np.concatenate([costs_rn, costs_dro])
    bins = np.linspace(np.percentile(all_costs, 1), np.percentile(all_costs, 99), 30)

    ax.hist(costs_rn, bins=bins, color="white", edgecolor="black",
            hatch="\\\\", linewidth=0.6, alpha=0.85, label="Risk-Neutral (RN)")
    ax.hist(costs_dro, bins=bins, color="dimgrey", edgecolor="black",
            linewidth=0.5, alpha=0.55, label="DRO-CVaR (λ=0.5)")

    # VaR lines
    var_rn = float(np.percentile(costs_rn, 95))
    var_dro = float(np.percentile(costs_dro, 95))
    ax.axvline(var_rn, color="black", linestyle="--", linewidth=1.0, label=f"VaR₉₅ RN=£{var_rn:,.0f}")
    ax.axvline(var_dro, color="black", linestyle=":", linewidth=1.0, label=f"VaR₉₅ DRO=£{var_dro:,.0f}")

    ax.set_xlabel("Scenario Cost (£)")
    ax.set_ylabel("Frequency")
    ax.set_title("(a) Cost Distribution: RN vs DRO-CVaR", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4)
    ax.set_axisbelow(True)

    # Panel (b): Percentile plot
    ax2 = axes[1]
    pcts = np.linspace(0, 100, len(scenarios))
    ax2.plot(pcts, np.sort(costs_rn), "k-", linewidth=1.2, label="Risk-Neutral")
    ax2.plot(pcts, np.sort(costs_dro), "k--", linewidth=1.2, label="DRO-CVaR (λ=0.5)")

    # Shade tail region > P90
    p90_pct = 90
    tail_pcts = pcts[pcts >= p90_pct]
    ax2.fill_between(
        tail_pcts,
        np.sort(costs_rn)[pcts >= p90_pct],
        np.sort(costs_dro)[pcts >= p90_pct],
        color="lightgrey",
        alpha=0.7,
        label="Tail region (>P90)",
    )
    ax2.axvline(90, color="dimgrey", linestyle="-.", linewidth=0.8)
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("Scenario Cost (£)")
    ax2.set_title("(b) Sorted Scenario Costs — Percentile View", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(True, linewidth=0.4)
    ax2.set_axisbelow(True)

    fig.suptitle("Figure 4: Cost Distributions — Risk-Neutral vs DRO-CVaR",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = save_path or FIGURES_DIR / "fig4_cost_distributions.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_carbon_sensitivity(cap_results, save_path=None):
    """
    fig5: 2-panel carbon sensitivity.
    cap_results: list of dicts with keys cap, Q, emissions, credit, expected_cost, waste_pct, service_level.
    (a) Emissions vs cap; trading credit bars on twin axis.
    (b) Cost, waste%, service level vs cap on twin axes.
    """
    _apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    caps = [r["cap"] for r in cap_results]
    emissions = [r["emissions"] for r in cap_results]
    credits = [r["credit"] for r in cap_results]
    costs = [r["expected_cost"] for r in cap_results]
    waste_pcts = [r["waste_pct"] for r in cap_results]
    sls = [r["service_level"] * 100 for r in cap_results]

    # Panel (a): Emissions vs cap + trading credit bars
    ax = axes[0]
    ax.plot(caps, caps, "k--", linewidth=0.9, label="Cap reference (y=x)")
    ax.plot(caps, emissions, "ko-", markersize=5, markerfacecolor="white",
            markeredgewidth=0.8, linewidth=1.0, label="Actual emissions")
    ax.set_xlabel("Carbon Cap (t CO₂e/week)")
    ax.set_ylabel("Emissions (t CO₂e/week)")
    ax.set_title("(a) Emissions vs Carbon Cap", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linewidth=0.4)
    ax.set_axisbelow(True)

    ax_twin = ax.twinx()
    bar_colors = ["white" if c >= 0 else "dimgrey" for c in credits]
    bars = ax_twin.bar(caps, credits, width=2.5, color=bar_colors,
                       edgecolor="black", linewidth=0.6, hatch="||",
                       alpha=0.7, label="Trading credit (£)")
    ax_twin.axhline(0, color="black", linewidth=0.5)
    ax_twin.set_ylabel("Trading Credit (£)")
    ax_twin.legend(fontsize=7, loc="lower right")

    # Panel (b): Cost, waste, service level vs cap
    ax2 = axes[1]
    ax2.plot(caps, costs, "k^-", markersize=6, markerfacecolor="white",
             markeredgewidth=0.8, linewidth=1.0, label="Exp. Cost (£)")
    ax2.plot(caps, waste_pcts, "ks--", markersize=6, markerfacecolor="black",
             linewidth=1.0, label="Waste %")
    ax2.set_xlabel("Carbon Cap (t CO₂e/week)")
    ax2.set_ylabel("Expected Cost (£)  /  Waste %")
    ax2.set_title("(b) Cost, Waste & Service Level vs Carbon Cap", fontsize=9)
    ax2.grid(True, linewidth=0.4)
    ax2.set_axisbelow(True)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(caps, sls, "kD:", markersize=6, markerfacecolor="white",
                  markeredgewidth=0.8, linewidth=1.0, label="Service Level (%)")
    ax2_twin.set_ylabel("Service Level (%)")
    ax2_twin.set_ylim(0, 110)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.suptitle("Figure 5: Carbon Cap Sensitivity Analysis",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = save_path or FIGURES_DIR / "fig5_carbon_sensitivity.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def generate_all_figures(train_w, scenarios, norm_stats, lambda_results,
                          rn_result, dro_result, cap_results):
    """Generate all 5 B&W journal figures and return list of saved paths."""
    paths = []
    paths.append(plot_methodology_flowchart())
    paths.append(plot_demand_analysis(train_w, scenarios, norm_stats))
    paths.append(plot_efficient_frontier(lambda_results))
    paths.append(plot_cost_distributions(scenarios, rn_result, dro_result))
    paths.append(plot_carbon_sensitivity(cap_results))
    return paths
