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
    fig1: INFORMS-style 4-stage top row -> central LP box (double border) ->
    3-output bottom row.  LaTeX mathtext equations throughout.
    """
    _apply_rc()

    def _box(ax, x, y, w, h, text, fontsize=8.5, bold=False, double=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor="white", edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        if double:
            rect2 = FancyBboxPatch((x + 0.08, y + 0.08), w - 0.16, h - 0.16,
                                   boxstyle="round,pad=0.02", facecolor="none",
                                   edgecolor="black", linewidth=0.6)
            ax.add_patch(rect2)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                linespacing=1.5)

    def _arrow(ax, x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                            mutation_scale=14, color="black", linewidth=1.0)
        ax.add_patch(a)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Two title lines ───────────────────────────────────────────────────
    ax.text(6, 7.6,
            "Methodology: Data-Driven Wasserstein DRO-CVaR Framework",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(6, 7.25,
            "Applied to Walmart FOODS_3 Perishable Inventory (M5 Dataset)",
            ha="center", va="center", fontsize=9, style="italic",
            color="dimgrey")

    # ── TOP ROW: 4 stage boxes at y=5.3, h=1.2 ───────────────────────────
    top_stages = [
        (0.3,  2.4,
         "INPUT DATA\n\nM5 Walmart FOODS_3\n42 weeks training\n10 weeks holdout"),
        (3.5,  2.6,
         "MONTE CARLO\nSCENARIO GENERATION\n\nN=500 bootstrap\n"
         "from empirical $P_N$"),
        (6.9,  2.6,
         "WASSERSTEIN\nAMBIGUITY SET\n\n$B_\\varepsilon(P_N)$\n"
         "$\\varepsilon=\\sigma\\sqrt{\\log N / N}$"),
        (10.3, 1.6,
         "CVaR$_{0.95}$\n\nRockafellar-\nUryasev (2000)"),
    ]
    box_y, box_h = 5.3, 1.2
    for x, w, txt in top_stages:
        _box(ax, x, box_y, w, box_h, txt, fontsize=8)

    # Filled black stage-number circles above each box
    for idx, (x, w, _) in enumerate(top_stages):
        cx = x + w / 2
        circ = plt.Circle((cx, 6.7), 0.22, color="black", zorder=5)
        ax.add_patch(circ)
        ax.text(cx, 6.7, str(idx + 1), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=6)

    # Horizontal arrows between adjacent top-row boxes
    for i in range(len(top_stages) - 1):
        x1 = top_stages[i][0] + top_stages[i][1]
        x2 = top_stages[i + 1][0]
        _arrow(ax, x1, box_y + box_h / 2, x2, box_y + box_h / 2)

    # ── DOWN ARROW: Stage 4 bottom -> LP box top ──────────────────────────
    stage4_cx = top_stages[3][0] + top_stages[3][1] / 2
    lp_top = 3.0 + 1.4
    _arrow(ax, stage4_cx, box_y, stage4_cx, lp_top)

    # ── CENTRAL LP BOX (double border) ───────────────────────────────────
    lp_text = (
        "DRO-CVaR LINEAR PROGRAMMING MODEL  (solved via CBC)\n\n"
        r"$\min_{Q,\tau,u_i,v_i,\eta_i}$  "
        r"$(1-\lambda)\bar{c} + \lambda\!\left(\tau + "
        r"\frac{1}{N(1-\alpha)}\sum_i \eta_i + \varepsilon L\right)$"
        "\n"
        r"s.t.  $u_i \geq Q-d_i$,   $v_i \geq d_i-Q$,   "
        r"$\eta_i \geq c(Q,d_i)-\tau$,   "
        r"$Q\cdot e_{\mathrm{factor}} \leq C_{\mathrm{cap}}$,   "
        r"$u_i, v_i, \eta_i \geq 0$"
    )
    _box(ax, 0.3, 3.0, 11.6, 1.4, lp_text, fontsize=8.5, double=True)

    # ── DOWN ARROW: LP box bottom -> output row ───────────────────────────
    lp_cx = 0.3 + 11.6 / 2
    _arrow(ax, lp_cx, 3.0, lp_cx, 0.7 + 1.4)

    # ── BOTTOM ROW: 3 output boxes at y=0.7, h=1.4 ───────────────────────
    out_boxes = [
        (0.3, 3.6,
         "OPTIMAL POLICY\n\n$Q^*$ (order quantity)\n$\\tau^*$ (VaR$_{0.95}$)\n"
         "Expected cost\nCVaR$_{0.95}$"),
        (4.2, 3.6,
         "CARBON IMPACT\n\nEmissions (tCO$_2$/wk)\nTrading credit (£)\n"
         "Cap-and-trade\nsensitivity"),
        (8.1, 3.8,
         "OUT-OF-SAMPLE VALIDATION\n\n10-week holdout test\n"
         "Generalisation error\nTail-risk reduction"),
    ]
    for x, w, txt in out_boxes:
        _box(ax, x, 0.7, w, 1.4, txt, fontsize=8)

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

    # Panel (a): Efficient frontier scatter — 5 distinct markers, axes in £000
    ax = axes[0]
    ec = exp_costs
    cv = cvar_vals
    marker_shapes = ["o", "s", "D", "^", "v"]
    ax.plot([e / 1000 for e in ec], [c / 1000 for c in cv],
            "k-", linewidth=0.8, alpha=0.7, zorder=2)
    for i, (lam, ec_val, cv_val) in enumerate(zip(lambdas, ec, cv)):
        mk = marker_shapes[i % len(marker_shapes)]
        ax.scatter(ec_val / 1000, cv_val / 1000, marker=mk,
                   facecolors="white", edgecolors="black", s=80,
                   linewidths=1.2, zorder=4, label=f"$\\lambda$={lam:.2f}")
    # Naive benchmark
    ax.scatter(naive_cost / 1000, naive_cvar / 1000, marker="x",
               color="black", s=120, linewidths=2.5, zorder=5)
    ax.annotate("Naïve", xy=(naive_cost / 1000, naive_cvar / 1000),
                xytext=(6, -12), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Expected cost (£000/week)")
    ax.set_ylabel("CVaR$_{0.95}$ (£000/week)")
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

    # Compute CVaR (expected cost above 95th percentile)
    def cvar95(c):
        threshold = np.percentile(c, 95)
        tail = c[c >= threshold]
        return float(np.mean(tail)) if len(tail) > 0 else threshold

    cvar_rn = cvar95(costs_rn)
    cvar_dro = cvar95(costs_dro)
    cvar_reduction_pct = (cvar_rn - cvar_dro) / cvar_rn * 100
    q_rn = float(rn_result["Q"])
    q_dro = float(dro_result["Q"])

    # Panel (a): Overlaid histograms — density, x-axis in £000
    ax = axes[0]
    all_costs = np.concatenate([costs_rn, costs_dro])
    bins = np.linspace(np.percentile(all_costs, 1), np.percentile(all_costs, 99), 30)
    bins_k = bins / 1000.0

    ax.hist(costs_rn / 1000, bins=bins_k, density=True, color="white", edgecolor="black",
            hatch="\\\\", linewidth=0.6, alpha=0.85,
            label=f"Risk-neutral (Q* = {q_rn:,.0f})")
    ax.hist(costs_dro / 1000, bins=bins_k, density=True, color="dimgrey", edgecolor="black",
            linewidth=0.5, alpha=0.55,
            label=f"DRO-CVaR λ=0.5 (Q* = {q_dro:,.0f})")

    # CVaR lines (true expected shortfall, not VaR)
    ax.axvline(cvar_rn / 1000, color="black", linestyle="--", linewidth=1.0,
               label=f"CVaR RN = £{cvar_rn/1000:.1f}k")
    ax.axvline(cvar_dro / 1000, color="black", linestyle=":", linewidth=1.0,
               label=f"CVaR DRO = £{cvar_dro/1000:.1f}k")

    ax.set_xlabel("Weekly cost (£000)")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) Cost distribution — CVaR reduction: {cvar_reduction_pct:.1f}%", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4)
    ax.set_axisbelow(True)

    # Panel (b): Percentile plot — P95 cutoff, costs in £000/week
    ax2 = axes[1]
    pcts = np.linspace(0, 100, len(scenarios))
    sorted_rn_k = np.sort(costs_rn) / 1000
    sorted_dro_k = np.sort(costs_dro) / 1000

    ax2.plot(pcts, sorted_rn_k, "k-", linewidth=1.2, label="Risk-neutral")
    ax2.plot(pcts, sorted_dro_k, "k--", linewidth=1.2, label="DRO-CVaR λ=0.5")

    # Shade tail-risk reduction between curves above P95
    tail_mask = pcts >= 95
    ax2.fill_between(
        pcts[tail_mask],
        sorted_rn_k[tail_mask],
        sorted_dro_k[tail_mask],
        color="lightgrey",
        alpha=0.7,
        label="Tail-risk reduction",
    )
    ax2.axvline(95, color="black", linestyle=":", linewidth=0.8)
    ax2.text(95.5, sorted_rn_k[tail_mask][0] * 0.99, "$P_{95}$", fontsize=7, va="top")
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("Cost (£000/week)")
    ax2.set_title("(b) Sorted scenario costs — tail-risk visualisation", fontsize=9)
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
