"""
5-scene workflow animation for DRO-CVaR inventory paper.
140 frames, fps=8, dpi=100. Strictly B&W.
Saved to results/animation/animation_workflow.gif.

Scenes:
  1 (0-25):   Supply chain visualisation (DC -> 3 stores -> customers)
  2 (25-65):  Monte Carlo histogram fills in (5 -> 500 scenarios)
  3 (65-90):  Wasserstein ball expands around empirical CDF
  4 (90-120): CVaR convergence curves (RN vs DRO)
  5 (120-140): Final comparison view
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

ANIM_DIR = Path(__file__).parent.parent / "results" / "animation"
ANIM_DIR.mkdir(parents=True, exist_ok=True)

RC = {
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
}

TOTAL_FRAMES = 140
FPS = 8


def _rng(seed=42):
    return np.random.default_rng(seed)


def make_animation(scenarios=None, train_w=None, epsilon=None, save_path=None):
    """
    Build and save the 5-scene workflow animation.
    If scenarios/train_w not provided, synthetic data are generated for demo.
    """
    plt.rcParams.update(RC)
    rng = _rng()

    # Synthetic data if not supplied
    if train_w is None:
        train_w = rng.normal(14000, 1500, 42).clip(9000, 20000)
    if scenarios is None:
        scenarios = rng.normal(14000, 1500, 500).clip(9000, 20000)
    if epsilon is None:
        epsilon = np.sqrt(np.log(500) / 500) * np.std(train_w, ddof=1)

    # Pre-compute CVaR curves for scene 4
    lambdas = np.linspace(0, 1, 30)
    rn_Q = float(np.mean(train_w))
    dro_Q = rn_Q * 1.08
    c_o, c_h, c_w, c_s = 2.50, 0.80, 1.50, 8.00

    def cost_arr(Q):
        return np.array([
            c_o * Q + (c_h + c_w) * max(Q - d, 0) + c_s * max(d - Q, 0)
            for d in scenarios
        ])

    rn_costs = cost_arr(rn_Q)
    dro_costs = cost_arr(dro_Q)
    rn_cvar_seq = [float(np.percentile(rn_costs, 95))] * 30
    dro_cvar_seq = [float(np.percentile(dro_costs, 95))] * 30
    # Simulate convergence as N grows
    N_seq = np.geomspace(5, 500, 30).astype(int)
    rn_cvar_seq = []
    dro_cvar_seq = []
    for n in N_seq:
        sub = rng.choice(scenarios, n, replace=False)
        rc = cost_arr(float(np.mean(sub)))
        dc = cost_arr(float(np.mean(sub)) * 1.08)
        rn_cvar_seq.append(float(np.percentile(rc, 95)))
        dro_cvar_seq.append(float(np.percentile(dc, 95)))

    # Figure layout: top=supply chain (large), bottom row = 3 panels
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], hspace=0.45, wspace=0.35)
    ax_sc = fig.add_subplot(gs[0, :])   # Scene 1 full-width supply chain
    ax_mc = fig.add_subplot(gs[1, 0])   # Scene 2 MC histogram
    ax_ws = fig.add_subplot(gs[1, 1])   # Scene 3 Wasserstein CDF
    ax_cv = fig.add_subplot(gs[1, 2])   # Scene 4 CVaR convergence

    for ax in [ax_sc, ax_mc, ax_ws, ax_cv]:
        ax.tick_params(labelsize=6)

    # ── Scene 1 helpers ──────────────────────────────────────────────────────

    def draw_supply_chain(ax, truck_progress=0.0):
        ax.cla()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title("Scene 1: Walmart Supply Chain", fontsize=8, fontweight="bold")

        # Distribution Centre
        dc = FancyBboxPatch((0.3, 1.5), 1.4, 1.0,
                            boxstyle="round,pad=0.05",
                            linewidth=1.0, edgecolor="black", facecolor="white")
        ax.add_patch(dc)
        ax.text(1.0, 2.0, "Walmart DC\n(CA_1)", ha="center", va="center",
                fontsize=6.5, fontweight="bold")

        # Roof triangle on DC
        tri = plt.Polygon([[0.3, 2.5], [1.0, 3.1], [1.7, 2.5]], closed=True,
                          edgecolor="black", facecolor="lightgrey", linewidth=0.8)
        ax.add_patch(tri)

        # 3 Stores
        store_xs = [4.5, 6.5, 8.5]
        store_ys = [3.0, 1.8, 0.6]
        for i, (sx, sy) in enumerate(zip(store_xs, store_ys)):
            store = FancyBboxPatch((sx - 0.5, sy - 0.35), 1.0, 0.7,
                                   boxstyle="round,pad=0.04",
                                   linewidth=0.8, edgecolor="black", facecolor="white")
            ax.add_patch(store)
            ax.text(sx, sy, f"Store {i+1}", ha="center", va="center", fontsize=6)

        # Customers (circles)
        cust_offsets = [(1.3, 0.2), (1.3, -0.2)]
        for i, (sx, sy) in enumerate(zip(store_xs, store_ys)):
            for dx, dy in cust_offsets:
                cust = Circle((sx + dx, sy + dy), 0.15,
                              edgecolor="black", facecolor="dimgrey", linewidth=0.6)
                ax.add_patch(cust)

        # Arrows DC -> stores
        for sx, sy in zip(store_xs, store_ys):
            ax.annotate("", xy=(sx - 0.5, sy),
                        xytext=(1.7, 2.0),
                        arrowprops=dict(arrowstyle="-|>", color="dimgrey",
                                        lw=0.7, connectionstyle="arc3,rad=0.1"))

        # Animated truck along first path
        path_x = 1.7 + truck_progress * (store_xs[0] - 0.5 - 1.7)
        path_y = 2.0 + truck_progress * (store_ys[0] - 2.0)
        truck = FancyBboxPatch((path_x - 0.25, path_y - 0.12), 0.5, 0.24,
                               boxstyle="round,pad=0.02",
                               linewidth=0.8, edgecolor="black", facecolor="white")
        ax.add_patch(truck)
        # Wheels
        for wx in [path_x - 0.12, path_x + 0.12]:
            wh = Circle((wx, path_y - 0.12), 0.06,
                        edgecolor="black", facecolor="black", linewidth=0.5)
            ax.add_patch(wh)
        ax.text(path_x, path_y, "DC→", ha="center", va="center", fontsize=5,
                color="dimgrey")

        ax.text(5.0, 3.7, "FOODS_3 | CA_1 | d_1577–d_1941",
                ha="center", va="center", fontsize=7, color="dimgrey",
                style="italic")

    # ── Scene 2: MC histogram fills in ───────────────────────────────────────

    hist_bins = np.linspace(np.min(scenarios) * 0.95, np.max(scenarios) * 1.05, 30)

    def draw_mc(ax, n_shown):
        ax.cla()
        sub = scenarios[:n_shown]
        ax.hist(sub, bins=hist_bins, color="lightgrey", edgecolor="black",
                linewidth=0.4)
        ax.set_title(f"Scene 2: MC (N={n_shown})", fontsize=7, fontweight="bold")
        ax.set_xlabel("Demand", fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        ax.axvline(np.mean(sub), color="black", lw=0.8, linestyle="--")
        ax.grid(True, linewidth=0.3)
        ax.set_axisbelow(True)

    # ── Scene 3: Wasserstein ball ─────────────────────────────────────────────

    sorted_s = np.sort(scenarios)
    cdf_y = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    # Empirical CDF of training data
    sorted_t = np.sort(train_w)
    cdf_t = np.arange(1, len(sorted_t) + 1) / len(sorted_t)

    def draw_wasserstein(ax, eps_frac):
        ax.cla()
        ax.plot(sorted_s, cdf_y, "k-", lw=0.8, label="Scenario CDF")
        ax.plot(sorted_t, cdf_t, "k--", lw=0.8, label="Empirical CDF")
        eps_now = eps_frac * epsilon
        # Show ball as shaded band around empirical CDF
        band_x = np.concatenate([sorted_t, sorted_t[::-1]])
        band_up = np.clip(cdf_t + eps_now / (np.std(train_w) + 1e-9), 0, 1)
        band_dn = np.clip(cdf_t - eps_now / (np.std(train_w) + 1e-9), 0, 1)
        band_y = np.concatenate([band_up, band_dn[::-1]])
        ax.fill(band_x, band_y, color="lightgrey", alpha=0.6,
                label=f"ε-ball (ε={eps_now:.0f})")
        ax.set_title(f"Scene 3: Wasserstein Ball\nε={eps_now:.1f}", fontsize=7, fontweight="bold")
        ax.set_xlabel("Demand", fontsize=6)
        ax.set_ylabel("CDF", fontsize=6)
        ax.legend(fontsize=5, loc="lower right")
        ax.grid(True, linewidth=0.3)
        ax.set_axisbelow(True)

    # ── Scene 4: CVaR convergence ─────────────────────────────────────────────

    def draw_cvar(ax, n_pts):
        ax.cla()
        x_vals = N_seq[:n_pts]
        ax.plot(x_vals, rn_cvar_seq[:n_pts], "k-", lw=1.0, label="Risk-Neutral CVaR")
        ax.plot(x_vals, dro_cvar_seq[:n_pts], "k--", lw=1.0, label="DRO-CVaR")
        if n_pts > 1:
            pct_red = (rn_cvar_seq[n_pts-1] - dro_cvar_seq[n_pts-1]) / rn_cvar_seq[n_pts-1] * 100
            ax.text(0.97, 0.97, f"CVaR↓{abs(pct_red):.1f}%",
                    transform=ax.transAxes, fontsize=6,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="black", lw=0.5))
        ax.set_title("Scene 4: CVaR Convergence", fontsize=7, fontweight="bold")
        ax.set_xlabel("N scenarios", fontsize=6)
        ax.set_ylabel("CVaR₉₅ (£)", fontsize=6)
        ax.legend(fontsize=5)
        ax.grid(True, linewidth=0.3)
        ax.set_axisbelow(True)

    # ── Scene 5: Final comparison ─────────────────────────────────────────────

    def draw_final(ax_sc_, ax_mc_, ax_ws_, ax_cv_):
        # Repurpose all panels for a final comparison table
        for ax in [ax_sc_, ax_mc_, ax_ws_, ax_cv_]:
            ax.cla()
            ax.axis("off")

        ax_sc_.set_title("Scene 5: DRO-CVaR vs Risk-Neutral Summary", fontsize=9,
                         fontweight="bold")
        rows = [
            ["Metric", "Risk-Neutral", "DRO-CVaR (λ=0.5)"],
            ["Order Q (units)", f"{rn_Q:,.0f}", f"{rn_Q * 1.08:,.0f}"],
            ["E[Cost] (£/wk)", f"{np.mean(rn_costs):,.0f}", f"{np.mean(dro_costs):,.0f}"],
            ["CVaR₉₅ (£/wk)", f"{np.percentile(rn_costs,95):,.0f}",
             f"{np.percentile(dro_costs,95):,.0f}"],
            ["Service Level", f"{np.mean(scenarios <= rn_Q):.1%}",
             f"{np.mean(scenarios <= rn_Q*1.08):.1%}"],
            ["Carbon (t/wk)", f"{0.003 * rn_Q:.1f}", f"{0.003 * rn_Q * 1.08:.1f}"],
        ]
        col_w = [0.35, 0.32, 0.33]
        x_starts = [0.05, 0.42, 0.72]
        y_start = 0.82
        row_h = 0.11
        for ri, row in enumerate(rows):
            y = y_start - ri * row_h
            for ci, (cell, xst) in enumerate(zip(row, x_starts)):
                weight = "bold" if ri == 0 else "normal"
                ax_sc_.text(xst, y, cell, transform=ax_sc_.transAxes,
                            fontsize=8, fontweight=weight,
                            va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      facecolor="lightgrey" if ri == 0 else "white",
                                      edgecolor="black", lw=0.4) if ri == 0 else None)

    # ── Animation update function ────────────────────────────────────────────

    def update(frame):
        if frame < 25:
            # Scene 1
            draw_supply_chain(ax_sc, truck_progress=frame / 24.0)
            ax_mc.cla(); ax_mc.axis("off")
            ax_ws.cla(); ax_ws.axis("off")
            ax_cv.cla(); ax_cv.axis("off")
        elif frame < 65:
            # Scene 2
            ax_sc.cla(); ax_sc.axis("off")
            ax_sc.set_title("Supply chain → scenario generation", fontsize=8,
                            color="dimgrey")
            n_shown = max(5, int((frame - 25) / 40.0 * 500))
            draw_mc(ax_mc, n_shown)
            ax_ws.cla(); ax_ws.axis("off")
            ax_cv.cla(); ax_cv.axis("off")
        elif frame < 90:
            # Scene 3
            ax_sc.cla(); ax_sc.axis("off")
            eps_frac = (frame - 65) / 25.0
            draw_wasserstein(ax_ws, eps_frac)
            ax_mc.cla(); ax_mc.axis("off")
            ax_cv.cla(); ax_cv.axis("off")
        elif frame < 120:
            # Scene 4
            ax_sc.cla(); ax_sc.axis("off")
            n_pts = max(1, int((frame - 90) / 30.0 * 30))
            draw_cvar(ax_cv, n_pts)
            ax_mc.cla(); ax_mc.axis("off")
            ax_ws.cla(); ax_ws.axis("off")
        else:
            # Scene 5
            draw_final(ax_sc, ax_mc, ax_ws, ax_cv)

        return []

    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000 // FPS, blit=False)
    save_path = save_path or ANIM_DIR / "animation_workflow.gif"
    writer = PillowWriter(fps=FPS)
    anim.save(str(save_path), writer=writer, dpi=100)
    plt.close(fig)
    return str(save_path)
