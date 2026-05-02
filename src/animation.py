"""
Five-scene animation showing the DRO-CVaR workflow from a real supply
chain (Walmart distribution centre -> stores -> customers) through Monte
Carlo scenario generation, Wasserstein ambiguity set construction, and
CVaR convergence to final policy comparison.

Output: results/animation/animation_workflow.gif
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon, Circle
from pathlib import Path

ANIM_DIR = Path(__file__).parent.parent / "results" / "animation"
ANIM_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 140
SCENE = {
    "intro":   (0, 25),
    "mc":      (25, 65),
    "wasser":  (65, 90),
    "cvar":    (90, 120),
    "compare": (120, 140),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

SCENE_TITLES = {
    "intro":
        "Stage 1: Walmart supply chain — uncertain customer demand drives ordering decisions",
    "mc":
        "Stage 2: Monte Carlo bootstrap — sampling 500 scenarios from 42-week empirical distribution",
    "wasser":
        "Stage 3: Wasserstein ambiguity set — robustifying against finite-sample uncertainty",
    "cvar":
        "Stage 4: DRO-CVaR optimisation — minimising worst-case tail cost via CBC solver",
    "compare":
        "Stage 5: Risk-neutral vs DRO-CVaR — comparing policies on tail-risk metrics",
}


def in_scene(f, name):
    s, e = SCENE[name]
    return s <= f < e


def scene_progress(f, name):
    s, e = SCENE[name]
    if f < s:
        return 0.0
    if f >= e:
        return 1.0
    return (f - s) / (e - s)


def current_scene(f):
    for name, (s, e) in SCENE.items():
        if s <= f < e:
            return name
    return "compare"


def scenario_costs(Q, sc, p):
    return (p["c_order"] * Q
            + (p["c_hold"] + p["c_waste"]) * np.maximum(Q - sc, 0)
            + p["c_short"] * np.maximum(sc - Q, 0))


def cvar95(c):
    v = np.percentile(c, 95)
    tail = c[c > v]
    return v + (np.mean(tail - v) if len(tail) else 0.0)


def draw_supply_chain(ax, frame, train_w, scenarios):
    ax.cla()
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ── Distribution Centre ───────────────────────────────────────────────
    wh = Rectangle((0.5, 1.2), 1.6, 1.6, linewidth=1.0,
                   edgecolor="black", facecolor="white")
    ax.add_patch(wh)
    roof = Polygon([(0.5, 2.8), (1.3, 3.5), (2.1, 2.8)], closed=True,
                   edgecolor="black", facecolor="lightgrey", linewidth=0.9)
    ax.add_patch(roof)
    door = Rectangle((1.15, 1.2), 0.3, 0.7, linewidth=0.8,
                     edgecolor="black", facecolor="black")
    ax.add_patch(door)
    # Inventory boxes (3 wide x 2 tall)
    for row in range(2):
        for col in range(3):
            bx = Rectangle((0.58 + col * 0.45, 1.95 + row * 0.37),
                            0.38, 0.28, linewidth=0.5,
                            edgecolor="black", facecolor="white")
            ax.add_patch(bx)
    ax.text(1.3, 0.88, "Walmart DC\n(FOODS_3)", ha="center", va="top",
            fontsize=7, fontweight="bold")

    # ── Three stores ──────────────────────────────────────────────────────
    store_xs = [5.5, 7.0, 8.5]
    store_labels = ["CA_1", "CA_2", "CA_3"]
    store_y = 1.6
    for sx, lbl in zip(store_xs, store_labels):
        sb = Rectangle((sx, store_y), 0.9, 1.1, linewidth=0.8,
                       edgecolor="black", facecolor="white")
        ax.add_patch(sb)
        sr = Polygon([(sx, store_y + 1.1), (sx + 0.45, store_y + 1.6),
                      (sx + 0.9, store_y + 1.1)], closed=True,
                     edgecolor="black", facecolor="lightgrey", linewidth=0.7)
        ax.add_patch(sr)
        sd = Rectangle((sx + 0.3, store_y), 0.3, 0.5, linewidth=0.7,
                       edgecolor="black", facecolor="black")
        ax.add_patch(sd)
        ax.text(sx + 0.45, store_y - 0.2, lbl, ha="center", va="top",
                fontsize=7)

    # ── Customer figures ──────────────────────────────────────────────────
    cx_positions = np.linspace(10.8, 12.6, 5)
    for cx in cx_positions:
        head = Circle((cx, 2.8), 0.12, linewidth=0.6,
                      edgecolor="black", facecolor="black")
        ax.add_patch(head)
        body = Polygon([(cx - 0.12, 2.68), (cx + 0.12, 2.68),
                        (cx + 0.18, 2.1), (cx - 0.18, 2.1)], closed=True,
                       edgecolor="black", facecolor="white", linewidth=0.6)
        ax.add_patch(body)
    ax.text(11.7, 1.95, "Customers\n(uncertain demand)",
            ha="center", va="top", fontsize=7, style="italic")

    # ── Animated truck (visible after frame 5) ────────────────────────────
    if frame >= 5:
        phase = (frame % 30) / 30.0
        tx = 2.3 + phase * 2.8
        ty = 1.7
        cab = Rectangle((tx, ty), 0.6, 0.5, linewidth=0.8,
                        edgecolor="black", facecolor="white")
        ax.add_patch(cab)
        box_r = Rectangle((tx + 0.6, ty + 0.1), 0.3, 0.4, linewidth=0.7,
                          edgecolor="black", facecolor="white")
        ax.add_patch(box_r)
        for wx in [tx + 0.15, tx + 0.5, tx + 0.78]:
            wh = Circle((wx, ty), 0.08, edgecolor="black", facecolor="black",
                        linewidth=0.5)
            ax.add_patch(wh)

    # ── Arrows ────────────────────────────────────────────────────────────
    a_main = FancyArrowPatch((2.1, 2.0), (5.5, 2.15),
                             arrowstyle="-|>", mutation_scale=10,
                             color="black", linewidth=1.0)
    ax.add_patch(a_main)
    for sx in store_xs:
        a = FancyArrowPatch((sx + 0.9, 2.15), (10.6, 2.45),
                            arrowstyle="-|>", mutation_scale=8,
                            color="dimgrey", linewidth=0.7,
                            connectionstyle="arc3,rad=0.05")
        ax.add_patch(a)

    # ── Demand counter ────────────────────────────────────────────────────
    if not in_scene(frame, "intro"):
        if in_scene(frame, "mc"):
            n_show = min(int(scene_progress(frame, "mc") * 500), 500)
        else:
            n_show = 500
        ax.text(11.7, 0.9,
                f"Demand scenario\nsamples: {n_show} / 500",
                ha="center", va="top", fontsize=7, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", linewidth=0.6))

    # ── Dynamic scene title ───────────────────────────────────────────────
    sc_name = current_scene(frame)
    ax.text(7.0, 4.25, SCENE_TITLES[sc_name],
            ha="center", va="center", fontsize=8, style="italic",
            color="dimgrey")


def draw_mc(ax, frame, train_w, scenarios):
    ax.cla()
    visible = (in_scene(frame, "mc") or in_scene(frame, "wasser")
               or in_scene(frame, "cvar") or in_scene(frame, "compare"))
    if not visible:
        ax.axis("off")
        return
    n_max = max(20, min(int(500 * scene_progress(frame, "mc")), 500)) \
        if in_scene(frame, "mc") else 500
    sub = scenarios[:n_max]
    bins = np.linspace(np.min(train_w) * 0.92, np.max(train_w) * 1.08, 28)
    ax.hist(train_w, bins=bins, density=True, color="white",
            edgecolor="black", hatch="//", linewidth=0.5, alpha=0.9,
            label="Training (n=42)")
    ax.hist(sub, bins=bins, density=True, color="lightgrey",
            edgecolor="black", linewidth=0.4, alpha=0.55,
            label=f"Scenarios (N={n_max})")
    ax.axvline(float(np.mean(train_w)), color="black", linestyle="--",
               linewidth=0.9, label="Mean")
    ax.set_xlabel("Weekly demand (units)", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_title("(a) Monte Carlo scenario generation",
                 fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3)
    ax.set_axisbelow(True)


def draw_wasser(ax, frame, train_w, epsilon):
    ax.cla()
    visible = (in_scene(frame, "wasser") or in_scene(frame, "cvar")
               or in_scene(frame, "compare"))
    if not visible:
        ax.axis("off")
        return
    sorted_t = np.sort(train_w)
    n = len(sorted_t)
    cdf_y = np.arange(1, n + 1) / n
    ax.step(sorted_t, cdf_y, "k-", linewidth=1.0, where="post",
            label="Empirical CDF $\\hat{P}_N$")
    eps_frac = scene_progress(frame, "wasser") if in_scene(frame, "wasser") else 1.0
    eps_now = eps_frac * epsilon
    ax.fill_betweenx(cdf_y,
                     sorted_t - eps_now * 0.5,
                     sorted_t + eps_now * 0.5,
                     color="lightgrey", alpha=0.5,
                     label=f"$\\varepsilon$-ball ({eps_now:.0f})")
    ax.set_xlabel("Demand (units)", fontsize=7)
    ax.set_ylabel("CDF", fontsize=7)
    ax.set_title("(b) Wasserstein ambiguity set $B_\\varepsilon(\\hat{{P}}_N)$",
                 fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3)
    ax.set_axisbelow(True)


def draw_conv(ax, frame, scenarios, Q_rn, Q_dro, params):
    ax.cla()
    visible = in_scene(frame, "cvar") or in_scene(frame, "compare")
    if not visible:
        ax.axis("off")
        return
    n_max = max(20, min(int(500 * scene_progress(frame, "cvar")), 500)) \
        if in_scene(frame, "cvar") else 500
    N_seq = np.unique(np.geomspace(20, n_max, 30).astype(int))
    rng = np.random.default_rng(42)
    rn_vals, dro_vals = [], []
    for n in N_seq:
        sub = scenarios[rng.choice(500, min(n, 500), replace=False)]
        rn_vals.append(cvar95(scenario_costs(Q_rn, sub, params)) / 1000)
        dro_vals.append(cvar95(scenario_costs(Q_dro, sub, params)) / 1000)
    ax.plot(N_seq, rn_vals, "k-", linewidth=1.1, label="Risk-neutral")
    ax.plot(N_seq, dro_vals, "k--", linewidth=1.1, label="DRO-CVaR")
    ax.fill_between(N_seq, rn_vals, dro_vals, color="lightgrey", alpha=0.6)
    if len(rn_vals) > 1 and rn_vals[-1] > 0:
        imp = (rn_vals[-1] - dro_vals[-1]) / rn_vals[-1] * 100
        ax.text(0.97, 0.05,
                f"CVaR$_{{95}}$ reduction\n{imp:.1f}%",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", linewidth=0.6))
    ax.set_xlabel("N scenarios", fontsize=7)
    ax.set_ylabel("CVaR$_{0.95}$ (£000/week)", fontsize=7)
    ax.set_title("(c) CVaR convergence", fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3)
    ax.set_axisbelow(True)


def generate_animation(scenarios, train_w, results, epsilon, params,
                        output_path=None):
    """Build the 5-scene animation. Saves to results/animation/animation_workflow.gif"""
    Q_rn  = float(results[0.0]["Q"])
    Q_dro = float(results[0.5]["Q"])

    fig = plt.figure(figsize=(13, 7), facecolor="white")
    gs = fig.add_gridspec(2, 3,
                          height_ratios=[1, 1.1],
                          width_ratios=[1.2, 1, 1],
                          hspace=0.35, wspace=0.3)
    ax_sc   = fig.add_subplot(gs[0, :])
    ax_mc   = fig.add_subplot(gs[1, 0])
    ax_wd   = fig.add_subplot(gs[1, 1])
    ax_conv = fig.add_subplot(gs[1, 2])

    def draw_frame(frame):
        draw_supply_chain(ax_sc, frame, train_w, scenarios)
        draw_mc(ax_mc, frame, train_w, scenarios)
        draw_wasser(ax_wd, frame, train_w, epsilon)
        draw_conv(ax_conv, frame, scenarios, Q_rn, Q_dro, params)
        fig.suptitle(
            f"Wasserstein DRO-CVaR for Perishable Inventory  •  "
            f"Walmart M5 FOODS 3  •  Frame {frame + 1}/{N_FRAMES}",
            fontsize=11, fontweight="bold", y=0.98,
        )
        return []

    ani = animation.FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                                   interval=100, blit=False)
    out = output_path or str(ANIM_DIR / "animation_workflow.gif")
    ani.save(out, writer="pillow", fps=8, dpi=100)
    plt.close(fig)
    return out
