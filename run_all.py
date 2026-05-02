"""
Master runner: loads data, runs DRO-CVaR sweeps, generates all figures and animation.
Usage: python run_all.py
Requires M5 CSV files in data/ — see README.md for download instructions.
"""

import numpy as np
from src.data_loader import load_data
from src.scenario_generator import generate_scenarios, wasserstein_radius
from src.dro_cvar_lp import solve_dro_cvar, PARAMS
from src.sensitivity_analysis import run_full_sensitivity
from src.plotting import generate_all_figures
from src.animation import generate_animation

SEED = 42
N_SCENARIOS = 500


def main():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("Loading M5 data...")
    train_w, holdout_w, norm_stats = load_data()
    print(f"  Train weeks: {len(train_w)}, Holdout: {len(holdout_w)}")
    print(f"  Mean demand: {np.mean(train_w):,.0f}  Std: {np.std(train_w, ddof=1):,.0f}")
    print(f"  Shapiro-Wilk p={norm_stats['shapiro_p']:.4f}  "
          f"D'Agostino p={norm_stats['dagostino_p']:.4f}")

    # ── 2. Generate scenarios & Wasserstein radius ───────────────────────────
    print("\nGenerating scenarios...")
    scenarios = generate_scenarios(train_w, N=N_SCENARIOS, seed=SEED)
    epsilon = wasserstein_radius(train_w, N=N_SCENARIOS)
    print(f"  ε (Wasserstein radius) = {epsilon:.4f}")

    # ── 3. Risk-neutral baseline ──────────────────────────────────────────────
    print("\nSolving risk-neutral LP (λ=0)...")
    rn = solve_dro_cvar(scenarios, epsilon, lam=0.0, **PARAMS)
    print(f"  RN Q={rn['Q']:,.0f}  E[cost]=£{rn['expected_cost']:,.0f}"
          f"  CVaR95=£{rn['CVaR95']:,.0f}  SL={rn['service_level']:.1%}")

    # ── 4. DRO λ=0.5 ─────────────────────────────────────────────────────────
    print("\nSolving DRO-CVaR LP (λ=0.5)...")
    dro = solve_dro_cvar(scenarios, epsilon, lam=0.5, **PARAMS)
    print(f"  DRO Q={dro['Q']:,.0f}  E[cost]=£{dro['expected_cost']:,.0f}"
          f"  CVaR95=£{dro['CVaR95']:,.0f}  SL={dro['service_level']:.1%}"
          f"  Carbon={dro['carbon']:.2f} t/wk")

    # ── 5. Full sensitivity sweeps ────────────────────────────────────────────
    print("\nRunning sensitivity sweeps...")
    lam_results, cap_results = run_full_sensitivity(
        train_w, holdout_w, scenarios, epsilon, **PARAMS
    )
    print(f"  λ sweep: {len(lam_results)} solutions saved.")
    print(f"  Carbon cap sweep: {len(cap_results)} solutions saved.")

    # ── 6. Generate all figures ───────────────────────────────────────────────
    print("\nGenerating B&W figures...")
    fig_paths = generate_all_figures(
        train_w, scenarios, norm_stats, lam_results, rn, dro, cap_results
    )
    for p in fig_paths:
        print(f"  Saved: {p}")

    # ── 7. Generate animation ─────────────────────────────────────────────────
    print("\nRendering 140-frame animation (this may take ~60s)...")
    anim_path = generate_animation(
        scenarios=scenarios,
        train_w=train_w,
        results={0.0: rn, 0.5: dro},
        epsilon=epsilon,
        params=PARAMS,
    )
    print(f"  Saved: {anim_path}")

    print("\n✓ All outputs written to results/")


if __name__ == "__main__":
    main()
