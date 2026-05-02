"""
Monte Carlo scenario generation with Wasserstein radius calibration.
Bootstrap resampling from training demand with 5% Gaussian noise.

References:
  Kim, Y.G. & Chung, B.D. (2024). Omega, 127.
  Gao, R., Chen, X. & Kleywegt, A.J. (2024). Operations Research, 72(3), 1177-1191.
"""

import numpy as np


def generate_scenarios(train_w, N=500, seed=42):
    """
    Generate N demand scenarios via bootstrap + 5% Gaussian noise.

    Procedure:
      1. Bootstrap N samples (with replacement) from train_w.
      2. Add Gaussian noise: noise_std = 0.05 * std(train_w).
      3. Clip to [0.7*min(train_w), 1.3*max(train_w)].

    Parameters
    ----------
    train_w : np.ndarray
        Weekly training demand observations.
    N : int
        Number of scenarios.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    scenarios : np.ndarray of shape (N,)
        Simulated weekly demand scenarios.

    References
    ----------
    Kim & Chung (2024, Omega); Gao et al. (2024, Operations Research).
    """
    rng = np.random.default_rng(seed)
    # Bootstrap
    bootstrapped = rng.choice(train_w, size=N, replace=True)
    # Add 5% Gaussian noise
    noise_std = 0.05 * np.std(train_w, ddof=1)
    noise = rng.normal(0, noise_std, size=N)
    scenarios = bootstrapped + noise
    # Clip
    lo = 0.7 * np.min(train_w)
    hi = 1.3 * np.max(train_w)
    scenarios = np.clip(scenarios, lo, hi)
    return scenarios.astype(float)


def wasserstein_radius(train_w, N=500):
    """
    Calibrate Wasserstein ball radius epsilon.

    Formula: epsilon = sqrt(log(N) / N) * std(train_w)

    This is the data-driven radius from Mohajerin Esfahani & Kuhn (2018) type-1
    Wasserstein calibration, as applied in Kim & Chung (2024) and Gao et al. (2024).

    Parameters
    ----------
    train_w : np.ndarray
        Weekly training demand observations.
    N : int
        Number of scenarios used in the Monte Carlo step.

    Returns
    -------
    epsilon : float
        Wasserstein radius.

    References
    ----------
    Mohajerin Esfahani & Kuhn (2018, Math. Programming);
    Kim & Chung (2024, Omega); Gao et al. (2024, Operations Research).
    """
    sigma = np.std(train_w, ddof=1)
    epsilon = np.sqrt(np.log(N) / N) * sigma
    return float(epsilon)
