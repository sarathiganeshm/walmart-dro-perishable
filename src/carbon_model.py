"""
Carbon cap-and-trade constraint model for cold chain inventory.

Computes carbon emissions and trading credits for the DRO-CVaR optimiser.
Supports carbon cap binding checks and credit accounting.

References:
  Di et al. (2025). Annals of Operations Research.
  Li et al. (2022). Alexandria Engineering Journal, 61(10), 7979-7986.
  Zhang et al. (2025). Transportation Research Part D.
"""

import numpy as np


# Emission factors (tonnes CO2e per unit/km)
E_PER_UNIT = 0.003    # cold storage emission per unit
E_PER_KM = 0.0008     # transport emission per unit per km
AVG_KM = 50.0         # average transport distance (km)
C_CARBON = 15.0       # carbon price £/tonne


def compute_emissions(Q, e_per_unit=E_PER_UNIT, e_per_km=E_PER_KM, avg_km=AVG_KM):
    """
    Compute total carbon emissions (tonnes CO2e) for order quantity Q.

    Formula: (e_per_unit + e_per_km * avg_km / 1000) * Q

    Parameters
    ----------
    Q : float
        Order quantity (units).
    e_per_unit, e_per_km, avg_km : float
        Emission factors.

    Returns
    -------
    float : total emissions in tonnes CO2e.
    """
    factor = e_per_unit + e_per_km * avg_km / 1000.0
    return factor * Q


def compute_trading_credit(Q, carbon_cap, c_carbon=C_CARBON,
                            e_per_unit=E_PER_UNIT, e_per_km=E_PER_KM, avg_km=AVG_KM):
    """
    Compute carbon trading credit (positive = surplus credits sold, negative = penalty).

    Credit = c_carbon * (carbon_cap - emissions)

    Parameters
    ----------
    Q : float
        Order quantity.
    carbon_cap : float
        Carbon allowance in tonnes CO2e.
    c_carbon : float
        Carbon price £/tonne.

    Returns
    -------
    float : trading credit (£). Positive if under cap; negative if over cap.
    """
    emissions = compute_emissions(Q, e_per_unit, e_per_km, avg_km)
    return c_carbon * (carbon_cap - emissions)


def carbon_sweep_stats(Q, caps, c_carbon=C_CARBON,
                        e_per_unit=E_PER_UNIT, e_per_km=E_PER_KM, avg_km=AVG_KM):
    """
    Compute emissions and trading credits across a range of carbon caps.

    Parameters
    ----------
    Q : float
        Order quantity.
    caps : list of float
        Carbon cap values to evaluate.

    Returns
    -------
    list of dict with keys: cap, emissions, credit, binding.
    """
    results = []
    emissions = compute_emissions(Q, e_per_unit, e_per_km, avg_km)
    for cap in caps:
        credit = c_carbon * (cap - emissions)
        results.append({
            "cap": cap,
            "emissions": emissions,
            "credit": credit,
            "binding": emissions >= cap,
        })
    return results
