"""SPO+ surrogate loss for predict-then-optimize (Elmachtoub & Grigas).

The battery arbitrage LP maximizes sum_t p_t * (discharge_t - charge_t).  SPO+
trains the price forecaster on the *decision* loss (regret in the downstream LP),
not on MAE.  Asymmetric errors matter: overpredicting a spike vs missing one
changes charge/discharge timing and dollar outcomes differently.
"""

from __future__ import annotations

import numpy as np

from src.dispatch.battery_arbitrage import BatteryConfig, solve_battery_arbitrage


def net_export_vector(prices: np.ndarray, config: BatteryConfig | None = None) -> np.ndarray:
    """Return hourly net export (discharge - charge) from the arbitrage LP."""
    result = solve_battery_arbitrage(prices, config)
    charge = result.schedule["charge_mw"].to_numpy(dtype=float)
    discharge = result.schedule["discharge_mw"].to_numpy(dtype=float)
    return discharge - charge


def spo_plus_loss(
    p_hat: np.ndarray,
    p_true: np.ndarray,
    config: BatteryConfig | None = None,
) -> float:
    """SPO+ surrogate loss for a maximization LP with price coefficients p.

    For minimization with cost c, the paper defines
    max_w (2 c_hat - c)^T w - c_hat^T w.  With c = -p this becomes
    max_z (p - p_hat)^T z over the battery feasible set.
    """
    p_hat = np.asarray(p_hat, dtype=float).reshape(-1)
    p_true = np.asarray(p_true, dtype=float).reshape(-1)
    offset = p_true - p_hat
    z_star = net_export_vector(offset, config)
    return float(np.dot(offset, z_star))


def spo_plus_price_subgradient(
    p_hat: np.ndarray,
    p_true: np.ndarray,
    config: BatteryConfig | None = None,
) -> np.ndarray:
    """Subgradient of SPO+ loss w.r.t. predicted prices p_hat.

    SPO+ for maximization max_p p^T z is max_z (p_true - p_hat)^T z (equivalent to
    the paper's min-cost form with c = -p).  By the envelope theorem,
    d loss / d p_hat = -z*(p_true - p_hat).
    """
    p_hat = np.asarray(p_hat, dtype=float).reshape(-1)
    p_true = np.asarray(p_true, dtype=float).reshape(-1)
    return -net_export_vector(p_true - p_hat, config)


def decision_regret(
    p_hat: np.ndarray,
    p_true: np.ndarray,
    config: BatteryConfig | None = None,
) -> float:
    """True decision regret: perfect-foresight revenue minus forecast-plan revenue."""
    from src.dispatch.battery_arbitrage import realized_revenue

    p_hat = np.asarray(p_hat, dtype=float).reshape(-1)
    p_true = np.asarray(p_true, dtype=float).reshape(-1)
    pf = solve_battery_arbitrage(p_true, config)
    fc = solve_battery_arbitrage(p_hat, config)
    fc_revenue = realized_revenue(fc.schedule, p_true)
    return float(pf.revenue - fc_revenue)
