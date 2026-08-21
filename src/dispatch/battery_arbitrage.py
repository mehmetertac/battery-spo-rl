"""Battery storage arbitrage LP: maximize day-ahead price revenue."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

SOLVER_FALLBACKS = ("ECOS", "CLARABEL", "SCS")


@dataclass(frozen=True)
class BatteryConfig:
    power_mw: float = 100.0
    energy_mwh: float = 400.0
    soc_min_mwh: float = 0.0
    round_trip_efficiency: float = 0.90
    initial_soc_mwh: float = 200.0
    terminal_soc_min_mwh: float = 200.0

    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)


@dataclass
class ArbitrageResult:
    schedule: pd.DataFrame  # hour, charge_mw, discharge_mw, soc_mwh, price_used
    revenue: float
    status: str

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.schedule.to_csv(path, index=False)


def _resolve_solver(solver_name: str) -> str:
    solver_name = solver_name.upper()
    if solver_name in cp.installed_solvers():
        return solver_name

    for candidate in SOLVER_FALLBACKS:
        if candidate in cp.installed_solvers():
            return candidate

    raise RuntimeError(
        "No supported LP solver found. Install cvxpy with ECOS, CLARABEL, or SCS."
    )


def solve_battery_arbitrage(
    prices: np.ndarray | pd.Series,
    config: BatteryConfig | None = None,
    solver: str = "ECOS",
) -> ArbitrageResult:
    """Solve hourly battery arbitrage LP and return schedule plus revenue."""
    if config is None:
        config = BatteryConfig()

    price = np.asarray(prices, dtype=float).reshape(-1)
    n_hours = price.shape[0]
    if n_hours == 0:
        raise ValueError("At least one hourly price is required.")

    charge = cp.Variable(n_hours, nonneg=True)
    discharge = cp.Variable(n_hours, nonneg=True)
    soc = cp.Variable(n_hours + 1)

    eta_c = config.eta_charge
    eta_d = config.eta_discharge

    constraints = [
        soc[0] == config.initial_soc_mwh,
        soc >= config.soc_min_mwh,
        soc <= config.energy_mwh,
        charge <= config.power_mw,
        discharge <= config.power_mw,
        soc[n_hours] >= config.terminal_soc_min_mwh,
    ]
    for t in range(n_hours):
        constraints.append(
            soc[t + 1] == soc[t] + eta_c * charge[t] - discharge[t] / eta_d
        )

    revenue_expr = cp.sum(
        cp.multiply(price, discharge - charge)
    )
    problem = cp.Problem(cp.Maximize(revenue_expr), constraints)
    solver_name = _resolve_solver(solver)
    problem.solve(solver=solver_name)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Battery arbitrage solve failed with status: {problem.status}")

    charge_vals = np.array(charge.value).reshape(-1)
    discharge_vals = np.array(discharge.value).reshape(-1)
    soc_vals = np.array(soc.value).reshape(-1)

    schedule = pd.DataFrame(
        {
            "hour": np.arange(n_hours),
            "charge_mw": charge_vals,
            "discharge_mw": discharge_vals,
            "soc_mwh": soc_vals[:-1],
            "price_used": price,
        }
    )

    return ArbitrageResult(
        schedule=schedule,
        revenue=float(problem.value),
        status=problem.status,
    )


def realized_revenue(
    schedule: pd.DataFrame | ArbitrageResult,
    actual_prices: np.ndarray | pd.Series,
) -> float:
    """Compute revenue when executing a fixed schedule against actual prices."""
    if isinstance(schedule, ArbitrageResult):
        frame = schedule.schedule
    else:
        frame = schedule

    actual = np.asarray(actual_prices, dtype=float).reshape(-1)
    if len(actual) != len(frame):
        raise ValueError("actual_prices length must match schedule length.")

    charge = frame["charge_mw"].to_numpy(dtype=float)
    discharge = frame["discharge_mw"].to_numpy(dtype=float)
    return float(np.sum(actual * (discharge - charge)))
