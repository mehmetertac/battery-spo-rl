"""Economic dispatch LP toy: minimize generation cost subject to demand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

SOLVER_FALLBACKS = ("ECOS", "CLARABEL", "SCS")


@dataclass(frozen=True)
class Generator:
    name: str
    marginal_cost: float  # $/MWh
    pmax: float  # MW


@dataclass
class DispatchResult:
    dispatch: pd.DataFrame  # columns: hour, generator, mw
    total_cost: float
    lmps: pd.Series  # dual of power-balance constraint per hour ($/MWh)
    status: str

    def dispatch_wide(self) -> pd.DataFrame:
        """Return dispatch as hour x generator matrix."""
        return self.dispatch.pivot(index="hour", columns="generator", values="mw").fillna(0.0)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dispatch.to_csv(path, index=False)


def default_generators() -> list[Generator]:
    return [
        Generator(name="gen0_cheap", marginal_cost=20.0, pmax=100.0),
        Generator(name="gen1_mid", marginal_cost=35.0, pmax=80.0),
        Generator(name="gen2_expensive", marginal_cost=50.0, pmax=50.0),
    ]


def default_demand(hours: int = 24) -> np.ndarray:
    """Sinusoidal 24h demand profile, base ~120 MW, peak ~200 MW."""
    t = np.arange(hours)
    return 120.0 + 40.0 * np.sin(2 * np.pi * (t - 6) / hours) + 20.0 * np.sin(4 * np.pi * t / hours)


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


def solve_economic_dispatch(
    generators: list[Generator],
    demand_mw: np.ndarray | pd.Series,
    solver: str = "ECOS",
) -> DispatchResult:
    """Solve hourly economic dispatch and return dispatch schedule plus LMPs."""
    demand = np.asarray(demand_mw, dtype=float).reshape(-1)
    n_hours = demand.shape[0]
    n_gens = len(generators)

    if n_gens == 0:
        raise ValueError("At least one generator is required.")

    p = cp.Variable((n_gens, n_hours), nonneg=True)
    costs = np.array([g.marginal_cost for g in generators])
    pmax = np.array([g.pmax for g in generators])

    objective = cp.Minimize(cp.sum(cp.multiply(costs[:, None], p)))
    balance_cons = [cp.sum(p[:, t]) == demand[t] for t in range(n_hours)]
    capacity_cons = [p[g, t] <= pmax[g] for g in range(n_gens) for t in range(n_hours)]
    constraints = balance_cons + capacity_cons

    problem = cp.Problem(objective, constraints)
    solver_name = _resolve_solver(solver)
    problem.solve(solver=solver_name)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Dispatch solve failed with status: {problem.status}")

    dispatch_values = np.array(p.value)
    rows: list[dict[str, object]] = []
    for hour in range(n_hours):
        for gen_idx, generator in enumerate(generators):
            rows.append(
                {
                    "hour": hour,
                    "generator": generator.name,
                    "mw": float(dispatch_values[gen_idx, hour]),
                }
            )

    lmps = pd.Series(
        [-float(c.dual_value) for c in balance_cons],
        index=range(n_hours),
        name="lmp",
    )

    return DispatchResult(
        dispatch=pd.DataFrame(rows),
        total_cost=float(problem.value),
        lmps=lmps,
        status=problem.status,
    )


def marginal_generator_cost(result: DispatchResult, generators: list[Generator]) -> pd.Series:
    """Return the marginal cost of the most expensive dispatched generator each hour."""
    wide = result.dispatch_wide()
    gen_by_name = {g.name: g for g in generators}
    marginal_costs: list[float] = []

    for hour in wide.index:
        active = wide.loc[hour]
        dispatched = active[active > 1e-6]
        if dispatched.empty:
            marginal_costs.append(np.nan)
            continue
        marginal_gen = max(dispatched.index, key=lambda name: gen_by_name[name].marginal_cost)
        marginal_costs.append(gen_by_name[marginal_gen].marginal_cost)

    return pd.Series(marginal_costs, index=wide.index, name="marginal_cost")


def verify_lmps(
    result: DispatchResult,
    generators: list[Generator],
    atol: float = 1e-4,
) -> pd.DataFrame:
    """Compare LMP duals to marginal generator cost hour by hour."""
    expected = marginal_generator_cost(result, generators)
    comparison = pd.DataFrame(
        {
            "lmp": result.lmps,
            "marginal_cost": expected,
        }
    )
    comparison["abs_error"] = (comparison["lmp"] - comparison["marginal_cost"]).abs()
    comparison["matches"] = comparison["abs_error"] <= atol
    return comparison


def main() -> None:
    generators = default_generators()
    demand = default_demand()
    result = solve_economic_dispatch(generators, demand)

    print(f"Status: {result.status}")
    print(f"Total cost: ${result.total_cost:,.2f}")
    print("\nLMP summary ($/MWh):")
    print(result.lmps.describe())

    verification = verify_lmps(result, generators)
    n_match = int(verification["matches"].sum())
    print(f"\nLMP verification: {n_match}/{len(verification)} hours match marginal cost")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    result.to_csv(output_dir / "dispatch_day1.csv")
    print(f"\nSaved dispatch to {output_dir / 'dispatch_day1.csv'}")


if __name__ == "__main__":
    main()
