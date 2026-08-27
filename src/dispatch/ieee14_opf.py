"""IEEE 14-bus AC OPF via pandapower — power flow, LMP extraction, line loadings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

# Line 0 (bus 0–1) is the heaviest corridor in the base case; cap it to expose congestion.
CONGESTION_LINE_INDEX = 0
CONGESTION_CAP_MULT = 0.96
BINDING_LOADING_THRESHOLD = 98.0
LOAD_SCALE_MIN = 0.9
LOAD_SCALE_MAX = 1.55


@dataclass
class OpfResult:
    """AC OPF solution with bus LMPs and line loadings."""

    status: str
    converged: bool
    total_cost: float
    bus_lmps: pd.Series  # index = pandapower bus, values in $/MWh
    line_loadings: pd.DataFrame  # line index, from_bus, to_bus, loading_percent, ...
    load_scale: float

    def congested_lines(self, threshold: float = BINDING_LOADING_THRESHOLD) -> pd.Index:
        """Return line indices at or above the binding threshold."""
        mask = self.line_loadings["loading_percent"] >= threshold
        return self.line_loadings.index[mask]


def load_ieee14_case() -> pp.pandapowerNet:
    """Load the IEEE 14-bus test case with polynomial generator costs."""
    net = pn.case14()
    net["_base_load_p_mw"] = net.load.p_mw.copy()
    return net


def prepare_congestion_case(
    net: pp.pandapowerNet,
    congestion_line: int = CONGESTION_LINE_INDEX,
    cap_mult: float = CONGESTION_CAP_MULT,
) -> pp.pandapowerNet:
    """Tighten a transmission corridor so peak load can bind under forecast error."""
    if "_base_load_p_mw" not in net:
        net["_base_load_p_mw"] = net.load.p_mw.copy()

    net.load.p_mw = net["_base_load_p_mw"].copy()
    net.line["max_loading_percent"] = 100.0
    pp.runpp(net, calculate_voltage_angles=True, numba=False)
    base_flow_i = abs(float(net.res_line.at[congestion_line, "i_ka"]))
    net.line.at[congestion_line, "max_i_ka"] = base_flow_i * cap_mult
    net["_congestion_line"] = congestion_line
    return net


def apply_load_scale(
    net: pp.pandapowerNet,
    scale: float,
    bus_weights: pd.Series | None = None,
) -> None:
    """Scale active loads relative to stored base values."""
    if "_base_load_p_mw" not in net:
        net["_base_load_p_mw"] = net.load.p_mw.copy()

    scale = float(np.clip(scale, LOAD_SCALE_MIN, LOAD_SCALE_MAX))
    base = net["_base_load_p_mw"]
    if bus_weights is None:
        net.load.p_mw = base * scale
        return

    weights = bus_weights.reindex(net.load.bus).fillna(1.0).to_numpy(dtype=float)
    net.load.p_mw = base.to_numpy(dtype=float) * scale * weights


def run_power_flow(net: pp.pandapowerNet) -> bool:
    """Run Newton-Raphson AC power flow."""
    pp.runpp(net, calculate_voltage_angles=True, init="auto", numba=False)
    return bool(net.converged)


def run_ac_opf(net: pp.pandapowerNet) -> bool:
    """Run AC optimal power flow after a warm-start power flow."""
    run_power_flow(net)
    try:
        pp.runopp(
            net,
            calculate_voltage_angles=True,
            init="pf",
            numba=False,
            suppress_warnings=True,
        )
    except pp.auxiliary.OPFNotConverged:
        net["OPF_converged"] = False
        return False
    return bool(getattr(net, "OPF_converged", True))


def extract_bus_lmps(net: pp.pandapowerNet) -> pd.Series:
    """Extract bus LMPs (LAM_P shadow prices) from the solved OPF case.

    Same economic meaning as Day-1 cvxpy power-balance duals in
    ``economic_dispatch.py``, but locational: when a line limit binds,
    LMPs diverge across buses to reflect congestion.
    """
    if not hasattr(net, "_ppc_opf") or net._ppc_opf is None:
        raise RuntimeError("AC OPF has not been run — call run_ac_opf() first.")

    ppc_bus = net._ppc_opf["bus"]
    lam_p = ppc_bus[:, 13]  # PYPOWER LAM_P column
    lookup = net._pd2ppc_lookups["bus"]

    lmps: dict[int, float] = {}
    for pp_bus in net.bus.index:
        ppc_idx = int(lookup[pp_bus])
        if ppc_idx >= 0:
            lmps[int(pp_bus)] = float(lam_p[ppc_idx])

    return pd.Series(lmps, name="lmp_usd_per_mwh").sort_index()


def extract_line_loadings(net: pp.pandapowerNet) -> pd.DataFrame:
    """Return line loading results joined with topology metadata."""
    cols = ["loading_percent", "p_from_mw", "p_to_mw", "i_ka"]
    loading = net.res_line[cols].copy()
    loading["from_bus"] = net.line["from_bus"].values
    loading["to_bus"] = net.line["to_bus"].values
    loading["name"] = net.line["name"].values
    return loading


def solve_opf_at_scale(net: pp.pandapowerNet, scale: float) -> OpfResult:
    """Apply a load scale factor, run AC OPF, and package results."""
    apply_load_scale(net, scale)
    converged = run_ac_opf(net)
    cost = float("nan")
    if converged and hasattr(net, "_ppc_opf") and net._ppc_opf is not None and "f" in net._ppc_opf:
        cost = float(net._ppc_opf["f"])
    elif converged and hasattr(net, "res_cost"):
        cost = float(net.res_cost)

    bus_lmps = extract_bus_lmps(net) if converged else pd.Series(dtype=float)
    line_loadings = extract_line_loadings(net) if converged else pd.DataFrame()

    return OpfResult(
        status="optimal" if converged else "failed",
        converged=converged,
        total_cost=cost,
        bus_lmps=bus_lmps,
        line_loadings=line_loadings,
        load_scale=scale,
    )


def compare_binding_status(
    forecast: OpfResult,
    actual: OpfResult,
    threshold: float = BINDING_LOADING_THRESHOLD,
) -> pd.DataFrame:
    """Flag lines binding in the actual case but not under forecast load."""
    if not forecast.converged or not actual.converged:
        raise ValueError("Both OPF results must be converged for binding comparison.")

    rows: list[dict[str, object]] = []
    all_lines = forecast.line_loadings.index.union(actual.line_loadings.index)

    for line_idx in all_lines:
        fc_pct = float(forecast.line_loadings.at[line_idx, "loading_percent"])
        ac_pct = float(actual.line_loadings.at[line_idx, "loading_percent"])
        fc_bind = fc_pct >= threshold
        ac_bind = ac_pct >= threshold
        rows.append(
            {
                "line": int(line_idx),
                "from_bus": int(actual.line_loadings.at[line_idx, "from_bus"]),
                "to_bus": int(actual.line_loadings.at[line_idx, "to_bus"]),
                "forecast_loading_pct": fc_pct,
                "actual_loading_pct": ac_pct,
                "forecast_binding": fc_bind,
                "actual_binding": ac_bind,
                "hidden_congestion": ac_bind and not fc_bind,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net)
    result = solve_opf_at_scale(net, scale=1.0)

    print(f"OPF status: {result.status}")
    print(f"Total cost: ${result.total_cost:,.2f}")
    print("\nBus LMPs ($/MWh):")
    print(result.bus_lmps.describe())
    print("\nTop line loadings (%):")
    top = result.line_loadings["loading_percent"].sort_values(ascending=False).head(5)
    print(top)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    lmp_path = output_dir / "ieee14_bus_lmps.csv"
    pd.DataFrame({"bus": result.bus_lmps.index, "lmp_usd_per_mwh": result.bus_lmps.values}).to_csv(
        lmp_path, index=False
    )
    line_path = output_dir / "ieee14_line_loadings.csv"
    result.line_loadings.to_csv(line_path)
    print(f"\nSaved LMPs to {lmp_path}")
    print(f"Saved line loadings to {line_path}")


if __name__ == "__main__":
    main()
