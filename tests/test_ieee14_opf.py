"""Tests for IEEE 14-bus AC OPF and load-forecast scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dispatch.ieee14_opf import (
    BINDING_LOADING_THRESHOLD,
    CONGESTION_LINE_INDEX,
    compare_binding_status,
    extract_bus_lmps,
    load_ieee14_case,
    prepare_congestion_case,
    run_ac_opf,
    run_power_flow,
    solve_opf_at_scale,
)
from src.eval.opf_figures import generate_opf_figures
from src.eval.opf_scenario import run_opf_scenarios
from src.forecast.load_forecaster import (
    build_system_load_profile,
    forecast_vs_actual_scales,
    load_forecast_mae,
)


def test_case14_power_flow_converges() -> None:
    net = load_ieee14_case()
    assert run_power_flow(net)


def test_ac_opf_converges_with_costs() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net)
    assert run_ac_opf(net)
    lmps = extract_bus_lmps(net)
    assert len(lmps) == len(net.bus)
    assert np.all(np.isfinite(lmps.to_numpy()))


def test_lmp_extraction_shape() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net)
    solve_opf_at_scale(net, scale=1.0)
    lmps = extract_bus_lmps(net)
    assert len(lmps) >= 14
    assert float(lmps.abs().sum()) > 0.0


def test_load_scaling_changes_total_cost() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net)
    low = solve_opf_at_scale(net, scale=0.9)
    high = solve_opf_at_scale(net, scale=1.3)
    assert high.total_cost > low.total_cost


def test_line_loadings_present() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net)
    result = solve_opf_at_scale(net, scale=1.0)
    assert "loading_percent" in result.line_loadings.columns
    assert len(result.line_loadings) == len(result.line_loadings.index)


def test_hidden_congestion_detected() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net, cap_mult=0.87)

    forecast = solve_opf_at_scale(net, scale=1.05)
    actual = solve_opf_at_scale(net, scale=1.35)
    comparison = compare_binding_status(forecast, actual, threshold=BINDING_LOADING_THRESHOLD)

    hidden = comparison[comparison["hidden_congestion"]]
    assert not hidden.empty, "Expected hidden congestion on at least one line"
    assert int(hidden["line"].iloc[0]) == CONGESTION_LINE_INDEX


def test_load_forecast_mae_non_negative() -> None:
    profile = build_system_load_profile(days=30, seed=0)
    eval_day = profile.index.normalize().unique()[-1]
    actual, forecast = forecast_vs_actual_scales(profile, eval_day)
    mae = load_forecast_mae(actual, forecast)
    assert mae >= 0.0


def test_opf_scenario_smoke_two_hours() -> None:
    df = run_opf_scenarios(n_hours=2, profile_days=30, seed=1)
    assert len(df) > 0
    assert set(df["approach"]) == {"forecast_load", "actual_load"}
    summary = df.attrs["summary"]
    assert len(summary) == 2


def test_opf_figures_write_png(tmp_path) -> None:
    lmp_path, scenario_path = generate_opf_figures(tmp_path, profile_days=30, seed=42)
    assert lmp_path.exists() and lmp_path.stat().st_size > 0
    assert scenario_path.exists() and scenario_path.stat().st_size > 0


def test_compare_binding_status_columns() -> None:
    net = load_ieee14_case()
    prepare_congestion_case(net, cap_mult=0.87)
    fc = solve_opf_at_scale(net, scale=1.05)
    ac = solve_opf_at_scale(net, scale=1.25)
    assert fc.converged and ac.converged
    df = compare_binding_status(fc, ac)
    expected = {
        "line",
        "forecast_loading_pct",
        "actual_loading_pct",
        "hidden_congestion",
    }
    assert expected.issubset(df.columns)
