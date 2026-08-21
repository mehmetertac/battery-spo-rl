"""Synthetic-price sanity checks for battery arbitrage LP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dispatch.battery_arbitrage import BatteryConfig, realized_revenue, solve_battery_arbitrage


def test_perfect_foresight_self_check() -> None:
    prices = np.array([30, 20, 80, 90, 40, 25] * 4, dtype=float)
    config = BatteryConfig(power_mw=50, energy_mwh=100, initial_soc_mwh=50, terminal_soc_min_mwh=50)
    result = solve_battery_arbitrage(prices, config)
    replay = realized_revenue(result.schedule, prices)
    assert abs(result.revenue - replay) < 1e-4


def test_soc_feasibility() -> None:
    prices = np.linspace(20, 100, 24)
    config = BatteryConfig()
    result = solve_battery_arbitrage(prices, config)
    soc = result.schedule["soc_mwh"]
    assert soc.min() >= config.soc_min_mwh - 1e-4
    assert soc.max() <= config.energy_mwh + 1e-4
    assert result.schedule["charge_mw"].max() <= config.power_mw + 1e-4
    assert result.schedule["discharge_mw"].max() <= config.power_mw + 1e-4


def test_toy_price_spike() -> None:
    prices = np.array([10, 10, 10, 10, 100, 100, 100, 100] + [50] * 16, dtype=float)
    config = BatteryConfig(power_mw=100, energy_mwh=400, initial_soc_mwh=200, terminal_soc_min_mwh=200)
    result = solve_battery_arbitrage(prices, config)
    low_charge = result.schedule.loc[result.schedule["hour"] < 4, "charge_mw"].sum()
    high_discharge = result.schedule.loc[result.schedule["hour"].between(4, 7), "discharge_mw"].sum()
    assert low_charge > 1.0
    assert high_discharge > 1.0


def test_forecast_regret_non_negative_on_actual_prices() -> None:
    prices = np.array([20, 25, 30, 80, 75, 60, 40, 35] * 3, dtype=float)
    config = BatteryConfig(power_mw=50, energy_mwh=100, initial_soc_mwh=50, terminal_soc_min_mwh=50)
    pf = solve_battery_arbitrage(prices, config)
    fc = solve_battery_arbitrage(prices * 1.1, config)
    fc_revenue = realized_revenue(fc.schedule, prices)
    assert pf.revenue >= fc_revenue - 1e-4


def test_build_feature_row_uses_only_past_data() -> None:
    from src.forecast.price_forecaster import build_feature_row

    index = pd.date_range("2020-06-01", periods=200, freq="h", tz="Europe/Berlin")
    history = pd.DataFrame({"price_eur_mwh": np.arange(200, dtype=float)}, index=index)
    target = index[-24]
    row = build_feature_row(history.iloc[:-24], target)
    assert row["lag_24"] == float(history.loc[target - pd.Timedelta(hours=24), "price_eur_mwh"])
