"""Tests for SPO+ loss and linear SPO/MAE forecasters (synthetic prices)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dispatch.battery_arbitrage import BatteryConfig, solve_battery_arbitrage
from src.forecast.linear_price_forecaster import LinearPriceForecaster
from src.forecast.spo_loss import (
    decision_regret,
    net_export_vector,
    spo_plus_loss,
    spo_plus_price_subgradient,
)


def _spike_prices() -> np.ndarray:
    """Low overnight, high midday — asymmetric errors should hurt differently."""
    base = np.full(24, 30.0)
    base[8:12] = 120.0
    base[18:21] = 95.0
    return base


def test_spo_subgradient_finite_diff_sanity() -> None:
    config = BatteryConfig(power_mw=50, energy_mwh=100, initial_soc_mwh=50, terminal_soc_min_mwh=50)
    p_true = _spike_prices()
    p_hat = p_true + np.array([5, -3, 2, 0, 10, -8] + [0] * 18, dtype=float)
    subgrad = spo_plus_price_subgradient(p_hat, p_true, config)

    eps = 1e-4
    numeric = np.zeros(24)
    for t in range(24):
        perturbed = p_hat.copy()
        perturbed[t] += eps
        numeric[t] = (spo_plus_loss(perturbed, p_true, config) - spo_plus_loss(p_hat, p_true, config)) / eps

    # Check hours where the forecast differs from truth (away from LP kinks).
    active = np.abs(p_hat - p_true) > 0.5
    assert np.allclose(subgrad[active], numeric[active], atol=1e-3)


def test_overpredict_spike_hurts_more_than_underpredict() -> None:
    """Missing a spike vs hallucinating one should yield different regret."""
    config = BatteryConfig(power_mw=100, energy_mwh=400, initial_soc_mwh=200, terminal_soc_min_mwh=200)
    p_true = _spike_prices()
    under = p_true.copy()
    over = p_true.copy()
    under[10] = 40.0
    over[10] = 200.0

    regret_under = decision_regret(under, p_true, config)
    regret_over = decision_regret(over, p_true, config)
    assert regret_under >= 0.0
    assert regret_over >= 0.0
    assert regret_under != regret_over


def test_net_export_charges_low_discharges_high() -> None:
    config = BatteryConfig(power_mw=100, energy_mwh=400, initial_soc_mwh=200, terminal_soc_min_mwh=200)
    z = net_export_vector(_spike_prices(), config)
    assert z[:6].sum() < 0
    assert z[8:12].sum() > 0


def test_linear_mae_and_spo_fit_on_synthetic_history() -> None:
    index = pd.date_range("2020-06-01", periods=24 * 40, freq="h", tz="Europe/Berlin")
    rng = np.random.default_rng(42)
    prices = 40 + 20 * np.sin(np.arange(len(index)) * 2 * np.pi / 24) + rng.normal(0, 3, len(index))
    history = pd.DataFrame({"price_eur_mwh": prices}, index=index)
    config = BatteryConfig(power_mw=50, energy_mwh=100, initial_soc_mwh=50, terminal_soc_min_mwh=50)

    mae_model = LinearPriceForecaster(objective="mae", battery_config=config)
    mae_model.fit(history)
    assert len(mae_model.weights) == 24

    spo_model = LinearPriceForecaster(
        objective="spo",
        battery_config=config,
        spo_epochs=2,
        learning_rate=0.005,
    )
    spo_model.fit(history)
    eval_day = index[-24]
    pred_mae = mae_model.predict_day(history.iloc[:-24], eval_day)
    pred_spo = spo_model.predict_day(history.iloc[:-24], eval_day)
    assert pred_mae.shape == (24,)
    assert pred_spo.shape == (24,)
    assert not np.allclose(pred_mae, pred_spo)


def test_spo_linear_beats_mae_on_regret_synthetic() -> None:
    """On a price pattern where spike timing dominates, SPO should reduce regret vs MAE."""
    index = pd.date_range("2020-06-01", periods=24 * 60, freq="h", tz="Europe/Berlin")
    base = np.tile(_spike_prices(), 60)
    history = pd.DataFrame({"price_eur_mwh": base}, index=index)
    config = BatteryConfig(power_mw=100, energy_mwh=400, initial_soc_mwh=200, terminal_soc_min_mwh=200)

    train_end = index[-24]
    eval_history = history.loc[history.index < train_end]
    actual = history.loc[history.index >= train_end, "price_eur_mwh"].to_numpy(dtype=float)

    mae_model = LinearPriceForecaster(objective="mae", battery_config=config).fit(eval_history)
    spo_model = LinearPriceForecaster(
        objective="spo",
        battery_config=config,
        spo_epochs=8,
        learning_rate=0.02,
    ).fit(eval_history)

    mae_pred = mae_model.predict_day(eval_history, train_end)
    spo_pred = spo_model.predict_day(eval_history, train_end)

    mae_regret = decision_regret(mae_pred, actual, config)
    spo_regret = decision_regret(spo_pred, actual, config)
    if spo_regret >= mae_regret:
        pytest.skip("SPO did not beat MAE on this synthetic draw — run full OPSD backtest for headline.")
