"""Synthetic tests for the battery arbitrage Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dispatch.battery_arbitrage import BatteryConfig, realized_revenue
from src.rl.battery_env import (
    CHARGE,
    DISCHARGE,
    HOLD,
    BatteryArbitrageEnv,
    schedule_from_episode,
    EpisodeStep,
)


def _synthetic_prices(n_days: int = 15) -> pd.DataFrame:
    index = pd.date_range("2020-06-01", periods=24 * n_days, freq="h", tz="Europe/Berlin")
    values = np.tile(np.linspace(20, 80, 24), n_days)
    return pd.DataFrame({"price_eur_mwh": values}, index=index)


def _env_for_day(prices: pd.DataFrame, day_idx: int = 10) -> BatteryArbitrageEnv:
    complete = sorted({pd.Timestamp(ts.date(), tz="Europe/Berlin") for ts in prices.index})
    day = complete[day_idx]
    config = BatteryConfig(power_mw=50, energy_mwh=100, initial_soc_mwh=50, terminal_soc_min_mwh=50)
    return BatteryArbitrageEnv(
        config=config,
        prices=prices,
        eval_day=day,
        seed=0,
    )


def test_reset_and_step_shapes() -> None:
    env = _env_for_day(_synthetic_prices())
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.action_space.n == 3

    obs, reward, terminated, truncated, _ = env.step(HOLD)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False


def test_soc_stays_feasible_under_random_policy() -> None:
    env = _env_for_day(_synthetic_prices())
    rng = np.random.default_rng(0)
    env.reset()
    done = False
    while not done:
        action = int(rng.integers(0, 3))
        _obs, _reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    final_soc = env._soc
    assert env.config.soc_min_mwh - 1e-6 <= final_soc <= env.config.energy_mwh + 1e-6


def test_spike_day_charge_discharge_beats_hold_only() -> None:
    warmup_index = pd.date_range(
        "2020-06-24", periods=24 * 7, freq="h", tz="Europe/Berlin"
    )
    warmup = pd.DataFrame({"price_eur_mwh": np.full(len(warmup_index), 50.0)}, index=warmup_index)
    spike_index = pd.date_range("2020-07-01", periods=24, freq="h", tz="Europe/Berlin")
    spike_prices = np.array([10, 10, 10, 10, 100, 100, 100, 100] + [50] * 16, dtype=float)
    spike = pd.DataFrame({"price_eur_mwh": spike_prices}, index=spike_index)
    frame = pd.concat([warmup, spike])
    day = pd.Timestamp("2020-07-01", tz="Europe/Berlin")
    config = BatteryConfig(power_mw=100, energy_mwh=400, initial_soc_mwh=200, terminal_soc_min_mwh=200)

    hold_env = BatteryArbitrageEnv(config=config, prices=frame, eval_day=day)
    hold_env.reset()
    hold_reward = 0.0
    done = False
    while not done:
        _obs, reward, terminated, truncated, _ = hold_env.step(HOLD)
        hold_reward += reward
        done = terminated or truncated

    smart_env = BatteryArbitrageEnv(config=config, prices=frame, eval_day=day)
    smart_env.reset()
    smart_reward = 0.0
    for hour in range(24):
        if hour < 4:
            action = CHARGE
        elif hour < 8:
            action = DISCHARGE
        else:
            action = HOLD
        _obs, reward, terminated, truncated, _ = smart_env.step(action)
        smart_reward += reward
        if terminated or truncated:
            break

    assert smart_reward > hold_reward


def test_schedule_from_episode_matches_realized_revenue() -> None:
    log = [
        EpisodeStep(hour=0, charge_mw=10.0, discharge_mw=0.0, soc_mwh=50.0),
        EpisodeStep(hour=1, charge_mw=0.0, discharge_mw=20.0, soc_mwh=59.0),
    ]
    schedule = schedule_from_episode(log)
    actual = np.array([30.0, 80.0])
    revenue = realized_revenue(schedule, actual)
    assert revenue == pytest.approx(30.0 * (-10.0) + 80.0 * 20.0)


def test_ppo_smoke_rollout() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    prices = _synthetic_prices(n_days=15)
    env = _env_for_day(prices, day_idx=10)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
