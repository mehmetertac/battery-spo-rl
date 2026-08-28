"""Gymnasium environment for day-ahead battery arbitrage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.opsd_prices import extract_day_prices
from src.dispatch.battery_arbitrage import BatteryConfig, realized_revenue
from src.forecast.price_forecaster import LAG_HOURS, build_feature_row

HOLD = 0
CHARGE = 1
DISCHARGE = 2
N_HOURS = 24
TERMINAL_SOC_PENALTY = 50.0  # EUR per MWh shortfall vs terminal minimum


@dataclass
class EpisodeStep:
    hour: int
    charge_mw: float
    discharge_mw: float
    soc_mwh: float


def schedule_from_episode(log: list[EpisodeStep]) -> pd.DataFrame:
    """Convert episode log to a schedule DataFrame for realized_revenue()."""
    return pd.DataFrame(
        {
            "hour": [step.hour for step in log],
            "charge_mw": [step.charge_mw for step in log],
            "discharge_mw": [step.discharge_mw for step in log],
            "soc_mwh": [step.soc_mwh for step in log],
        }
    )


class BatteryArbitrageEnv(gym.Env):
    """One-day battery arbitrage with causal price observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: BatteryConfig,
        prices: pd.DataFrame,
        training_days: list[pd.Timestamp] | None = None,
        eval_day: pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if eval_day is None and not training_days:
            raise ValueError("Provide eval_day or a non-empty training_days list.")

        self.config = config
        self.prices = prices
        self.training_days = list(training_days or [])
        if eval_day is not None:
            ts = pd.Timestamp(eval_day)
            self.eval_day = ts.tz_convert("Europe/Berlin") if ts.tzinfo else ts.tz_localize("Europe/Berlin")
        else:
            self.eval_day = None

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + len(LAG_HOURS) + 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)
        self._day: pd.Timestamp | None = None
        self._day_start: pd.Timestamp | None = None
        self._actual: np.ndarray | None = None
        self._hour = 0
        self._soc = config.initial_soc_mwh
        self._episode_log: list[EpisodeStep] = []

    def _select_day(self) -> pd.Timestamp:
        if self.eval_day is not None:
            return self.eval_day
        return self._rng.choice(self.training_days)

    def _build_obs(self) -> np.ndarray:
        assert self._day_start is not None
        target_ts = self._day_start + pd.Timedelta(hours=self._hour)
        history = self.prices.loc[self.prices.index < target_ts]
        row = build_feature_row(history, target_ts)
        lag_values = [row[f"lag_{lag}"] for lag in LAG_HOURS]
        obs = np.array(
            [
                self._soc / self.config.energy_mwh,
                self._hour / 23.0,
                *lag_values,
                row["day_of_week"] / 6.0,
                row["month"] / 12.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _clip_action(self, action: int) -> tuple[float, float]:
        charge_mw = 0.0
        discharge_mw = 0.0
        if action == CHARGE:
            charge_mw = self.config.power_mw
        elif action == DISCHARGE:
            discharge_mw = self.config.power_mw

        eta_c = self.config.eta_charge
        eta_d = self.config.eta_discharge

        if charge_mw > 0:
            max_charge_energy = self.config.energy_mwh - self._soc
            max_charge_mw = max(0.0, max_charge_energy / eta_c)
            charge_mw = min(charge_mw, max_charge_mw, self.config.power_mw)

        if discharge_mw > 0:
            available = self._soc - self.config.soc_min_mwh
            max_discharge_mw = max(0.0, available * eta_d)
            discharge_mw = min(discharge_mw, max_discharge_mw, self.config.power_mw)

        return charge_mw, discharge_mw

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._day = self._select_day()
        self._day_start = pd.Timestamp(self._day.date(), tz="Europe/Berlin")
        self._actual = extract_day_prices(self.prices, self._day).to_numpy(dtype=float)
        self._hour = 0
        self._soc = self.config.initial_soc_mwh
        self._episode_log = []
        return self._build_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._actual is not None

        charge_mw, discharge_mw = self._clip_action(int(action))
        price = float(self._actual[self._hour])
        reward = price * (discharge_mw - charge_mw)

        self._episode_log.append(
            EpisodeStep(
                hour=self._hour,
                charge_mw=charge_mw,
                discharge_mw=discharge_mw,
                soc_mwh=self._soc,
            )
        )

        eta_c = self.config.eta_charge
        eta_d = self.config.eta_discharge
        self._soc = self._soc + eta_c * charge_mw - discharge_mw / eta_d
        self._soc = float(np.clip(self._soc, self.config.soc_min_mwh, self.config.energy_mwh))

        self._hour += 1
        terminated = self._hour >= N_HOURS
        truncated = False

        if terminated:
            shortfall = max(0.0, self.config.terminal_soc_min_mwh - self._soc)
            reward -= TERMINAL_SOC_PENALTY * shortfall

        obs = self._build_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, {}

    @property
    def episode_log(self) -> list[EpisodeStep]:
        return list(self._episode_log)


def evaluate_rl_day(
    model: Any,
    prices: pd.DataFrame,
    day: pd.Timestamp,
    config: BatteryConfig,
    *,
    deterministic: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Roll out a trained SB3 policy for one day and return schedule + revenue."""
    env = BatteryArbitrageEnv(config=config, prices=prices, eval_day=day)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        done = terminated or truncated

    schedule = schedule_from_episode(env.episode_log)
    actual = extract_day_prices(prices, day).to_numpy(dtype=float)
    revenue = realized_revenue(schedule, actual)
    return schedule, revenue
