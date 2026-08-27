"""Synthetic system load profile and linear day-ahead load forecaster."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.dispatch.economic_dispatch import default_demand
from src.forecast.price_forecaster import FEATURE_COLUMNS, LAG_HOURS, build_feature_matrix

MIN_TRAINING_ROWS = 24 * 7
MAX_TRAINING_ROWS = 24 * 90


def build_system_load_profile(
    days: int = 30,
    start: str = "2020-07-01",
    noise_std: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a multi-day system load scale profile normalized around 1.0.

    Uses the Day-1 ``default_demand()`` sinusoid as the diurnal shape, repeated
    over ``days`` with small Gaussian noise — a stand-in for Week 1/4 load data.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="Europe/Berlin")
    hourly_shape = default_demand(24)
    shape = hourly_shape / hourly_shape.mean()

    values: list[float] = []
    for ts in index:
        base = float(shape[ts.hour])
        noise = float(rng.normal(0.0, noise_std))
        values.append(max(0.5, base + noise))

    return pd.DataFrame({"load_scale": values}, index=index)


def _as_load_series(loads: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(loads, pd.DataFrame):
        if "load_scale" not in loads.columns:
            raise ValueError("loads DataFrame must contain load_scale column.")
        series = loads["load_scale"]
    else:
        series = loads

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("load series must have a DatetimeIndex.")

    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("Europe/Berlin")
    else:
        series = series.copy()
        series.index = series.index.tz_convert("Europe/Berlin")

    return series.sort_index()


def build_load_feature_matrix(
    loads: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Build supervised features for load-scale forecasting."""
    series = _as_load_series(loads)
    price_frame = series.to_frame(name="price_eur_mwh")  # reuse price feature builder
    return build_feature_matrix(price_frame)


@dataclass
class LinearLoadForecaster:
    """OLS per hour-of-day on lag/calendar features — minimizes load-scale MAE."""

    weights: dict[int, np.ndarray] = field(default_factory=dict)
    bias: dict[int, float] = field(default_factory=dict)
    scale_clip: tuple[float, float] | None = field(default=None, repr=False)

    def fit(
        self,
        history: pd.DataFrame,
        train_end: pd.Timestamp | None = None,
    ) -> LinearLoadForecaster:
        features, targets, _ = build_load_feature_matrix(history)

        if train_end is not None:
            mask = features.index < train_end
            features = features.loc[mask]
            targets = targets.loc[mask]

        if len(features) > MAX_TRAINING_ROWS:
            features = features.iloc[-MAX_TRAINING_ROWS:]
            targets = targets.iloc[-MAX_TRAINING_ROWS:]
        if len(features) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"Need at least {MIN_TRAINING_ROWS} training rows, got {len(features)}."
            )

        y = targets.to_numpy(dtype=float)
        margin = max(0.1, 0.25 * float(np.std(y)))
        self.scale_clip = (float(np.min(y) - margin), float(np.max(y) + margin))

        self.weights = {}
        self.bias = {}
        x_all = features[FEATURE_COLUMNS].to_numpy(dtype=float)

        for hour in range(24):
            mask = features["hour_of_day"] == hour
            x_hour = x_all[mask.to_numpy()]
            y_hour = y[mask.to_numpy()]
            if len(x_hour) < 3:
                raise ValueError(f"Not enough training samples for hour {hour}.")

            design = np.column_stack([np.ones(len(x_hour)), x_hour])
            coef, _, _, _ = np.linalg.lstsq(design, y_hour, rcond=None)
            self.bias[hour] = float(coef[0])
            self.weights[hour] = coef[1:]

        return self

    def predict_day(self, history: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        """Predict 24 hourly load-scale factors for a calendar day."""
        if not self.weights:
            raise RuntimeError("Forecaster is not fitted.")

        from src.forecast.price_forecaster import build_feature_row

        day_start = pd.Timestamp(day.date(), tz="Europe/Berlin")
        history_frame = _as_load_series(history).to_frame(name="price_eur_mwh")
        forecasts: list[float] = []

        for hour in range(24):
            target_ts = day_start + pd.Timedelta(hours=hour)
            row = build_feature_row(history_frame, target_ts)
            x = np.array([row[c] for c in FEATURE_COLUMNS], dtype=float)
            pred = float(self.bias[hour] + self.weights[hour] @ x)
            if self.scale_clip is not None:
                pred = float(np.clip(pred, self.scale_clip[0], self.scale_clip[1]))
            forecasts.append(pred)

        return np.asarray(forecasts, dtype=float)


def forecast_vs_actual_scales(
    history: pd.DataFrame,
    eval_day: pd.Timestamp,
    train_end: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (actual_scales, forecast_scales) for a 24-hour evaluation day."""
    series = _as_load_series(history)
    day_start = pd.Timestamp(eval_day.date(), tz="Europe/Berlin")
    day_end = day_start + pd.Timedelta(hours=23)
    actual = series.loc[day_start:day_end].to_numpy(dtype=float)
    if len(actual) != 24:
        raise ValueError(f"Expected 24 actual hours for {eval_day.date()}, got {len(actual)}.")

    forecaster = LinearLoadForecaster()
    forecaster.fit(history, train_end=train_end or day_start)
    forecast = forecaster.predict_day(history, eval_day)
    return actual, forecast


def load_forecast_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error of load-scale forecasts."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def stress_actual_scales(
    actual: np.ndarray,
    forecast: np.ndarray,
    underforecast_gain: float = 3.0,
    scale_min: float = 0.9,
    scale_max: float = 1.55,
) -> np.ndarray:
    """Amplify peak under-forecast for OPF stress — models realized load above forecast."""
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    positive_error = np.maximum(0.0, actual_arr - forecast_arr)
    stressed = actual_arr + underforecast_gain * positive_error
    return np.clip(stressed, scale_min, scale_max)
