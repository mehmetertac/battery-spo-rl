"""LightGBM day-ahead price forecaster with lag and calendar features."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

LAG_HOURS = (1, 2, 3, 24, 48, 168)
FEATURE_COLUMNS = [f"lag_{lag}" for lag in LAG_HOURS] + [
    "hour_of_day",
    "day_of_week",
    "month",
]
MIN_TRAINING_ROWS = 24 * 7
MAX_TRAINING_ROWS = 24 * 90


@dataclass
class PriceForecaster:
    lags: tuple[int, ...] = LAG_HOURS
    models: dict[int, lgb.LGBMRegressor] = field(default_factory=dict)

    def fit(
        self,
        history: pd.DataFrame,
        precomputed_features: pd.DataFrame | None = None,
        precomputed_targets: pd.Series | None = None,
        train_end: pd.Timestamp | None = None,
    ) -> PriceForecaster:
        """Fit one LightGBM model per hour-of-day on historical prices."""
        if precomputed_features is not None and precomputed_targets is not None:
            features = precomputed_features
            targets = precomputed_targets
            if train_end is not None:
                mask = features.index < train_end
                features = features.loc[mask]
                targets = targets.loc[mask]
        else:
            features, targets, _ = build_feature_matrix(history)

        if len(features) > MAX_TRAINING_ROWS:
            features = features.iloc[-MAX_TRAINING_ROWS:]
            targets = targets.iloc[-MAX_TRAINING_ROWS:]
        if len(features) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"Need at least {MIN_TRAINING_ROWS} training rows, got {len(features)}."
            )

        self.models = {}
        for hour in range(24):
            mask = features["hour_of_day"] == hour
            x_hour = features.loc[mask, FEATURE_COLUMNS]
            y_hour = targets.loc[mask]
            if len(x_hour) < 24:
                raise ValueError(f"Not enough training samples for hour {hour}.")

            model = lgb.LGBMRegressor(
                n_estimators=50,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            model.fit(x_hour, y_hour)
            self.models[hour] = model

        return self

    def predict_day(self, history: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        """Predict 24 hourly prices for a calendar day using history before day start."""
        if not self.models:
            raise RuntimeError("Forecaster is not fitted.")

        day_start = pd.Timestamp(day.date(), tz="Europe/Berlin")
        forecasts: list[float] = []
        for hour in range(24):
            target_ts = day_start + pd.Timedelta(hours=hour)
            row = build_feature_row(history, target_ts)
            x_row = pd.DataFrame([row])[FEATURE_COLUMNS]
            forecasts.append(float(self.models[hour].predict(x_row)[0]))

        return np.asarray(forecasts, dtype=float)


def _as_price_series(prices: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if "price_eur_mwh" not in prices.columns:
            raise ValueError("prices DataFrame must contain price_eur_mwh column.")
        series = prices["price_eur_mwh"]
    else:
        series = prices

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("price series must have a DatetimeIndex.")

    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("Europe/Berlin")
    else:
        series = series.copy()
        series.index = series.index.tz_convert("Europe/Berlin")

    return series.sort_index()


def _price_at(series: pd.Series, timestamp: pd.Timestamp) -> float:
    """Return the price at or immediately before timestamp (handles DST gaps)."""
    timestamp = pd.Timestamp(timestamp).tz_convert("Europe/Berlin")
    position = series.index.get_indexer([timestamp], method="pad")
    if position[0] == -1:
        raise ValueError(f"No price available at or before {timestamp}.")
    return float(series.iloc[position[0]])


def build_feature_row(history: pd.DataFrame | pd.Series, target_ts: pd.Timestamp) -> dict[str, float]:
    """Build feature vector for a single target timestamp."""
    series = _as_price_series(history)
    target_ts = pd.Timestamp(target_ts).tz_convert("Europe/Berlin")

    row: dict[str, float] = {
        "hour_of_day": float(target_ts.hour),
        "day_of_week": float(target_ts.dayofweek),
        "month": float(target_ts.month),
    }
    for lag in LAG_HOURS:
        lag_ts = target_ts - pd.Timedelta(hours=lag)
        row[f"lag_{lag}"] = _price_at(series, lag_ts)

    return row


def build_feature_matrix(
    prices: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Build supervised learning matrix from hourly prices."""
    series = _as_price_series(prices)
    max_lag = max(LAG_HOURS)
    if len(series) <= max_lag:
        raise ValueError(f"Need more than {max_lag} hourly rows to build features.")

    feature_rows: list[dict[str, float]] = []
    target_values: list[float] = []
    target_index: list[pd.Timestamp] = []

    start_idx = max_lag
    history_frame = series.to_frame(name="price_eur_mwh")
    for idx in range(start_idx, len(series)):
        target_ts = series.index[idx]
        row = build_feature_row(history_frame, target_ts)
        feature_rows.append(row)
        target_values.append(float(series.iloc[idx]))
        target_index.append(target_ts)

    features = pd.DataFrame(feature_rows, index=pd.DatetimeIndex(target_index))
    targets = pd.Series(target_values, index=pd.DatetimeIndex(target_index), name="price_eur_mwh")
    return features, targets, pd.DatetimeIndex(target_index)


def forecast_mae(actual: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> float:
    """Mean absolute forecast error in EUR/MWh."""
    actual_arr = np.asarray(actual, dtype=float).reshape(-1)
    forecast_arr = np.asarray(forecast, dtype=float).reshape(-1)
    return float(np.mean(np.abs(actual_arr - forecast_arr)))
