"""Linear day-ahead price forecaster — MAE or SPO+ training on identical features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.dispatch.battery_arbitrage import BatteryConfig
from src.forecast.price_forecaster import (
    FEATURE_COLUMNS,
    MIN_TRAINING_ROWS,
    MAX_TRAINING_ROWS,
    build_feature_matrix,
    build_feature_row,
)
from src.forecast.spo_loss import spo_plus_price_subgradient


@dataclass
class LinearPriceForecaster:
    """One linear model per hour-of-day; train with MAE (OLS) or SPO+ subgradient."""

    objective: str = "mae"
    learning_rate: float = 5.0
    spo_epochs: int = 5
    max_spo_training_days: int = 20
    spo_tune_bias_only: bool = True
    battery_config: BatteryConfig | None = None
    weights: dict[int, np.ndarray] = field(default_factory=dict)
    bias: dict[int, float] = field(default_factory=dict)
    price_clip: tuple[float, float] | None = field(default=None, repr=False)

    def fit(
        self,
        history: pd.DataFrame,
        precomputed_features: pd.DataFrame | None = None,
        precomputed_targets: pd.Series | None = None,
        train_end: pd.Timestamp | None = None,
    ) -> LinearPriceForecaster:
        if self.objective not in {"mae", "spo"}:
            raise ValueError("objective must be 'mae' or 'spo'")

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

        y = targets.to_numpy(dtype=float)
        margin = max(20.0, 0.25 * float(np.std(y)))
        self.price_clip = (float(np.min(y) - margin), float(np.max(y) + margin))

        if self.objective == "mae":
            self._fit_mae(features, targets)
        else:
            self._fit_spo(features, targets)
        return self

    def _fit_mae(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Ordinary least squares per hour — minimizes squared error (MAE proxy)."""
        self.weights = {}
        self.bias = {}
        x_all = features[FEATURE_COLUMNS].to_numpy(dtype=float)

        for hour in range(24):
            mask = features["hour_of_day"] == hour
            x_hour = x_all[mask]
            y_hour = targets.loc[mask].to_numpy(dtype=float)
            if len(x_hour) < 24:
                raise ValueError(f"Not enough training samples for hour {hour}.")

            design = np.column_stack([x_hour, np.ones(len(x_hour))])
            coef, _, _, _ = np.linalg.lstsq(design, y_hour, rcond=None)
            self.weights[hour] = coef[:-1]
            self.bias[hour] = float(coef[-1])

    def _fit_spo(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """SPO+ subgradient fine-tuning from MAE init (Elmachtoub & Grigas)."""
        self._fit_mae(features, targets)

        day_groups = self._group_complete_days(features, targets)
        if len(day_groups) < 7:
            raise ValueError(f"Need at least 7 complete training days for SPO+, got {len(day_groups)}.")
        if len(day_groups) > self.max_spo_training_days:
            day_groups = day_groups[-self.max_spo_training_days :]

        config = self.battery_config or BatteryConfig()
        n_days = len(day_groups)

        for _ in range(self.spo_epochs):
            for day_features, day_targets in day_groups:
                p_hat = self._predict_from_matrix(day_features)
                p_true = day_targets.to_numpy(dtype=float)
                subgrad = spo_plus_price_subgradient(p_hat, p_true, config)
                subgrad = np.clip(subgrad, -config.power_mw, config.power_mw)
                subgrad_norm = float(np.linalg.norm(subgrad))
                if subgrad_norm > 1e-8:
                    subgrad = subgrad / subgrad_norm

                step = self.learning_rate / n_days
                for hour in range(24):
                    if self.spo_tune_bias_only:
                        self.bias[hour] -= step * subgrad[hour]
                    else:
                        row = day_features.loc[day_features["hour_of_day"] == hour].iloc[0]
                        x_h = row[FEATURE_COLUMNS].to_numpy(dtype=float)
                        self.weights[hour] -= step * subgrad[hour] * x_h
                        self.bias[hour] -= step * subgrad[hour]

    @staticmethod
    def _group_complete_days(
        features: pd.DataFrame,
        targets: pd.Series,
    ) -> list[tuple[pd.DataFrame, pd.Series]]:
        """Group feature rows into 24-hour days (skip DST-incomplete days)."""
        groups: list[tuple[pd.DataFrame, pd.Series]] = []
        for _, day_feats in features.groupby(features.index.date):
            if len(day_feats) != 24:
                continue
            hours = sorted(day_feats["hour_of_day"].unique())
            if hours != list(range(24)):
                continue
            day_feats = day_feats.sort_values("hour_of_day")
            day_targets = targets.loc[day_feats.index].sort_index()
            groups.append((day_feats, day_targets))
        return groups

    def _predict_from_matrix(self, day_features: pd.DataFrame) -> np.ndarray:
        forecasts: list[float] = []
        for hour in range(24):
            row = day_features.loc[day_features["hour_of_day"] == hour].iloc[0]
            x = row[FEATURE_COLUMNS].to_numpy(dtype=float)
            forecasts.append(float(np.dot(self.weights[hour], x) + self.bias[hour]))
        return self._clip_prices(np.asarray(forecasts, dtype=float))

    def _clip_prices(self, prices: np.ndarray) -> np.ndarray:
        if self.price_clip is None:
            return prices
        lo, hi = self.price_clip
        return np.clip(prices, lo, hi)

    def predict_day(self, history: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("Forecaster is not fitted.")

        day_start = pd.Timestamp(day.date(), tz="Europe/Berlin")
        forecasts: list[float] = []
        for hour in range(24):
            target_ts = day_start + pd.Timedelta(hours=hour)
            row = build_feature_row(history, target_ts)
            x = np.array([row[col] for col in FEATURE_COLUMNS], dtype=float)
            forecasts.append(float(np.dot(self.weights[hour], x) + self.bias[hour]))

        return self._clip_prices(np.asarray(forecasts, dtype=float))
