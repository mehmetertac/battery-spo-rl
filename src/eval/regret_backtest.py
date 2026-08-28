"""Rolling backtest harness: perfect foresight vs MAE vs SPO forecast LP regret."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.opsd_prices import (
    MIN_HISTORY_HOURS,
    available_day_range,
    extract_day_prices,
    load_de_day_ahead,
)
from src.dispatch.battery_arbitrage import (
    BatteryConfig,
    realized_revenue,
    solve_battery_arbitrage,
)
from src.forecast.linear_price_forecaster import LinearPriceForecaster
from src.forecast.price_forecaster import PriceForecaster, build_feature_matrix, forecast_mae
from src.rl.battery_env import evaluate_rl_day


class DayForecaster(Protocol):
    def fit(
        self,
        history: pd.DataFrame,
        precomputed_features: pd.DataFrame | None = None,
        precomputed_targets: pd.Series | None = None,
        train_end: pd.Timestamp | None = None,
    ) -> DayForecaster: ...

    def predict_day(self, history: pd.DataFrame, day: pd.Timestamp) -> np.ndarray: ...


def _complete_days(prices: pd.DataFrame) -> list[pd.Timestamp]:
    daily_counts = prices["price_eur_mwh"].groupby(prices.index.date).count()
    complete = daily_counts[daily_counts == 24].index.tolist()
    return [pd.Timestamp(day, tz="Europe/Berlin") for day in complete]


def _default_eval_days(
    prices: pd.DataFrame,
    n_days: int,
    warmup_days: int,
) -> list[pd.Timestamp]:
    complete = _complete_days(prices)
    min_start = MIN_HISTORY_HOURS // 24 + warmup_days
    if len(complete) <= min_start:
        raise ValueError(
            f"Not enough complete days ({len(complete)}) for warmup ({min_start})."
        )

    eligible = complete[min_start:]
    if len(eligible) < n_days:
        raise ValueError(
            f"Requested {n_days} evaluation days but only {len(eligible)} are available "
            f"after {min_start}-day warmup."
        )
    return eligible[-n_days:]


def _evaluate_forecast_day(
    day: pd.Timestamp,
    actual: np.ndarray,
    pf_revenue: float,
    cumulative_regret: float,
    approach: str,
    forecaster: DayForecaster,
    history: pd.DataFrame,
    all_features: pd.DataFrame,
    all_targets: pd.Series,
    battery_config: BatteryConfig,
) -> tuple[dict[str, object], float]:
    """Fit forecaster, predict, run LP, return result row and updated cumulative regret."""
    forecaster.fit(
        history,
        precomputed_features=all_features,
        precomputed_targets=all_targets,
        train_end=pd.Timestamp(day.date(), tz="Europe/Berlin"),
    )
    forecast = forecaster.predict_day(history, day)
    mae = forecast_mae(actual, forecast)

    fc_result = solve_battery_arbitrage(forecast, battery_config)
    fc_revenue = realized_revenue(fc_result.schedule, actual)
    regret = pf_revenue - fc_revenue
    cumulative_regret += regret

    row = {
        "date": day.date().isoformat(),
        "approach": approach,
        "pf_revenue": pf_revenue,
        "approach_revenue": fc_revenue,
        "regret": regret,
        "cumulative_regret": cumulative_regret,
        "forecast_mae": mae,
    }
    return row, cumulative_regret


def _evaluate_rl_day(
    day: pd.Timestamp,
    actual: np.ndarray,
    pf_revenue: float,
    cumulative_regret: float,
    model: object,
    prices: pd.DataFrame,
    battery_config: BatteryConfig,
) -> tuple[dict[str, object], float]:
    """Roll out trained RL policy and return result row."""
    _schedule, fc_revenue = evaluate_rl_day(
        model,
        prices,
        day,
        battery_config,
        deterministic=True,
    )
    regret = pf_revenue - fc_revenue
    cumulative_regret += regret

    row = {
        "date": day.date().isoformat(),
        "approach": "rl_ppo",
        "pf_revenue": pf_revenue,
        "approach_revenue": fc_revenue,
        "regret": regret,
        "cumulative_regret": cumulative_regret,
        "forecast_mae": np.nan,
    }
    return row, cumulative_regret


def run_regret_backtest(
    n_days: int = 90,
    start: str | pd.Timestamp | None = None,
    output_path: str | Path = "results/regret.csv",
    battery_config: BatteryConfig | None = None,
    warmup_days: int = 7,
    include_lightgbm: bool = False,
    include_rl: bool = False,
    rl_model_path: str | Path = "models/rl_ppo.zip",
    spo_epochs: int = 5,
    spo_learning_rate: float = 0.01,
) -> pd.DataFrame:
    """Run day-by-day regret backtest and write results CSV."""
    if battery_config is None:
        battery_config = BatteryConfig()

    prices = load_de_day_ahead()
    if start is None:
        eval_days = _default_eval_days(prices, n_days, warmup_days)
    else:
        start_ts = pd.Timestamp(start, tz="Europe/Berlin")
        complete = _complete_days(prices)
        eval_days = [day for day in complete if day >= start_ts][:n_days]
        if len(eval_days) < n_days:
            raise ValueError(
                f"Only {len(eval_days)} complete days available from start={start_ts.date()}."
            )

    rows: list[dict[str, object]] = []
    cumulative: dict[str, float] = {
        "mae_linear": 0.0,
        "spo_linear": 0.0,
        "point_forecast": 0.0,
        "rl_ppo": 0.0,
    }

    rl_model = None
    if include_rl:
        from stable_baselines3 import PPO

        rl_path = Path(rl_model_path)
        if not rl_path.exists():
            raise FileNotFoundError(
                f"RL model not found at {rl_path}. Train with: python -m src.rl.train"
            )
        rl_model = PPO.load(str(rl_path))
        print(f"Loaded RL model from {rl_path}")

    print("Building feature matrix for forecasters...")
    all_features, all_targets, _ = build_feature_matrix(prices)

    for day_idx, day in enumerate(eval_days, start=1):
        if day_idx == 1 or day_idx % 10 == 0 or day_idx == len(eval_days):
            print(f"Backtest day {day_idx}/{len(eval_days)}: {day.date()}", flush=True)

        actual = extract_day_prices(prices, day).to_numpy(dtype=float)
        pf_result = solve_battery_arbitrage(actual, battery_config)
        pf_revenue = pf_result.revenue

        rows.append(
            {
                "date": day.date().isoformat(),
                "approach": "perfect_foresight",
                "pf_revenue": pf_revenue,
                "approach_revenue": pf_revenue,
                "regret": 0.0,
                "cumulative_regret": np.nan,
                "forecast_mae": np.nan,
            }
        )

        day_start = pd.Timestamp(day.date(), tz="Europe/Berlin")
        history = prices.loc[prices.index < day_start]

        mae_forecaster = LinearPriceForecaster(objective="mae", battery_config=battery_config)
        row, cumulative["mae_linear"] = _evaluate_forecast_day(
            day,
            actual,
            pf_revenue,
            cumulative["mae_linear"],
            "mae_linear",
            mae_forecaster,
            history,
            all_features,
            all_targets,
            battery_config,
        )
        rows.append(row)

        if include_rl and rl_model is not None:
            row, cumulative["rl_ppo"] = _evaluate_rl_day(
                day,
                actual,
                pf_revenue,
                cumulative["rl_ppo"],
                rl_model,
                prices,
                battery_config,
            )
            rows.append(row)

        spo_forecaster = LinearPriceForecaster(
            objective="spo",
            battery_config=battery_config,
            spo_epochs=spo_epochs,
            learning_rate=spo_learning_rate,
        )
        row, cumulative["spo_linear"] = _evaluate_forecast_day(
            day,
            actual,
            pf_revenue,
            cumulative["spo_linear"],
            "spo_linear",
            spo_forecaster,
            history,
            all_features,
            all_targets,
            battery_config,
        )
        rows.append(row)

        if include_lightgbm:
            lgbm = PriceForecaster()
            row, cumulative["point_forecast"] = _evaluate_forecast_day(
                day,
                actual,
                pf_revenue,
                cumulative["point_forecast"],
                "point_forecast",
                lgbm,
                history,
                all_features,
                all_targets,
                battery_config,
            )
            rows.append(row)

    results = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    pf_total = float(results.loc[results["approach"] == "perfect_foresight", "pf_revenue"].sum())
    print(f"Evaluation days: {len(eval_days)} ({eval_days[0].date()} to {eval_days[-1].date()})")
    print(f"Perfect foresight total revenue: EUR {pf_total:,.2f}")

    for approach, label in [
        ("mae_linear", "MAE linear"),
        ("rl_ppo", "RL PPO"),
        ("spo_linear", "SPO linear"),
        ("point_forecast", "LightGBM point forecast"),
    ]:
        subset = results.loc[results["approach"] == approach]
        if subset.empty:
            continue
        total_rev = float(subset["approach_revenue"].sum())
        total_regret = float(subset["regret"].sum())
        mae_values = subset["forecast_mae"].dropna()
        mae_str = f"{float(mae_values.mean()):.2f} EUR/MWh" if not mae_values.empty else "n/a"
        print(
            f"{label:28s} revenue EUR {total_rev:>12,.2f}  "
            f"cumulative regret EUR {total_regret:>12,.2f}  avg MAE {mae_str}"
        )

    print(f"Saved results to {output_path}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run battery arbitrage regret backtest.")
    parser.add_argument("--days", type=int, default=90, help="Number of evaluation days.")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="First evaluation day (YYYY-MM-DD). Default: last N days in OPSD.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/regret.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--power-mw", type=float, default=100.0, help="Battery power rating.")
    parser.add_argument("--energy-mwh", type=float, default=400.0, help="Battery energy capacity.")
    parser.add_argument(
        "--efficiency",
        type=float,
        default=0.90,
        help="Round-trip efficiency.",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=7,
        help="Extra warmup days beyond 168h lag requirement.",
    )
    parser.add_argument(
        "--include-lightgbm",
        action="store_true",
        help="Also run legacy LightGBM point-forecast baseline.",
    )
    parser.add_argument(
        "--include-rl",
        action="store_true",
        help="Include trained PPO agent (requires --rl-model).",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default="models/rl_ppo.zip",
        help="Path to trained SB3 PPO model.",
    )
    parser.add_argument(
        "--spo-epochs",
        type=int,
        default=5,
        help="SPO+ subgradient epochs per training window.",
    )
    parser.add_argument(
        "--spo-lr",
        type=float,
        default=5.0,
        help="SPO+ subgradient learning rate (bias-only fine-tuning from MAE init).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    first_available, last_available = available_day_range()
    print(
        "OPSD DE-LU day-ahead prices available "
        f"from {first_available.date()} to {last_available.date()}."
    )

    initial_soc = args.energy_mwh / 2.0
    battery = BatteryConfig(
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        round_trip_efficiency=args.efficiency,
        initial_soc_mwh=initial_soc,
        terminal_soc_min_mwh=initial_soc,
    )
    run_regret_backtest(
        n_days=args.days,
        start=args.start,
        output_path=args.output,
        battery_config=battery,
        warmup_days=args.warmup_days,
        include_lightgbm=args.include_lightgbm,
        include_rl=args.include_rl,
        rl_model_path=args.rl_model,
        spo_epochs=args.spo_epochs,
        spo_learning_rate=args.spo_lr,
    )


if __name__ == "__main__":
    main()
