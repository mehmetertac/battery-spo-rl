"""Rolling backtest harness: perfect foresight vs point-forecast LP regret."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from src.forecast.price_forecaster import PriceForecaster, build_feature_matrix, forecast_mae


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


def run_regret_backtest(
    n_days: int = 90,
    start: str | pd.Timestamp | None = None,
    output_path: str | Path = "results/regret.csv",
    battery_config: BatteryConfig | None = None,
    warmup_days: int = 7,
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
    cumulative_regret = 0.0

    print("Building feature matrix for forecaster...")
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
                "cumulative_regret": cumulative_regret,
                "forecast_mae": np.nan,
            }
        )

        day_start = pd.Timestamp(day.date(), tz="Europe/Berlin")
        history = prices.loc[prices.index < day_start]
        forecaster = PriceForecaster().fit(
            history,
            precomputed_features=all_features,
            precomputed_targets=all_targets,
            train_end=day_start,
        )
        forecast = forecaster.predict_day(history, day)
        mae = forecast_mae(actual, forecast)

        fc_result = solve_battery_arbitrage(forecast, battery_config)
        fc_revenue = realized_revenue(fc_result.schedule, actual)
        regret = pf_revenue - fc_revenue
        cumulative_regret += regret

        rows.append(
            {
                "date": day.date().isoformat(),
                "approach": "point_forecast",
                "pf_revenue": pf_revenue,
                "approach_revenue": fc_revenue,
                "regret": regret,
                "cumulative_regret": cumulative_regret,
                "forecast_mae": mae,
            }
        )

    results = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    pf_total = float(results.loc[results["approach"] == "perfect_foresight", "pf_revenue"].sum())
    fc_total = float(results.loc[results["approach"] == "point_forecast", "approach_revenue"].sum())
    avg_regret = float(results.loc[results["approach"] == "point_forecast", "regret"].mean())

    print(f"Evaluation days: {len(eval_days)} ({eval_days[0].date()} to {eval_days[-1].date()})")
    print(f"Perfect foresight total revenue: EUR {pf_total:,.2f}")
    print(f"Point forecast total revenue:    EUR {fc_total:,.2f}")
    print(f"Total cumulative regret:         EUR {cumulative_regret:,.2f}")
    print(f"Average daily regret:            EUR {avg_regret:,.2f}")
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
    )


if __name__ == "__main__":
    main()
