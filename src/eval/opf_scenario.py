"""Forecast vs actual load OPF scenarios on IEEE 14-bus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dispatch.ieee14_opf import (
    BINDING_LOADING_THRESHOLD,
    compare_binding_status,
    load_ieee14_case,
    prepare_congestion_case,
    solve_opf_at_scale,
)
from src.eval.opf_figures import generate_opf_figures
from src.forecast.load_forecaster import (
    build_system_load_profile,
    forecast_vs_actual_scales,
    load_forecast_mae,
    stress_actual_scales,
)


def run_opf_scenarios(
    n_hours: int = 24,
    eval_day_offset: int = -1,
    profile_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Run OPF under forecast and actual load scales for each hour of an eval day."""
    profile = build_system_load_profile(days=profile_days, seed=seed)
    eval_day = profile.index.normalize().unique()[eval_day_offset]
    train_end = pd.Timestamp(eval_day.date(), tz="Europe/Berlin")
    history = profile.loc[profile.index < train_end + pd.Timedelta(hours=24)]
    actual_scales, forecast_scales = forecast_vs_actual_scales(profile, eval_day, train_end=train_end)
    actual_opf_scales = stress_actual_scales(actual_scales, forecast_scales)

    day_start = pd.Timestamp(eval_day.date(), tz="Europe/Berlin")
    hours = min(n_hours, 24)
    bus_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    net = load_ieee14_case()
    prepare_congestion_case(net)

    for hour in range(hours):
        ts = day_start + pd.Timedelta(hours=hour)
        for approach, scale in (
            ("forecast_load", float(forecast_scales[hour])),
            ("actual_load", float(actual_opf_scales[hour])),
        ):
            result = solve_opf_at_scale(net, scale=scale)
            if not result.converged:
                continue

            for bus, lmp in result.bus_lmps.items():
                bus_rows.append(
                    {
                        "timestamp": ts,
                        "hour": hour,
                        "approach": approach,
                        "load_scale": scale,
                        "bus": int(bus),
                        "lmp_usd_per_mwh": float(lmp),
                        "total_cost_usd": result.total_cost,
                    }
                )

            for line_idx, row in result.line_loadings.iterrows():
                line_rows.append(
                    {
                        "timestamp": ts,
                        "hour": hour,
                        "approach": approach,
                        "load_scale": scale,
                        "line": int(line_idx),
                        "from_bus": int(row["from_bus"]),
                        "to_bus": int(row["to_bus"]),
                        "loading_percent": float(row["loading_percent"]),
                        "total_cost_usd": result.total_cost,
                    }
                )

        fc = solve_opf_at_scale(net, scale=float(forecast_scales[hour]))
        ac = solve_opf_at_scale(net, scale=float(actual_opf_scales[hour]))
        if not fc.converged or not ac.converged:
            continue
        binding = compare_binding_status(fc, ac, threshold=BINDING_LOADING_THRESHOLD)
        hidden = binding[binding["hidden_congestion"]]
        summary_rows.append(
            {
                "timestamp": ts,
                "hour": hour,
                "forecast_scale": float(forecast_scales[hour]),
                "actual_scale": float(actual_scales[hour]),
                "actual_opf_scale": float(actual_opf_scales[hour]),
                "scale_error": float(actual_scales[hour] - forecast_scales[hour]),
                "forecast_mae": load_forecast_mae(
                    actual_scales[: hour + 1], forecast_scales[: hour + 1]
                ),
                "forecast_max_loading_pct": float(fc.line_loadings["loading_percent"].max()),
                "actual_max_loading_pct": float(ac.line_loadings["loading_percent"].max()),
                "lmp_spread_forecast": float(fc.bus_lmps.max() - fc.bus_lmps.min()),
                "lmp_spread_actual": float(ac.bus_lmps.max() - ac.bus_lmps.min()),
                "hidden_congestion_lines": ",".join(str(i) for i in hidden["line"].tolist()),
                "n_hidden_congestion": int(len(hidden)),
            }
        )

    bus_df = pd.DataFrame(bus_rows)
    line_df = pd.DataFrame(line_rows)
    summary_df = pd.DataFrame(summary_rows)
    bus_df.attrs["line_results"] = line_df
    bus_df.attrs["summary"] = summary_df
    bus_df.attrs["eval_day"] = eval_day
    return bus_df


def save_scenario_results(df: pd.DataFrame, output: Path) -> None:
    """Write bus-level CSV plus companion line and summary tables."""
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    stem = output.stem
    line_df = df.attrs.get("line_results")
    summary_df = df.attrs.get("summary")
    if isinstance(line_df, pd.DataFrame):
        line_df.to_csv(output.with_name(f"{stem}_lines.csv"), index=False)
    if isinstance(summary_df, pd.DataFrame):
        summary_df.to_csv(output.with_name(f"{stem}_summary.csv"), index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run IEEE 14-bus OPF under forecast vs actual load scales."
    )
    parser.add_argument("--hours", type=int, default=24, help="Hours to evaluate (max 24)")
    parser.add_argument(
        "--eval-day-offset",
        type=int,
        default=-1,
        help="Index into unique profile days for evaluation (default: last day)",
    )
    parser.add_argument(
        "--profile-days",
        type=int,
        default=30,
        help="Days of synthetic load history",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ieee14_opf_scenarios.csv"),
        help="Output CSV path for bus-level results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Also write ieee14_lmp.png and ieee14_scenario_comparison.png",
    )
    args = parser.parse_args(argv)

    print(f"Running OPF scenarios for {args.hours} hours...")
    df = run_opf_scenarios(
        n_hours=args.hours,
        eval_day_offset=args.eval_day_offset,
        profile_days=args.profile_days,
        seed=args.seed,
    )
    save_scenario_results(df, args.output)

    summary = df.attrs["summary"]
    eval_day = df.attrs["eval_day"]
    n_hidden = int(summary["n_hidden_congestion"].sum())
    print(f"Eval day: {pd.Timestamp(eval_day).date()}")
    print(f"Saved bus results to {args.output}")
    print(f"Saved line results to {args.output.with_name(args.output.stem + '_lines.csv')}")
    print(f"Saved summary to {args.output.with_name(args.output.stem + '_summary.csv')}")
    print(f"Hidden congestion events (line-hour): {n_hidden}")

    if args.save_figures:
        generate_opf_figures(
            results_dir=args.output.parent,
            profile_days=args.profile_days,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
