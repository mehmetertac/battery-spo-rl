"""Capstone orchestrator: backtest, sensitivity grid, tables, and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.opsd_prices import available_day_range
from src.dispatch.battery_arbitrage import BatteryConfig
from src.eval.capstone_figures import generate_all_figures
from src.eval.capstone_summary import (
    load_regret_csv,
    run_battery_sensitivity_grid,
    save_master_table,
    save_regime_robustness,
)
from src.eval.regret_backtest import run_regret_backtest


def _battery_from_args(args: argparse.Namespace) -> BatteryConfig:
    initial_soc = args.energy_mwh / 2.0
    return BatteryConfig(
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        round_trip_efficiency=args.efficiency,
        initial_soc_mwh=initial_soc,
        terminal_soc_min_mwh=initial_soc,
    )


def run_capstone(args: argparse.Namespace) -> None:
    """Execute capstone pipeline per CLI flags."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    regret_path = Path(args.from_csv) if args.from_csv else results_dir / "regret.csv"

    if not args.skip_backtest:
        battery = _battery_from_args(args)
        run_regret_backtest(
            n_days=args.days,
            start=args.start,
            output_path=regret_path,
            battery_config=battery,
            warmup_days=args.warmup_days,
            include_rl=args.include_rl,
            rl_model_path=args.rl_model,
            spo_epochs=args.spo_epochs,
            spo_learning_rate=args.spo_lr,
        )
    elif not regret_path.exists():
        raise FileNotFoundError(
            f"--skip-backtest set but regret CSV not found at {regret_path}. "
            "Run backtest first or pass --from-csv."
        )

    regret_df = load_regret_csv(regret_path)
    master_table = save_master_table(regret_df, results_dir / "master_table.csv")
    regime_df = save_regime_robustness(regret_df, results_dir / "regime_robustness.csv")

    sens_df = None
    sens_path = results_dir / "battery_sensitivity.csv"
    if not args.skip_sensitivity:
        sens_df = run_battery_sensitivity_grid(
            n_days=args.days,
            start=args.start,
            output_path=sens_path,
            power_mw=args.power_mw,
            warmup_days=args.warmup_days,
            spo_epochs=args.spo_epochs,
            spo_learning_rate=args.spo_lr,
        )
    elif sens_path.exists():
        import pandas as pd

        sens_df = pd.read_csv(sens_path)
        print(f"Loaded existing battery sensitivity from {sens_path}")

    generate_all_figures(regret_df, regime_df, sens_df, results_dir)

    print("\n=== Capstone complete ===")
    print(f"Results directory: {results_dir.resolve()}")
    if not master_table.empty:
        spo = master_table.loc[master_table["approach"] == "spo_linear"]
        mae = master_table.loc[master_table["approach"] == "mae_linear"]
        if not spo.empty and not mae.empty:
            delta = float(spo["dollar_regret"].iloc[0] - mae["dollar_regret"].iloc[0])
            print(f"SPO vs MAE headline: EUR {delta:,.0f} regret delta (negative = SPO wins)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-way capstone: backtest, sensitivity, tables, and figures."
    )
    parser.add_argument("--days", type=int, default=90, help="Evaluation days.")
    parser.add_argument("--start", type=str, default=None, help="First eval day (YYYY-MM-DD).")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for all capstone outputs.",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Load regret CSV from this path instead of results-dir/regret.csv.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest; load existing regret CSV.",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip 3x3 battery sensitivity grid (~4-6 hours).",
    )
    parser.add_argument("--power-mw", type=float, default=100.0)
    parser.add_argument("--energy-mwh", type=float, default=400.0)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--include-rl", action="store_true")
    parser.add_argument("--rl-model", type=str, default="models/rl_ppo.zip")
    parser.add_argument("--spo-epochs", type=int, default=5)
    parser.add_argument("--spo-lr", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    first_available, last_available = available_day_range()
    print(
        "OPSD DE-LU day-ahead prices available "
        f"from {first_available.date()} to {last_available.date()}."
    )
    run_capstone(args)


if __name__ == "__main__":
    main()
