"""Capstone summary tables: master table, regime robustness, battery sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.opsd_prices import load_de_day_ahead
from src.dispatch.battery_arbitrage import BatteryConfig
from src.eval.regret_backtest import run_regret_backtest
from src.forecast.price_forecaster import build_feature_matrix

APPROACH_LABELS = {
    "perfect_foresight": "LP + perfect foresight",
    "mae_linear": "LP + MAE forecast",
    "rl_ppo": "RL agent (PPO)",
    "spo_linear": "LP + SPO forecast",
}

CAPSTONE_APPROACHES = ("perfect_foresight", "mae_linear", "rl_ppo", "spo_linear")

DEFAULT_EFFICIENCIES = (0.85, 0.90, 0.95)
DEFAULT_ENERGIES = (300.0, 400.0, 500.0)


def load_regret_csv(path: str | Path) -> pd.DataFrame:
    """Load regret backtest CSV and normalize dtypes."""
    df = pd.read_csv(path)
    for col in ("pf_revenue", "approach_revenue", "regret", "cumulative_regret", "forecast_mae"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _pf_total_revenue(df: pd.DataFrame) -> float:
    pf = df.loc[df["approach"] == "perfect_foresight", "pf_revenue"]
    if pf.empty:
        raise ValueError("No perfect_foresight rows in regret CSV.")
    return float(pf.sum())


def build_master_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regret CSV into one row per capstone approach."""
    pf_total = _pf_total_revenue(df)
    rows: list[dict[str, object]] = []

    for approach in CAPSTONE_APPROACHES:
        subset = df.loc[df["approach"] == approach]
        if subset.empty:
            continue
        total_revenue = float(subset["approach_revenue"].sum())
        dollar_regret = float(subset["regret"].sum())
        regret_pct = (dollar_regret / pf_total * 100.0) if pf_total > 0 else 0.0
        mae_values = subset["forecast_mae"].dropna()
        avg_mae = float(mae_values.mean()) if not mae_values.empty else np.nan

        rows.append(
            {
                "approach": approach,
                "label": APPROACH_LABELS.get(approach, approach),
                "total_revenue": total_revenue,
                "dollar_regret": dollar_regret,
                "regret_pct_of_pf": regret_pct,
                "avg_forecast_mae": avg_mae,
            }
        )

    return pd.DataFrame(rows)


def print_master_table(table: pd.DataFrame) -> None:
    """Pretty-print master table for slide copy-paste."""
    print("\n=== Capstone master table ===")
    print(
        f"{'Approach':<28} {'Revenue (EUR)':>14} {'Regret (EUR)':>14} "
        f"{'Regret % PF':>12} {'Avg MAE':>10}"
    )
    print("-" * 82)
    for _, row in table.iterrows():
        mae_str = f"{row['avg_forecast_mae']:.2f}" if pd.notna(row["avg_forecast_mae"]) else "n/a"
        print(
            f"{row['label']:<28} {row['total_revenue']:>14,.0f} "
            f"{row['dollar_regret']:>14,.0f} {row['regret_pct_of_pf']:>11.1f}% "
            f"{mae_str:>10}"
        )
    print(
        "\nNote: RL uses intra-day realized price lags; MAE/SPO commit at day start "
        "(see WEEK_07_REFLECTION.md)."
    )


def save_master_table(
    df: pd.DataFrame,
    output_path: str | Path = "results/master_table.csv",
) -> pd.DataFrame:
    """Build and write master table CSV."""
    table = build_master_table(df)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print_master_table(table)
    print(f"Saved master table to {path}")
    return table


def _weekly_price_volatility(prices: pd.DataFrame) -> pd.Series:
    """Std of hourly prices per ISO calendar week."""
    hourly = prices["price_eur_mwh"].copy()
    hourly.index = pd.to_datetime(hourly.index)
    week_key = hourly.index.isocalendar().week.astype(int)
    year_key = hourly.index.isocalendar().year.astype(int)
    grouped = hourly.groupby([year_key, week_key])
    return grouped.std()


def tag_price_regimes(
    df: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tag each eval day as volatile or calm based on weekly price std."""
    if prices is None:
        prices = load_de_day_ahead()

    week_vol = _weekly_price_volatility(prices)
    eval_dates = df.loc[df["approach"] != "perfect_foresight", "date"].drop_duplicates()
    week_vols: list[float] = []

    for date_str in eval_dates:
        day = pd.Timestamp(date_str, tz="Europe/Berlin")
        iso = day.isocalendar()
        key = (iso.year, iso.week)
        if key in week_vol.index:
            week_vols.append(float(week_vol.loc[key]))
        else:
            week_vols.append(np.nan)

    valid_vols = [v for v in week_vols if not np.isnan(v)]
    median_vol = float(np.median(valid_vols)) if valid_vols else 0.0

    regime_map: dict[str, str] = {}
    for date_str, vol in zip(eval_dates, week_vols, strict=True):
        if np.isnan(vol):
            regime_map[date_str] = "unknown"
        else:
            regime_map[date_str] = "volatile" if vol >= median_vol else "calm"

    tagged = df.copy()
    tagged["regime"] = tagged["date"].map(regime_map)
    return tagged


def build_regime_robustness(
    df: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate regret by market regime (volatile vs calm weeks)."""
    tagged = tag_price_regimes(df, prices)
    pf_total = _pf_total_revenue(tagged)
    rows: list[dict[str, object]] = []

    for regime in ("calm", "volatile"):
        regime_df = tagged.loc[tagged["regime"] == regime]
        if regime_df.empty:
            continue
        pf_regime = float(
            regime_df.loc[regime_df["approach"] == "perfect_foresight", "pf_revenue"].sum()
        )
        for approach in CAPSTONE_APPROACHES:
            subset = regime_df.loc[regime_df["approach"] == approach]
            if subset.empty:
                continue
            dollar_regret = float(subset["regret"].sum())
            regret_pct = (dollar_regret / pf_regime * 100.0) if pf_regime > 0 else 0.0
            mae_values = subset["forecast_mae"].dropna()
            avg_mae = float(mae_values.mean()) if not mae_values.empty else np.nan
            rows.append(
                {
                    "regime": regime,
                    "approach": approach,
                    "label": APPROACH_LABELS.get(approach, approach),
                    "n_days": int(subset["date"].nunique()),
                    "total_revenue": float(subset["approach_revenue"].sum()),
                    "dollar_regret": dollar_regret,
                    "regret_pct_of_pf": regret_pct,
                    "avg_forecast_mae": avg_mae,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result.attrs["pf_total"] = pf_total
    return result


def save_regime_robustness(
    df: pd.DataFrame,
    output_path: str | Path = "results/regime_robustness.csv",
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and write regime robustness CSV."""
    table = build_regime_robustness(df, prices)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

    print("\n=== Regime robustness (volatile vs calm weeks) ===")
    for regime in ("calm", "volatile"):
        subset = table.loc[table["regime"] == regime]
        if subset.empty:
            continue
        print(f"\n{regime.upper()} ({int(subset['n_days'].iloc[0])} days):")
        for _, row in subset.iterrows():
            mae_str = f"{row['avg_forecast_mae']:.2f}" if pd.notna(row["avg_forecast_mae"]) else "n/a"
            print(
                f"  {row['label']:<28} regret EUR {row['dollar_regret']:>10,.0f}  "
                f"({row['regret_pct_of_pf']:.1f}% of PF)  MAE {mae_str}"
            )
        mae_row = subset.loc[subset["approach"] == "mae_linear", "dollar_regret"]
        spo_row = subset.loc[subset["approach"] == "spo_linear", "dollar_regret"]
        if not mae_row.empty and not spo_row.empty:
            delta = float(spo_row.iloc[0] - mae_row.iloc[0])
            winner = "SPO" if delta < 0 else "MAE"
            print(f"  SPO vs MAE: EUR {delta:,.0f} ({winner} wins on dollars)")

    print(f"\nSaved regime robustness to {path}")
    return table


def _battery_config(
    power_mw: float,
    energy_mwh: float,
    efficiency: float,
) -> BatteryConfig:
    initial_soc = energy_mwh / 2.0
    return BatteryConfig(
        power_mw=power_mw,
        energy_mwh=energy_mwh,
        round_trip_efficiency=efficiency,
        initial_soc_mwh=initial_soc,
        terminal_soc_min_mwh=initial_soc,
    )


def summarize_sensitivity_cell(results: pd.DataFrame) -> dict[str, float]:
    """Extract MAE/SPO regret totals from one backtest run."""
    mae = results.loc[results["approach"] == "mae_linear", "regret"].sum()
    spo = results.loc[results["approach"] == "spo_linear", "regret"].sum()
    return {
        "mae_regret": float(mae),
        "spo_regret": float(spo),
        "spo_minus_mae_regret": float(spo - mae),
    }


def run_battery_sensitivity_grid(
    n_days: int = 90,
    start: str | pd.Timestamp | None = None,
    output_path: str | Path = "results/battery_sensitivity.csv",
    power_mw: float = 100.0,
    efficiencies: tuple[float, ...] = DEFAULT_EFFICIENCIES,
    energies: tuple[float, ...] = DEFAULT_ENERGIES,
    warmup_days: int = 7,
    spo_epochs: int = 5,
    spo_learning_rate: float = 5.0,
) -> pd.DataFrame:
    """Run LP-only 3x3 grid over efficiency and energy capacity."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: pd.DataFrame | None = None
    done: set[tuple[float, float]] = set()
    if path.exists():
        existing = pd.read_csv(path)
        for _, row in existing.iterrows():
            done.add((float(row["efficiency"]), float(row["energy_mwh"])))
        if done:
            print(f"Resuming battery sensitivity: {len(done)} cells already complete.")

    print("Loading OPSD prices and feature matrix (shared across grid)...")
    prices = load_de_day_ahead()
    all_features, all_targets, _ = build_feature_matrix(prices)

    rows: list[dict[str, object]] = []
    if existing is not None:
        rows.extend(existing.to_dict(orient="records"))

    total = len(efficiencies) * len(energies)
    cell = 0

    for efficiency in efficiencies:
        for energy_mwh in energies:
            cell += 1
            key = (efficiency, energy_mwh)
            if key in done:
                print(
                    f"Skipping battery sensitivity {cell}/{total}: "
                    f"eff={efficiency:.2f}, energy={energy_mwh:.0f} MWh (done)",
                    flush=True,
                )
                continue

            print(
                f"\n=== Battery sensitivity {cell}/{total}: "
                f"eff={efficiency:.2f}, energy={energy_mwh:.0f} MWh ===",
                flush=True,
            )
            config = _battery_config(power_mw, energy_mwh, efficiency)
            results = run_regret_backtest(
                n_days=n_days,
                start=start,
                output_path=None,
                battery_config=config,
                warmup_days=warmup_days,
                include_rl=False,
                include_lightgbm=False,
                spo_epochs=spo_epochs,
                spo_learning_rate=spo_learning_rate,
                prices=prices,
                precomputed_features=all_features,
                precomputed_targets=all_targets,
            )
            summary = summarize_sensitivity_cell(results)
            row = {
                "efficiency": efficiency,
                "energy_mwh": energy_mwh,
                "power_mw": power_mw,
                **summary,
                "spo_wins": summary["spo_minus_mae_regret"] < 0,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"Checkpoint saved to {path}", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(path, index=False)

    wins = int(grid["spo_wins"].sum())
    print(f"\n=== Battery sensitivity summary: SPO wins {wins}/{len(grid)} cells ===")
    print(grid.to_string(index=False, formatters={"spo_minus_mae_regret": "{:,.0f}".format}))
    print(f"Saved battery sensitivity to {path}")
    return grid
