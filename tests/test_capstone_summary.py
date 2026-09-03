"""Tests for capstone summary and figures (synthetic data, no OPSD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.capstone_figures import (
    generate_all_figures,
    plot_battery_sensitivity,
    plot_cumulative_regret,
    plot_forecast_mae_vs_regret,
    plot_regime_robustness,
)
from src.eval.capstone_summary import (
    build_master_table,
    build_regime_robustness,
    summarize_sensitivity_cell,
    tag_price_regimes,
)


def _synthetic_regret_df(n_days: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cum = {"mae_linear": 0.0, "spo_linear": 0.0, "rl_ppo": 0.0}
    for i in range(n_days):
        date = f"2020-07-{i + 1:02d}"
        pf = 10_000.0 + i * 100
        rows.append(
            {
                "date": date,
                "approach": "perfect_foresight",
                "pf_revenue": pf,
                "approach_revenue": pf,
                "regret": 0.0,
                "cumulative_regret": np.nan,
                "forecast_mae": np.nan,
            }
        )
        for approach, mae, regret in [
            ("mae_linear", 9.0 + i * 0.1, 800.0 + i * 10),
            ("rl_ppo", np.nan, 300.0 + i * 5),
            ("spo_linear", 9.2 + i * 0.1, 750.0 + i * 10),
        ]:
            cum_key = approach if approach != "rl_ppo" else "rl_ppo"
            if approach in cum:
                cum[cum_key] += regret
            rows.append(
                {
                    "date": date,
                    "approach": approach,
                    "pf_revenue": pf,
                    "approach_revenue": pf - regret,
                    "regret": regret,
                    "cumulative_regret": cum.get(cum_key, regret),
                    "forecast_mae": mae,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_prices(n_days: int = 5) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2020-07-01 00:00",
        periods=n_days * 24,
        freq="h",
        tz="Europe/Berlin",
    )
    # Higher volatility in second half of window
    base = np.where(np.arange(len(timestamps)) >= n_days * 12, 80.0, 30.0)
    noise = np.random.default_rng(0).normal(0, 5, len(timestamps))
    return pd.DataFrame({"price_eur_mwh": base + noise}, index=timestamps)


def test_build_master_table_totals():
    df = _synthetic_regret_df()
    table = build_master_table(df)
    assert len(table) == 4
    pf_total = df.loc[df["approach"] == "perfect_foresight", "pf_revenue"].sum()
    mae = table.loc[table["approach"] == "mae_linear"].iloc[0]
    assert mae["dollar_regret"] == pytest.approx(
        df.loc[df["approach"] == "mae_linear", "regret"].sum()
    )
    assert mae["regret_pct_of_pf"] == pytest.approx(
        mae["dollar_regret"] / pf_total * 100
    )
    assert mae["avg_forecast_mae"] == pytest.approx(9.2, abs=0.01)
    pf_row = table.loc[table["approach"] == "perfect_foresight"].iloc[0]
    assert pf_row["dollar_regret"] == 0.0


def test_tag_price_regimes_assigns_labels():
    df = _synthetic_regret_df()
    prices = _synthetic_prices()
    tagged = tag_price_regimes(df, prices)
    assert tagged["regime"].notna().all()
    assert set(tagged["regime"].unique()) <= {"calm", "volatile", "unknown"}


def test_build_regime_robustness_has_mae_and_spo():
    df = _synthetic_regret_df()
    prices = _synthetic_prices()
    regime = build_regime_robustness(df, prices)
    assert not regime.empty
    for regime_name in regime["regime"].unique():
        subset = regime.loc[regime["regime"] == regime_name]
        assert "mae_linear" in subset["approach"].values
        assert "spo_linear" in subset["approach"].values


def test_summarize_sensitivity_cell():
    df = _synthetic_regret_df()
    summary = summarize_sensitivity_cell(df)
    assert summary["mae_regret"] > summary["spo_regret"]
    assert summary["spo_minus_mae_regret"] < 0


def test_figures_write_png(tmp_path):
    df = _synthetic_regret_df()
    prices = _synthetic_prices()
    regime = build_regime_robustness(df, prices)
    sens = pd.DataFrame(
        [
            {"efficiency": 0.85, "energy_mwh": 300, "spo_minus_mae_regret": -500},
            {"efficiency": 0.90, "energy_mwh": 300, "spo_minus_mae_regret": -400},
            {"efficiency": 0.95, "energy_mwh": 300, "spo_minus_mae_regret": -300},
            {"efficiency": 0.85, "energy_mwh": 400, "spo_minus_mae_regret": -600},
            {"efficiency": 0.90, "energy_mwh": 400, "spo_minus_mae_regret": -450},
            {"efficiency": 0.95, "energy_mwh": 400, "spo_minus_mae_regret": -350},
            {"efficiency": 0.85, "energy_mwh": 500, "spo_minus_mae_regret": -550},
            {"efficiency": 0.90, "energy_mwh": 500, "spo_minus_mae_regret": -420},
            {"efficiency": 0.95, "energy_mwh": 500, "spo_minus_mae_regret": -320},
        ]
    )
    plot_cumulative_regret(df, tmp_path / "cumulative_regret.png")
    plot_forecast_mae_vs_regret(df, tmp_path / "forecast_mae_vs_regret.png")
    plot_regime_robustness(regime, tmp_path / "regime_robustness.png")
    plot_battery_sensitivity(sens, tmp_path / "battery_sensitivity.png")
    generate_all_figures(df, regime, sens, tmp_path)
    for name in (
        "cumulative_regret.png",
        "forecast_mae_vs_regret.png",
        "regime_robustness.png",
        "battery_sensitivity.png",
    ):
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 0
