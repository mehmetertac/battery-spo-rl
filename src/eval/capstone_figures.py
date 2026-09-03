"""Capstone figures: cumulative regret, MAE vs regret scatter, regime bars, sensitivity heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.capstone_summary import APPROACH_LABELS, CAPSTONE_APPROACHES


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

APPROACH_COLORS = {
    "perfect_foresight": "#2ca02c",
    "mae_linear": "#d62728",
    "spo_linear": "#1f77b4",
    "rl_ppo": "#ff7f0e",
}

APPROACH_MARKERS = {
    "mae_linear": "o",
    "spo_linear": "s",
}


def _apply_slide_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _cumulative_regret_series(df: pd.DataFrame, approach: str) -> pd.Series:
    subset = df.loc[df["approach"] == approach].copy()
    if subset.empty:
        return pd.Series(dtype=float)
    subset["date_ts"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date_ts")
    if approach == "perfect_foresight":
        return pd.Series(0.0, index=subset["date_ts"].values)
    if subset["cumulative_regret"].notna().any():
        return pd.Series(subset["cumulative_regret"].values, index=subset["date_ts"].values)
    return subset["regret"].cumsum().reset_index(drop=True).pipe(
        lambda s: pd.Series(s.values, index=subset["date_ts"].values)
    )


def plot_cumulative_regret(
    df: pd.DataFrame,
    output_path: str | Path = "results/cumulative_regret.png",
) -> Path:
    """Money plot: cumulative regret over the backtest for all four approaches."""
    _apply_slide_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for approach in CAPSTONE_APPROACHES:
        series = _cumulative_regret_series(df, approach)
        if series.empty:
            continue
        label = APPROACH_LABELS.get(approach, approach)
        ax.plot(
            series.index,
            series.values,
            label=label,
            color=APPROACH_COLORS.get(approach, "gray"),
            linewidth=2 if approach != "perfect_foresight" else 1.5,
            linestyle="-" if approach != "perfect_foresight" else "--",
        )
        if approach != "perfect_foresight":
            final = float(series.iloc[-1])
            ax.annotate(
                f"{final/1000:.0f}k",
                xy=(series.index[-1], final),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                color=APPROACH_COLORS.get(approach, "gray"),
            )

    ax.set_title("Cumulative regret vs perfect foresight (90-day backtest)")
    ax.set_xlabel("Evaluation day")
    ax.set_ylabel("Cumulative regret (EUR)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cumulative regret plot to {path}")
    return path


def plot_forecast_mae_vs_regret(
    df: pd.DataFrame,
    output_path: str | Path = "results/forecast_mae_vs_regret.png",
) -> Path:
    """Scatter: daily forecast MAE vs daily regret — accuracy != decision quality."""
    _apply_slide_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for approach in ("mae_linear", "spo_linear"):
        subset = df.loc[df["approach"] == approach].dropna(subset=["forecast_mae", "regret"])
        if subset.empty:
            continue
        x = subset["forecast_mae"].to_numpy()
        y = subset["regret"].to_numpy()
        label = APPROACH_LABELS.get(approach, approach)
        color = APPROACH_COLORS.get(approach, "gray")
        ax.scatter(
            x,
            y,
            alpha=0.55,
            s=36,
            label=label,
            color=color,
            marker=APPROACH_MARKERS.get(approach, "o"),
            edgecolors="white",
            linewidths=0.4,
        )
        if len(x) >= 2:
            r = _pearson_r(x, y)
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, coef[0] * x_line + coef[1], color=color, alpha=0.35, linewidth=1.5)
            y_pos = 0.95 if approach == "mae_linear" else 0.88
            ax.text(
                0.02,
                y_pos,
                f"{label}: r = {r:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                color=color,
                verticalalignment="top",
            )

    ax.set_title("Forecast accuracy ≠ decision quality")
    ax.set_xlabel("Daily forecast MAE (EUR/MWh)")
    ax.set_ylabel("Daily regret vs perfect foresight (EUR)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MAE vs regret scatter to {path}")
    return path


def plot_regime_robustness(
    regime_df: pd.DataFrame,
    output_path: str | Path = "results/regime_robustness.png",
) -> Path:
    """Grouped bar chart of cumulative regret by regime."""
    _apply_slide_style()
    plot_approaches = [a for a in ("mae_linear", "spo_linear", "rl_ppo") if a in regime_df["approach"].values]
    regimes = [r for r in ("calm", "volatile") if r in regime_df["regime"].values]
    if not plot_approaches or not regimes:
        raise ValueError("Regime dataframe missing required approaches or regimes.")

    x = np.arange(len(regimes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, approach in enumerate(plot_approaches):
        values = []
        for regime in regimes:
            row = regime_df.loc[
                (regime_df["regime"] == regime) & (regime_df["approach"] == approach),
                "dollar_regret",
            ]
            values.append(float(row.iloc[0]) if not row.empty else 0.0)
        offset = (i - (len(plot_approaches) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=APPROACH_LABELS.get(approach, approach),
            color=APPROACH_COLORS.get(approach, "gray"),
        )
        for bar, val in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val/1000:.0f}k",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_title("Regret by market regime (weekly price volatility)")
    ax.set_ylabel("Total regret (EUR)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regime robustness plot to {path}")
    return path


def plot_battery_sensitivity(
    sens_df: pd.DataFrame,
    output_path: str | Path = "results/battery_sensitivity.png",
) -> Path:
    """Heatmap of SPO minus MAE regret over efficiency x energy grid."""
    _apply_slide_style()
    pivot = sens_df.pivot(index="energy_mwh", columns="efficiency", values="spo_minus_mae_regret")
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(7, 5))
    data = pivot.to_numpy()
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1.0)
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(e)}" for e in pivot.index])
    ax.set_xlabel("Round-trip efficiency")
    ax.set_ylabel("Energy capacity (MWh)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            ax.text(
                j,
                i,
                f"{val:,.0f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black" if abs(val) < vmax * 0.6 else "white",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("SPO regret advantage vs MAE (EUR)\n(negative = SPO wins)")
    ax.set_title("Battery sensitivity: SPO vs MAE regret (LP-only 3×3 grid)")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved battery sensitivity plot to {path}")
    return path


def generate_all_figures(
    regret_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    sens_df: pd.DataFrame | None = None,
    results_dir: str | Path = "results",
) -> None:
    """Generate all capstone figures into results/."""
    results_dir = Path(results_dir)
    plot_cumulative_regret(regret_df, results_dir / "cumulative_regret.png")
    plot_forecast_mae_vs_regret(regret_df, results_dir / "forecast_mae_vs_regret.png")
    plot_regime_robustness(regime_df, results_dir / "regime_robustness.png")
    if sens_df is not None and not sens_df.empty:
        plot_battery_sensitivity(sens_df, results_dir / "battery_sensitivity.png")
