"""Slide-ready IEEE 14-bus OPF figures (baseline LMPs, hidden congestion)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.dispatch.ieee14_opf import (
    BINDING_LOADING_THRESHOLD,
    compare_binding_status,
    load_ieee14_case,
    prepare_congestion_case,
    run_power_flow,
    solve_opf_at_scale,
)
from src.forecast.load_forecaster import (
    build_system_load_profile,
    forecast_vs_actual_scales,
    stress_actual_scales,
)


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


def plot_baseline_opf(
    output_path: str | Path = "results/ieee14_lmp.png",
    cap_mult: float = 0.96,
) -> Path:
    """Baseline bus LMPs and top line loadings at unit load scale."""
    _apply_slide_style()
    net = load_ieee14_case()
    prepare_congestion_case(net, cap_mult=cap_mult)
    assert run_power_flow(net)

    baseline = solve_opf_at_scale(net, scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    baseline.bus_lmps.sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Bus LMPs ($/MWh)")
    axes[0].set_xlabel("Bus")
    axes[0].set_ylabel("LMP")
    axes[0].grid(True, alpha=0.3)

    top = baseline.line_loadings["loading_percent"].sort_values(ascending=False).head(8)
    top.plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_title("Top line loadings (%)")
    axes[1].set_xlabel("Loading %")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline OPF figure to {path}")
    return path


def plot_forecast_vs_actual(
    output_path: str | Path = "results/ieee14_scenario_comparison.png",
    profile_days: int = 30,
    seed: int = 42,
    cap_mult: float = 0.87,
) -> Path:
    """LMP shift and line loading comparison at peak load hour."""
    _apply_slide_style()
    profile = build_system_load_profile(days=profile_days, seed=seed)
    eval_day = profile.index.normalize().unique()[-1]
    actual, forecast = forecast_vs_actual_scales(profile, eval_day)
    actual_stressed = stress_actual_scales(actual, forecast)

    demo_net = load_ieee14_case()
    prepare_congestion_case(demo_net, cap_mult=cap_mult)

    peak_hour = int(actual.argmax())
    fc = solve_opf_at_scale(demo_net, scale=float(forecast[peak_hour]))
    ac = solve_opf_at_scale(demo_net, scale=float(actual_stressed[peak_hour]))
    binding = compare_binding_status(fc, ac, threshold=BINDING_LOADING_THRESHOLD)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lmp_shift = (ac.bus_lmps - fc.bus_lmps).sort_index()
    lmp_shift.plot(kind="bar", ax=axes[0], color="purple")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title(f"LMP shift (actual − forecast) at hour {peak_hour}")
    axes[0].set_ylabel("$/MWh")
    axes[0].grid(True, alpha=0.3)

    cmp = binding[["forecast_loading_pct", "actual_loading_pct"]].set_index(binding["line"])
    cmp.loc[cmp.max(axis=1).sort_values(ascending=False).head(6).index].plot(
        kind="barh", ax=axes[1]
    )
    axes[1].axvline(
        BINDING_LOADING_THRESHOLD,
        color="r",
        ls="--",
        label="binding threshold",
    )
    axes[1].set_xlabel("Loading %")
    axes[1].set_title("Line loading: forecast vs actual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forecast vs actual OPF figure to {path}")
    return path


def generate_opf_figures(
    results_dir: str | Path = "results",
    profile_days: int = 30,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write both IEEE 14-bus OPF figures into results_dir."""
    results_dir = Path(results_dir)
    lmp_path = plot_baseline_opf(results_dir / "ieee14_lmp.png")
    scenario_path = plot_forecast_vs_actual(
        results_dir / "ieee14_scenario_comparison.png",
        profile_days=profile_days,
        seed=seed,
    )
    return lmp_path, scenario_path
