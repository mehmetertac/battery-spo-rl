"""Animate cumulative regret plot for README demo (docs/figures/cumulative_regret.gif)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REGRET_CSV = ROOT / "results" / "regret.csv"
OUT_GIF = ROOT / "docs" / "figures" / "cumulative_regret.gif"

APPROACH_COLORS = {
    "perfect_foresight": "#2ca02c",
    "mae_linear": "#d62728",
    "spo_linear": "#1f77b4",
    "rl_ppo": "#ff7f0e",
}
APPROACH_LABELS = {
    "perfect_foresight": "Perfect foresight",
    "mae_linear": "MAE linear",
    "spo_linear": "SPO linear",
    "rl_ppo": "RL PPO",
}
ORDER = ["perfect_foresight", "mae_linear", "spo_linear", "rl_ppo"]


def _frame(df: pd.DataFrame, *, end_day: pd.Timestamp, dpi: int = 100) -> Image.Image:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=dpi)
    view = df.loc[pd.to_datetime(df["date"]) <= end_day]
    for approach in ORDER:
        sub = view.loc[view["approach"] == approach].sort_values("date")
        if sub.empty:
            continue
        cum = sub["regret"].cumsum()
        dates = pd.to_datetime(sub["date"])
        ax.plot(
            dates,
            cum,
            label=APPROACH_LABELS.get(approach, approach),
            color=APPROACH_COLORS.get(approach, "gray"),
            linewidth=2 if approach != "perfect_foresight" else 1.5,
            linestyle="-" if approach != "perfect_foresight" else "--",
        )
    ax.set_title("Cumulative regret vs perfect foresight (90-day backtest)")
    ax.set_xlabel("Evaluation day")
    ax.set_ylabel("Cumulative regret (EUR)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(buf[:, :, :3])


def main() -> int:
    if not REGRET_CSV.is_file():
        raise SystemExit(f"missing {REGRET_CSV} — run python -m src.eval.capstone first")
    df = pd.read_csv(REGRET_CSV)
    dates = sorted(pd.to_datetime(df["date"].unique()))
    if len(dates) < 4:
        raise SystemExit("not enough days in regret.csv")
    step = max(1, len(dates) // 12)
    pick = dates[step - 1 :: step]
    if dates[-1] not in pick:
        pick.append(dates[-1])
    frames = [_frame(df, end_day=d) for d in pick]
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUT_GIF} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
