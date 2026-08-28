"""Train a PPO agent on historical OPSD battery arbitrage days."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.opsd_prices import load_de_day_ahead
from src.dispatch.battery_arbitrage import BatteryConfig
from src.eval.regret_backtest import _complete_days, _default_eval_days
from src.rl.battery_env import BatteryArbitrageEnv


def _training_days(
    prices,
    n_eval_days: int,
    warmup_days: int,
    train_days: int,
) -> list:
    eval_days = _default_eval_days(prices, n_eval_days, warmup_days)
    eval_start = eval_days[0]
    complete = _complete_days(prices)
    eligible = [day for day in complete if day < eval_start]
    if len(eligible) < train_days:
        raise ValueError(
            f"Need {train_days} training days before eval window; found {len(eligible)}."
        )
    return eligible[-train_days:]


def train_ppo(
    output_path: str | Path = "models/rl_ppo.zip",
    timesteps: int = 50_000,
    n_eval_days: int = 90,
    train_days: int = 60,
    warmup_days: int = 7,
    battery_config: BatteryConfig | None = None,
    seed: int = 42,
) -> Path:
    if battery_config is None:
        battery_config = BatteryConfig()

    prices = load_de_day_ahead()
    days = _training_days(prices, n_eval_days, warmup_days, train_days)
    print(
        f"Training PPO on {len(days)} days "
        f"({days[0].date()} to {days[-1].date()}), eval window starts after."
    )

    def make_env() -> BatteryArbitrageEnv:
        return BatteryArbitrageEnv(
            config=battery_config,
            prices=prices,
            training_days=days,
            seed=seed,
        )

    vec_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=timesteps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Saved model to {output_path}")
    vec_env.close()
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO battery arbitrage agent.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="PPO training timesteps.")
    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_ppo.zip",
        help="Output model path.",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=60,
        help="Number of complete days before eval window for training resets.",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=90,
        help="Eval window size used to locate training days before it.",
    )
    parser.add_argument("--power-mw", type=float, default=100.0)
    parser.add_argument("--energy-mwh", type=float, default=400.0)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    initial_soc = args.energy_mwh / 2.0
    battery = BatteryConfig(
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        round_trip_efficiency=args.efficiency,
        initial_soc_mwh=initial_soc,
        terminal_soc_min_mwh=initial_soc,
    )
    train_ppo(
        output_path=args.output,
        timesteps=args.timesteps,
        n_eval_days=args.eval_days,
        train_days=args.train_days,
        battery_config=battery,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
