"""Post-install smoke test: imports and one synthetic battery LP (no OPSD / RL training)."""

from __future__ import annotations

import sys

import numpy as np


def test_python_version_supported() -> None:
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 11)
    assert (major, minor) < (3, 13)


def test_wheel_stack_imports() -> None:
    import cvxpy  # noqa: F401
    import gymnasium  # noqa: F401
    import lightgbm  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy as np_mod  # noqa: F401
    import pandas  # noqa: F401
    import pandapower  # noqa: F401
    import sklearn  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch  # noqa: F401

    assert np_mod.__version__.startswith("1.")


def test_battery_lp_replays_revenue() -> None:
    from src.dispatch.battery_arbitrage import BatteryConfig, realized_revenue, solve_battery_arbitrage

    prices = np.linspace(20, 100, 24)
    config = BatteryConfig()
    result = solve_battery_arbitrage(prices, config)
    replay = realized_revenue(result.schedule, prices)
    assert abs(result.revenue - replay) < 1e-4