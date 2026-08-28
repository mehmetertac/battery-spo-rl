# Week 07 Reflection — RL Primer + Capstone Regret

## What did I build?

This week closed the loop on the battery-spo-rl capstone: a **Gymnasium environment** for day-ahead battery arbitrage, a **PPO training script** (Stable-Baselines3), and integration of the trained agent into the existing **regret backtest harness** as approach `rl_ppo`.

| Component | Path | Role |
|---|---|---|
| Gymnasium env | `src/rl/battery_env.py` | 24-step episodes; discrete charge/hold/discharge; SOC dynamics match the LP |
| PPO training | `src/rl/train.py` | 50k timesteps on 60 OPSD days before the eval window |
| Harness hook | `src/eval/regret_backtest.py` | `--include-rl` adds `rl_ppo` rows to `results/regret.csv` |
| Tests | `tests/test_battery_env.py` | Synthetic SOC bounds, spike-day sanity, SB3 smoke |

The env reuses the same physics as `BatteryConfig` in `src/dispatch/battery_arbitrage.py` (round-trip efficiency, power/energy limits, terminal SoC penalty during training). Evaluation revenue uses `realized_revenue()` — the same function as MAE/SPO forecast LPs — so all approaches share one dollar metric.

**Run commands:**

```powershell
python -m src.rl.train --timesteps 50000 --output models/rl_ppo.zip
python -m src.eval --days 90 --include-rl
pytest tests/ -q
```

**90-day capstone results (Jul–Sep 2020):**

| Approach | Total revenue (EUR) | Cumulative regret (EUR) | Avg MAE (EUR/MWh) |
|---|---|---|---|
| Perfect foresight | 876,849 | 0 | — |
| MAE linear → LP | 113,171 | 763,678 | 9.38 |
| **RL PPO** | **610,037** | **266,813** | n/a |
| SPO linear → LP | 122,369 | 754,481 | 9.40 |

SPO still beats MAE on dollars (−EUR 9,198 regret) as before. The surprise is that **RL PPO beats both forecast-driven LPs** — but not for the reason we initially expected.

---

## What's still fuzzy?

### SPO+ gradient

The SPO+ surrogate gives a subgradient of the LP objective w.r.t. forecast prices, but the implementation here only fine-tunes a **bias term** on top of MAE-init OLS weights. I understand the envelope theorem intuition ( perturb forecast → re-solve LP → read off net export as sensitivity), but I don't yet have a crisp mental model for *when* bias-only SPO+ is enough vs when full weight updates (cvxpylayers / implicit differentiation) are necessary.

### OPF convergence

The IEEE 14-bus AC OPF runs reliably on the modified case14, but I haven't stress-tested ill-conditioned cases (near voltage collapse, transformer tap limits). Hidden congestion detection depends on comparing binding sets between forecast-load and actual-load solves — if OPF fails to converge on one side, the comparison breaks silently unless we assert solver status.

### RL reward shaping & information structure

Three open questions from this week's RL block:

1. **Terminal SoC penalty** (`TERMINAL_SOC_PENALTY = 50 EUR/MWh shortfall`) affects training but not the harness metric. Should evaluation also penalize infeasible terminal SOC?
2. **Mutually exclusive actions** (charge *or* discharge *or* hold at full power) vs the LP's simultaneous continuous variables — a modeling gap.
3. **Information asymmetry** (see honest finding below): the env observation at hour *t* includes **realized within-day prices** via lag features, while MAE/SPO LPs commit a full-day schedule from a day-start forecast only. That makes the RL vs forecast-LP comparison apples-to-oranges.

### cvxpylayers

End-to-end differentiable LP (PyTorch + cvxpylayers) could close the gap between SPO+ bias tuning and full decision-focused learning. I haven't tried it; the handover lists it as optional follow-up.

---

## Honest finding: RL vs forecast-driven LP

**Expected:** model-free RL trails a well-posed LP fed by a decent forecast — the problem is small, structured, and exactly what LPs solve.

**Observed:** RL PPO **beats** MAE and SPO forecast LPs on cumulative regret (266k vs 754–764k), but **still trails perfect foresight** by EUR 267k.

**Why the surprise is explainable, not a contradiction:**

| Pipeline | When decisions are made | Price information |
|---|---|---|
| MAE/SPO → LP | Once, before the day | 24h **forecast** from history before day start |
| RL PPO | Each hour, sequentially | History before day start **plus realized prices within the day** (lag features) |
| Perfect foresight LP | Once, oracle | All 24h **actual** prices |

The RL agent is closer to **real-time re-optimization** with growing information, not to **day-ahead market submission** from a single forecast. Of course it captures more arbitrage than a one-shot forecast LP — it partially sees the day unfold.

Against **perfect foresight** (the oracle upper bound in the regret table), RL still loses badly: EUR 267k cumulative regret over 90 days. That is the fair headline for "how close to optimal dispatch?"

For production storage operators in day-ahead markets, the relevant benchmark is usually **forecast → optimize → submit schedule**, not hourly re-learning from realized prices. In that framing, **SPO → LP is the more honest production pipeline**, and SPO's win over MAE on regret (−EUR 9k) is the core differentiator of this repo.

---

## Regret vs foresight — how is a real operator judged?

Real storage traders and asset optimizers are **not** graded on forecast MAE. They are graded on:

- **Realized P&L** vs budget, forward curve, or peer fleet
- **Opportunity cost** vs what they *could* have earned with perfect information (regret / value of perfect foresight)
- **Benchmark tracking error** — e.g. vs a simple heuristic or vs the firm's official optimizer

The regret table in `results/regret.csv` mirrors that logic:

```
regret = perfect_foresight_revenue − approach_revenue
```

Perfect foresight is unattainable; it sets the ceiling. MAE/SPO measure how much **forecast error costs you in dollars** when fed through the same LP. RL measures how much a **learned policy** costs you vs that ceiling — but only if the information structure matches the market you are simulating.

A real operator judged on day-ahead submission would care about SPO vs MAE regret, not about an RL agent that peeks at intra-day realizations through lag features. A real operator in **intraday continuous markets** might care about sequential policies — then RL's information structure is closer, but you'd still need realistic latency, fees, and degradation.

**Takeaway:** regret vs perfect foresight connects directly to **value of information**. Forecast quality (MAE) is a diagnostic; **decision regret** is the scoreboard. SPO aligns training with that scoreboard for the predict-then-optimize path. RL is a useful primer — it shows that information timing dominates algorithm choice — but for this repo's senior signal, **SPO + LP remains the story**.

---

## If behind schedule (not taken this week)

The master plan pre-authorizes dropping RL and using the time on SPO solidification. That trade was not needed — RL shipped. If descoped, note in README: *"RL deliberately descoped per master plan; SPO+ solidification prioritized."*
