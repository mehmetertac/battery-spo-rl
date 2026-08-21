# Handover — battery-spo-rl

**Repo:** https://github.com/mehmetertac/battery-spo-rl  
**Last updated:** 2026-08-21  
**Branch:** `main` (up to date with remote)

---

## Project goal

Bridge ML and the optimization layer of power systems. The headline metric is **regret in dollars vs perfect foresight** (decision quality), not forecast accuracy (MAE). Forecasting is a means, not an end.

This is the most senior-signal artifact of the 12-week master plan. If the week slips, drop RL — never drop SPO.

---

## What is done (Day 1)

| Item | Status |
|---|---|
| Repo scaffold (`src/`, `notebooks/`, `results/`, `requirements.txt`) | Done |
| LP economic dispatch toy (3 generators, 24h demand) | Done |
| LMP extraction from cvxpy power-balance duals | Done |
| LMP verification (dual = marginal generator cost) | Done — 24/24 hours pass |
| Walkthrough notebook with plots | Done |
| README with LP/MILP mental model and dispatch framing | Done |
| Git push to remote | Done (`7968a79`) |

### Key commit

```
7968a79 Day 1: scaffold repo and economic dispatch LP toy with LMP duals
```

---

## Repo layout

```
battery-spo-rl/
├── README.md
├── handover.md              ← this file
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   └── 01_economic_dispatch_toy.ipynb
├── src/
│   └── dispatch/
│       ├── __init__.py
│       ├── __main__.py      ← CLI entry: python -m src.dispatch
│       └── economic_dispatch.py
└── results/                 ← generated outputs (gitignored except .gitkeep)
    └── .gitkeep
```

---

## How to run

```powershell
cd battery-spo-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# CLI — prints LMP summary, saves dispatch CSV
python -m src.dispatch

# Notebook walkthrough
jupyter notebook notebooks/01_economic_dispatch_toy.ipynb
```

Expected CLI output:

```
Status: optimal
Total cost: $66,827.90
LMP verification: 24/24 hours match marginal cost
Saved dispatch to results\dispatch_day1.csv
```

Generated artifacts (local, not committed):

- `results/dispatch_day1.csv`
- `results/economic_dispatch_lmp.png`

---

## Core module API

**File:** `src/dispatch/economic_dispatch.py`

| Symbol | Purpose |
|---|---|
| `Generator` | Dataclass: name, marginal_cost ($/MWh), pmax (MW) |
| `DispatchResult` | dispatch DataFrame, total_cost, lmps Series, status |
| `default_generators()` | 3-gen merit-order stack ($20, $35, $50/MWh) |
| `default_demand()` | 24h sinusoidal profile (~120–200 MW) |
| `solve_economic_dispatch()` | cvxpy LP solver; returns dispatch + LMPs |
| `verify_lmps()` | Assert duals match marginal generator cost |
| `marginal_generator_cost()` | Expected LMP from dispatch (for verification) |

Default generators:

| Generator | Marginal cost | Pmax |
|---|---|---|
| gen0_cheap | $20/MWh | 100 MW |
| gen1_mid | $35/MWh | 80 MW |
| gen2_expensive | $50/MWh | 50 MW |

---

## Technical notes for the next person

### cvxpy dual sign convention

cvxpy returns **negative** dual values for equality constraints in minimization problems. The module negates them:

```python
lmps = [-float(c.dual_value) for c in balance_cons]
```

Without this negation, LMPs appear as -$20 / -$35 instead of +$20 / +$35.

### Solver fallback chain

ECOS → CLARABEL → SCS. The module raises if none are installed (they ship with cvxpy on Windows).

### LMP = marginal generator cost

At each hour, the LMP equals the marginal cost of the **most expensive generator still dispatched** (merit-order stacking). When demand exceeds gen0's 100 MW capacity, LMP jumps from $20 to $35. This is the price signal that pays off on pandapower day (Thursday).

---

## Week roadmap (remaining)

| Day | Topic | Priority if slipping |
|---|---|---|
| Day 2+ | Battery storage arbitrage LP (perfect foresight vs point forecast) | Keep |
| Mid-week | RL agent (Stable-Baselines3 / Gymnasium) on battery arbitrage | **Drop first** |
| Core | SPO-trained vs MAE-trained forecaster feeding the LP | **Never drop** |
| Thursday | pandapower AC OPF on IEEE 14-bus | Keep |
| Capstone | Regret comparison: LP / RL / SPO vs MAE | Keep |
| Friday | `WEEK_07_REFLECTION.md` | Required |

Cross-cutting threads all week:

- Headline metric: regret in \$ vs perfect foresight
- Rolling/backtest evaluation over many days (not one cherry-picked day)
- README section: "what this means in dispatch/market terms"
- Git push every day

---

## Dependencies installed (full week)

Already in `requirements.txt` for later days:

- `cvxpy`, `pandas`, `matplotlib`, `numpy` — Day 1 (in use)
- `stable-baselines3`, `gymnasium` — RL (mid-week)
- `pandapower` — AC OPF (Thursday)
- `jupyter` — notebooks

---

## Suggested next step (Day 2)

Add battery storage to the LP:

1. New module: `src/dispatch/battery_arbitrage.py`
2. Decision vars: charge/discharge rate, state of charge
3. Two baselines: perfect foresight LP and LP with point forecast
4. Metric: regret in \$ vs perfect foresight over a rolling backtest

Reuse `economic_dispatch.py` patterns (cvxpy formulation, dual extraction, `DispatchResult`-style output).

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/battery-spo-rl.git`
- Push requires GitHub credentials for `mehmetertac` (not `pardus-ai`)
- Python venv at `.venv/` (gitignored)
