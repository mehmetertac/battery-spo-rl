# Handover — battery-spo-rl

**Repo:** https://github.com/mehmetertac/battery-spo-rl  
**Last updated:** 2026-08-21  
**Branch:** `main` (up to date with remote)

---

## Project goal

Bridge ML and the optimization layer of power systems. The headline metric is **regret in dollars vs perfect foresight** (decision quality), not forecast accuracy (MAE). Forecasting is a means, not an end.

This is the most senior-signal artifact of the 12-week master plan. If the week slips, drop RL — never drop SPO.

---

## What is done

| Item | Status |
|---|---|
| Repo scaffold (`src/`, `notebooks/`, `results/`, `requirements.txt`) | Done |
| LP economic dispatch toy (3 generators, 24h demand) | Done |
| LMP extraction from cvxpy power-balance duals | Done |
| LMP verification (dual = marginal generator cost) | Done — 24/24 hours pass |
| Walkthrough notebook with plots | Done |
| README with LP/MILP mental model and dispatch framing | Done |
| Battery arbitrage LP (`src/dispatch/battery_arbitrage.py`) | Done |
| OPSD DE-LU day-ahead price loader | Done |
| LightGBM point forecaster (lags + calendar) | Done |
| Regret harness + `python -m src.eval` CLI | Done |
| Synthetic-price sanity tests (`tests/test_battery_arbitrage.py`) | Done |

---

## Repo layout

```
battery-spo-rl/
├── README.md
├── handover.md              ← this file
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/                    ← cached OPSD CSV (gitignored)
├── notebooks/
│   └── 01_economic_dispatch_toy.ipynb
├── src/
│   ├── dispatch/
│   │   ├── __init__.py
│   │   ├── __main__.py      ← CLI entry: python -m src.dispatch
│   │   ├── economic_dispatch.py
│   │   └── battery_arbitrage.py
│   ├── data/
│   │   └── opsd_prices.py
│   ├── forecast/
│   │   └── price_forecaster.py
│   └── eval/
│       ├── regret_backtest.py
│       └── __main__.py      ← CLI entry: python -m src.eval
├── tests/
│   └── test_battery_arbitrage.py
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

# Day 1 — economic dispatch
python -m src.dispatch

# Day 2 — 90-day regret backtest
python -m src.eval --days 90

# Sanity tests
pytest tests/test_battery_arbitrage.py -q

# Notebook walkthrough
jupyter notebook notebooks/01_economic_dispatch_toy.ipynb
```

Expected Day 2 output (values vary with OPSD window):

```
OPSD DE-LU day-ahead prices available from YYYY-MM-DD to YYYY-MM-DD.
Evaluation days: 90 (YYYY-MM-DD to YYYY-MM-DD)
Perfect foresight total revenue: EUR ...
Point forecast total revenue:    EUR ...
Total cumulative regret:         EUR ...
Average daily regret:            EUR ...
Saved results to results\regret.csv
```

Generated artifacts (local, not committed):

- `results/dispatch_day1.csv`
- `results/regret.csv`
- `data/opsd/time_series_60min_singleindex.csv`

---

## Core module API

### Economic dispatch — `src/dispatch/economic_dispatch.py`

| Symbol | Purpose |
|---|---|
| `Generator` | Dataclass: name, marginal_cost ($/MWh), pmax (MW) |
| `DispatchResult` | dispatch DataFrame, total_cost, lmps Series, status |
| `default_generators()` | 3-gen merit-order stack ($20, $35, $50/MWh) |
| `default_demand()` | 24h sinusoidal profile (~120–200 MW) |
| `solve_economic_dispatch()` | cvxpy LP solver; returns dispatch + LMPs |
| `verify_lmps()` | Assert duals match marginal generator cost |
| `marginal_generator_cost()` | Expected LMP from dispatch (for verification) |

### Battery arbitrage — `src/dispatch/battery_arbitrage.py`

| Symbol | Purpose |
|---|---|
| `BatteryConfig` | Dataclass: power, energy, efficiency, SOC bounds |
| `ArbitrageResult` | schedule DataFrame, revenue, status |
| `solve_battery_arbitrage()` | cvxpy LP; maximize price × (discharge − charge) |
| `realized_revenue()` | Execute fixed schedule against actual prices |

Default battery: 100 MW / 400 MWh, 90% round-trip efficiency, 50% initial SOC, terminal SOC ≥ initial.

### Price data — `src/data/opsd_prices.py`

| Symbol | Purpose |
|---|---|
| `load_de_day_ahead()` | DE-LU day-ahead prices (Europe/Berlin), date slice |
| `extract_day_prices()` | 24 hourly prices for one calendar day |
| `available_day_range()` | First/last complete days in OPSD cache |

OPSD time series frozen at 2020-10-06; backtest uses last N complete days in that dataset.

### Forecaster — `src/forecast/price_forecaster.py`

| Symbol | Purpose |
|---|---|
| `PriceForecaster` | 24 LightGBM models (one per hour-of-day) |
| `build_feature_matrix()` | Lags (1,2,3,24,48,168) + calendar features |

### Regret harness — `src/eval/regret_backtest.py`

| Symbol | Purpose |
|---|---|
| `run_regret_backtest()` | Rolling day-by-day perfect foresight vs forecast LP |
| `python -m src.eval` | CLI: `--days`, `--start`, `--output`, battery overrides |

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
| Day 2 | Battery storage arbitrage LP + regret harness | Done |
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
- `lightgbm`, `scikit-learn`, `pytest` — Day 2 (in use)
- `stable-baselines3`, `gymnasium` — RL (mid-week)
- `pandapower` — AC OPF (Thursday)
- `jupyter` — notebooks

---

## Suggested next step (mid-week)

1. RL agent on battery arbitrage (Gymnasium env wrapping `solve_battery_arbitrage` or heuristic dispatch)
2. SPO-trained vs MAE-trained forecaster feeding the same LP
3. Extend `results/regret.csv` with RL and SPO rows for capstone comparison

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/battery-spo-rl.git`
- Push requires GitHub credentials for `mehmetertac` (not `pardus-ai`)
- Python venv at `.venv/` (gitignored)
