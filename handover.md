# Handover — battery-spo-rl

**Repo:** https://github.com/mehmetertac/battery-spo-rl  
**Last updated:** 2026-08-26  
**Branch:** `main` (up to date with remote)

### Key commit

```
SPO+ linear forecaster vs MAE baseline on battery arbitrage LP
```

---

## Project goal

Bridge ML and the optimization layer of power systems. The headline metric is **regret in dollars vs perfect foresight** (decision quality), not forecast accuracy (MAE). Forecasting is a means, not an end.

This is the most senior-signal artifact of the 12-week master plan. If the week slips, drop RL — never drop SPO.

Agent workflow rules: [AGENT.md](AGENT.md) (includes mandatory handover update before every push).

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
| **Linear MAE forecaster (OLS on same features)** | Done |
| **SPO+ linear forecaster (bias fine-tune from MAE init)** | Done |
| **Regret harness: MAE vs SPO vs perfect foresight** | Done |
| SPO+ loss + subgradient tests (`tests/test_spo_forecaster.py`) | Done |
| Synthetic-price sanity tests (`tests/test_battery_arbitrage.py`) | Done |
| Agent workflow rules ([AGENT.md](AGENT.md)) | Done |
| Test config ([pytest.ini](pytest.ini)) | Done |

---

## Headline result (90-day backtest, Jul–Sep 2020)

Run `python -m src.eval --days 90` after OPSD cache is present. Default SPO tuning (`--spo-lr 5 --spo-epochs 5`), Jul–Sep 2020:

| Approach | Total revenue (EUR) | Cumulative regret (EUR) | Avg MAE (EUR/MWh) |
|---|---|---|---|
| Perfect foresight | 876,849 | 0 | — |
| MAE linear | 113,171 | 763,678 | 9.38 |
| SPO linear | 122,369 | 754,481 | 9.40 |

**SPO loses on MAE (+0.02 EUR/MWh), wins on dollars (−EUR 9,198 regret, +EUR 9,198 revenue).**

---

## Repo layout

```
battery-spo-rl/
├── README.md
├── AGENT.md
├── handover.md
├── pytest.ini
├── requirements.txt
├── data/                    ← cached OPSD CSV (gitignored)
├── notebooks/
│   └── 01_economic_dispatch_toy.ipynb
├── src/
│   ├── dispatch/
│   │   ├── economic_dispatch.py
│   │   └── battery_arbitrage.py
│   ├── data/
│   │   └── opsd_prices.py
│   ├── forecast/
│   │   ├── price_forecaster.py       ← LightGBM (legacy baseline)
│   │   ├── linear_price_forecaster.py ← MAE / SPO linear
│   │   └── spo_loss.py               ← SPO+ surrogate + subgradient
│   └── eval/
│       ├── regret_backtest.py
│       └── __main__.py
├── tests/
│   ├── test_battery_arbitrage.py
│   └── test_spo_forecaster.py
└── results/
```

---

## How to run

```powershell
cd battery-spo-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 90-day MAE vs SPO regret backtest
python -m src.eval --days 90

# Optional LightGBM baseline
python -m src.eval --days 90 --include-lightgbm

# Sanity tests
pytest tests/ -q
```

---

## Core module API

### SPO+ loss — `src/forecast/spo_loss.py`

| Symbol | Purpose |
|---|---|
| `net_export_vector()` | Optimal hourly net export (discharge − charge) from LP |
| `spo_plus_loss()` | SPO+ surrogate: max_z (p_true − p_hat)^T z |
| `spo_plus_price_subgradient()` | Envelope subgradient: −z*(p_true − p_hat) |
| `decision_regret()` | True dollar regret vs perfect foresight |

### Linear forecaster — `src/forecast/linear_price_forecaster.py`

| Symbol | Purpose |
|---|---|
| `LinearPriceForecaster(objective='mae')` | OLS per hour-of-day on lag/calendar features |
| `LinearPriceForecaster(objective='spo')` | MAE init + SPO+ bias subgradient fine-tuning |
| `spo_epochs`, `learning_rate`, `max_spo_training_days` | SPO tuning knobs |

SPO training: initialize from MAE weights, then run normalized SPO+ subgradient steps on complete training days (default last 20 days, 5 epochs). Predictions clipped to training price range ± margin.

### Regret harness — `src/eval/regret_backtest.py`

| Symbol | Purpose |
|---|---|
| `run_regret_backtest()` | Rolling perfect foresight + MAE linear + SPO linear |
| `--include-lightgbm` | Add legacy LightGBM `point_forecast` rows |
| `--spo-lr`, `--spo-epochs` | SPO+ hyperparameters |

CSV `approach` values: `perfect_foresight`, `mae_linear`, `spo_linear`, optionally `point_forecast`.

---

## Suggested next step

1. RL agent on battery arbitrage (drop if week slips)
2. Extend capstone CSV with RL rows
3. Optional: cvxpylayers end-to-end gradient (PyTorch) for stronger SPO gains
4. pandapower AC OPF on IEEE 14-bus (Thursday)

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/battery-spo-rl.git`
- Python venv at `.venv/` (gitignored)
- 90-day backtest with SPO takes ~15–20 min (LP solve per subgradient step)
