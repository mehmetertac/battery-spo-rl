# Handover — battery-spo-rl

**Repo:** https://github.com/mehmetertac/battery-spo-rl  
**Last updated:** 2026-08-27  
**Branch:** `main`

### Key commit

```
IEEE 14-bus AC OPF with load-forecast scenarios and hidden congestion
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
| **IEEE 14-bus AC OPF (`src/dispatch/ieee14_opf.py`)** | Done |
| **Load forecaster + OPF scenario harness** | Done |
| **Hidden congestion detection (forecast vs actual load)** | Done |
| Notebook `02_ieee14_opf_forecast_error.ipynb` | Done |
| IEEE 14-bus tests (`tests/test_ieee14_opf.py`) | Done |
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
│   ├── 01_economic_dispatch_toy.ipynb
│   └── 02_ieee14_opf_forecast_error.ipynb
├── src/
│   ├── dispatch/
│   │   ├── economic_dispatch.py
│   │   ├── battery_arbitrage.py
│   │   └── ieee14_opf.py           ← pandapower AC OPF + LMP extraction
│   ├── data/
│   │   └── opsd_prices.py
│   ├── forecast/
│   │   ├── price_forecaster.py       ← LightGBM (legacy baseline)
│   │   ├── linear_price_forecaster.py ← MAE / SPO linear
│   │   ├── load_forecaster.py        ← synthetic load scale + OLS
│   │   └── spo_loss.py               ← SPO+ surrogate + subgradient
│   └── eval/
│       ├── regret_backtest.py
│       ├── opf_scenario.py           ← forecast vs actual OPF CLI
│       └── __main__.py
├── tests/
│   ├── test_battery_arbitrage.py
│   ├── test_spo_forecaster.py
│   └── test_ieee14_opf.py
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

# IEEE 14-bus AC OPF scenarios (forecast vs actual load)
python -m src.eval.opf_scenario --hours 24

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

### IEEE 14-bus OPF — `src/dispatch/ieee14_opf.py`

| Symbol | Purpose |
|---|---|
| `load_ieee14_case()` | pandapower `case14()` with stored base loads |
| `prepare_congestion_case()` | Cap heaviest line to expose binding under peak load |
| `run_power_flow()` / `run_ac_opf()` | AC PF warm-start then AC OPF |
| `extract_bus_lmps()` | Bus LMPs from PYPOWER `LAM_P` shadow prices |
| `extract_line_loadings()` | `res_line.loading_percent` + topology |
| `solve_opf_at_scale(scale)` | Scale loads, run OPF, return `OpfResult` |
| `compare_binding_status()` | Hidden congestion: binds in actual, not forecast |

### Load forecaster — `src/forecast/load_forecaster.py`

| Symbol | Purpose |
|---|---|
| `build_system_load_profile()` | Synthetic multi-day load scale (Day-1 sinusoid + noise) |
| `LinearLoadForecaster` | OLS per hour on lag/calendar features |
| `stress_actual_scales()` | Amplify under-forecast peaks for OPF stress scenarios |

### OPF scenario harness — `src/eval/opf_scenario.py`

| Symbol | Purpose |
|---|---|
| `run_opf_scenarios()` | Hourly forecast vs actual OPF loop |
| `--hours`, `--profile-days`, `--output` | CLI knobs |

CSV `approach` values: `forecast_load`, `actual_load`.

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

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/battery-spo-rl.git`
- Python venv at `.venv/` (gitignored)
- 90-day backtest with SPO takes ~15–20 min (LP solve per subgradient step)
- 24-hour OPF scenario run takes ~1–2 min (48 AC OPF solves)
