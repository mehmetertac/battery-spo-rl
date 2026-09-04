# Handover — battery-spo-rl

**Repo:** https://github.com/mehmetertac/battery-spo-rl  
**Last updated:** 2026-09-04  
**Branch:** `main`

### Key commit

```
README finalization: narrative-first landing page, docs/figures, Week 12 seed post
```

---

## Project goal

Bridge ML and the optimization layer of power systems. The headline metric is **regret in dollars vs perfect foresight** (decision quality), not forecast accuracy (MAE). Forecasting is a means, not an end.

This is the most senior-signal artifact of the 12-week master plan. If the week slips, drop RL — never drop SPO.

Agent workflow rules: [AGENT.md](AGENT.md) (includes mandatory handover update before every push).

Week 07 reflection: [WEEK_07_REFLECTION.md](WEEK_07_REFLECTION.md)

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
| **README finalization (narrative-first, pipeline diagram, embedded figures)** | Done |
| **Public write-up (`docs/mae-lost-money.md`, Week 12 article seed)** | Done |
| **Committed capstone + OPF figures (`docs/figures/`)** | Done |
| **OPF figure CLI (`src/eval/opf_figures.py`, `--save-figures`)** | Done |
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
| **Gymnasium battery env (`src/rl/battery_env.py`)** | Done |
| **PPO training CLI (`src/rl/train.py`)** | Done |
| **Regret harness RL integration (`--include-rl`, `rl_ppo`)** | Done |
| RL env tests (`tests/test_battery_env.py`) | Done |
| Week 07 reflection (`WEEK_07_REFLECTION.md`) | Done |
| **Capstone reporting (`src/eval/capstone*.py`)** | Done |
| **Master table + regret figures in `results/`** | Done |
| **Regime robustness (volatile vs calm weeks)** | Done |
| **Battery sensitivity 3×3 grid (LP-only)** | Done |
| Capstone tests (`tests/test_capstone_summary.py`) | Done |
| Agent workflow rules ([AGENT.md](AGENT.md)) | Done |
| Test config ([pytest.ini](pytest.ini)) | Done |

---

## Headline result (90-day backtest, Jul–Sep 2020)

Run `python -m src.eval --days 90 --include-rl` after training (`python -m src.rl.train`). Default SPO tuning (`--spo-lr 5 --spo-epochs 5`), PPO 50k timesteps:

| Approach | Total revenue (EUR) | Cumulative regret (EUR) | Avg MAE (EUR/MWh) |
|---|---|---|---|
| Perfect foresight | 876,849 | 0 | — |
| MAE linear | 113,171 | 763,678 | 9.38 |
| **RL PPO** | **610,037** | **266,813** | n/a |
| SPO linear | 122,369 | 754,481 | 9.40 |

**SPO loses on MAE (+0.02 EUR/MWh), wins on dollars vs MAE (−EUR 9,198 regret).**

**Regime split:** SPO wins on dollars in **volatile** weeks (−EUR 23,711 vs MAE); MAE wins in calm weeks (+EUR 14,514). Headline 90-day total still favors SPO.

**RL beats forecast LPs but trails perfect foresight** — see [WEEK_07_REFLECTION.md](WEEK_07_REFLECTION.md) for the information-structure caveat (RL sees intra-day realized prices via lags; forecast LPs commit at day start).

### Capstone artifacts (`results/`)

| File | Description |
|---|---|
| `regret.csv` | Daily rows for all four approaches |
| `master_table.csv` | Total revenue, dollar regret, regret % of PF, avg MAE |
| `regime_robustness.csv` | Aggregates by volatile vs calm ISO weeks |
| `battery_sensitivity.csv` | 3×3 LP grid (efficiency × energy); SPO vs MAE regret |
| `cumulative_regret.png` | Money plot — cumulative regret over backtest |
| `forecast_mae_vs_regret.png` | Scatter: daily MAE vs daily regret |
| `regime_robustness.png` | Bar chart by market regime |
| `battery_sensitivity.png` | Heatmap of SPO regret advantage vs MAE |

### Public artifacts (`docs/`)

| File | Description |
|---|---|
| `docs/figures/cumulative_regret.png` | Money plot (committed for GitHub README) |
| `docs/figures/forecast_mae_vs_regret.png` | MAE vs regret scatter |
| `docs/figures/ieee14_lmp.png` | Baseline bus LMPs + line loadings |
| `docs/figures/ieee14_scenario_comparison.png` | LMP shift + hidden congestion |
| `docs/mae-lost-money.md` | Short public post — Week 12 long-form seed |

---

## Repo layout

```
battery-spo-rl/
├── README.md
├── AGENT.md
├── handover.md
├── WEEK_07_REFLECTION.md
├── docs/
│   ├── figures/             ← committed capstone + OPF PNGs (GitHub-rendered)
│   └── mae-lost-money.md    ← Week 12 article seed
├── pytest.ini
├── requirements.txt
├── data/                    ← cached OPSD CSV (gitignored)
├── models/                  ← SB3 checkpoints (gitignored)
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
│   ├── rl/
│   │   ├── battery_env.py            ← Gymnasium env
│   │   └── train.py                    ← PPO training CLI
│   └── eval/
│       ├── regret_backtest.py
│       ├── capstone.py               ← four-way orchestrator CLI
│       ├── capstone_summary.py       ← master table, regime, sensitivity
│       ├── capstone_figures.py       ← slide-ready plots
│       ├── opf_figures.py            ← IEEE 14-bus OPF PNG CLI
│       ├── opf_scenario.py           ← forecast vs actual OPF CLI
│       └── __main__.py
├── tests/
│   ├── test_battery_arbitrage.py
│   ├── test_battery_env.py
│   ├── test_capstone_summary.py
│   ├── test_spo_forecaster.py
│   └── test_ieee14_opf.py
└── results/                          ← regret CSV, tables, capstone figures
```

---

## How to run

```powershell
cd battery-spo-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train PPO agent (~5 min CPU for 50k steps)
python -m src.rl.train --timesteps 50000

# Four-way capstone: backtest + tables + figures (~35–45 min)
python -m src.eval.capstone --days 90 --include-rl

# Replot from existing regret.csv (skip backtest and sensitivity grid)
python -m src.eval.capstone --days 90 --skip-backtest --skip-sensitivity

# Backtest only (legacy CLI)
python -m src.eval --days 90 --include-rl

# 90-day MAE vs SPO only (no RL)
python -m src.eval --days 90

# IEEE 14-bus AC OPF scenarios (forecast vs actual load)
python -m src.eval.opf_scenario --hours 24

# OPF figures only (~1 min, no OPSD)
python -m src.eval.opf_scenario --hours 24 --save-figures

# Sanity tests
pytest tests/ -q
```

**Note:** Full capstone with 3×3 battery sensitivity adds ~4–6 hours (LP-only reruns). Use `--skip-sensitivity` for faster iteration.

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

### RL — `src/rl/`

| Symbol | Purpose |
|---|---|
| `BatteryArbitrageEnv` | Gymnasium env: discrete charge/hold/discharge, causal lag obs |
| `evaluate_rl_day()` | Deterministic SB3 rollout → schedule + `realized_revenue()` |
| `schedule_from_episode()` | Episode log → DataFrame for harness |
| `python -m src.rl.train` | PPO training; `--timesteps`, `--output`, `--train-days` |

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
| `--save-figures` | Write `ieee14_lmp.png` + `ieee14_scenario_comparison.png` |

CSV `approach` values: `forecast_load`, `actual_load`.

### OPF figures — `src/eval/opf_figures.py`

| Symbol | Purpose |
|---|---|
| `plot_baseline_opf()` | Bus LMPs + top line loadings at unit scale |
| `plot_forecast_vs_actual()` | LMP shift + hidden congestion at peak hour |
| `generate_opf_figures()` | Write both PNGs to a results directory |

### Regret harness — `src/eval/regret_backtest.py`

| Symbol | Purpose |
|---|---|
| `run_regret_backtest()` | Rolling perfect foresight + MAE + RL + SPO |
| `--include-rl`, `--rl-model` | Add trained PPO `rl_ppo` rows |
| `--include-lightgbm` | Add legacy LightGBM `point_forecast` rows |
| `--spo-lr`, `--spo-epochs` | SPO+ hyperparameters |

CSV `approach` values: `perfect_foresight`, `mae_linear`, `rl_ppo`, `spo_linear`, optionally `point_forecast`.

### Capstone orchestrator — `src/eval/capstone.py`

| Symbol / flag | Purpose |
|---|---|
| `python -m src.eval.capstone` | Backtest + master table + regime + figures (+ sensitivity grid) |
| `--include-rl`, `--rl-model` | Add trained PPO to four-way comparison |
| `--skip-backtest` | Load existing `results/regret.csv` |
| `--skip-sensitivity` | Skip 3×3 battery grid (~4–6 hours) |
| `--results-dir` | Output directory (default `results/`) |

### Capstone summary — `src/eval/capstone_summary.py`

| Symbol | Purpose |
|---|---|
| `build_master_table()` | Total revenue, regret, regret % of PF, avg MAE |
| `build_regime_robustness()` | Volatile vs calm week aggregates |
| `run_battery_sensitivity_grid()` | LP-only 3×3 efficiency × energy reruns |

### Capstone figures — `src/eval/capstone_figures.py`

| Symbol | Purpose |
|---|---|
| `plot_cumulative_regret()` | Money plot — all four approaches |
| `plot_forecast_mae_vs_regret()` | Accuracy ≠ decision quality scatter |
| `plot_regime_robustness()` | Regime bar chart |
| `plot_battery_sensitivity()` | SPO advantage heatmap |

---

## Suggested next step

1. **Week 12 long-form article** — expand [docs/mae-lost-money.md](docs/mae-lost-money.md) (SPO+ theory, MAE–regret decoupling, regime robustness, OPF congestion)
2. Optional: cvxpylayers end-to-end gradient (PyTorch) for stronger SPO gains
3. Optional: fair RL baseline with day-start-only observations (match day-ahead LP information)

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/battery-spo-rl.git`
- Python venv at `.venv/` (gitignored)
- 90-day capstone with SPO + RL takes ~35–45 min (LP per SPO subgradient + PPO rollouts)
- Full capstone + 3×3 battery sensitivity takes ~4–6 hours additional (LP-only reruns)
- PPO training 50k steps takes ~5–7 min CPU
- 24-hour OPF scenario run takes ~1–2 min (48 AC OPF solves)
