# battery-spo-rl

Bridge ML and the optimization layer of power systems — **forecasting is a means, not an end.**

The headline metric for this project is **regret in dollars vs perfect foresight** (decision quality), not forecast accuracy (MAE). A forecaster that minimizes MAE can still produce suboptimal dispatch decisions; decision-focused learning (SPO loss) targets the optimization outcome directly.

## Week roadmap

- **Day 1:** LP economic dispatch toy — objective, constraints, duals as LMPs
- **Day 2+:** Battery storage arbitrage LP (perfect foresight vs point forecast)
- **Mid-week:** RL agent (Stable-Baselines3 / Gymnasium) on battery arbitrage
- **Core differentiator:** SPO-trained vs MAE-trained forecaster feeding the LP
- **Thursday:** pandapower AC OPF on IEEE 14-bus — perturb load, observe LMPs and line flows
- **Capstone:** Regret comparison across LP, RL, and SPO vs MAE pipelines

## LP / MILP mental model

Linear programs optimize a linear objective subject to linear constraints. Mixed-integer linear programs (MILPs) add binary or integer decision variables for on/off or indivisible choices.

| Problem type | Decision variables | Power systems example |
|---|---|---|
| Pure LP | Continuous (MW dispatch) | Economic dispatch, storage arbitrage |
| MILP | Continuous + binary (on/off) | Unit commitment |

**Objective:** minimize total generation cost (or maximize social welfare).

**Constraints:** meet demand every hour, respect generator capacity limits.

**Duals = shadow prices = LMPs:** the dual variable on the power-balance constraint is the marginal cost of serving one more MW at that hour. Generators "see" this price signal — it equals the marginal cost of the last (most expensive) dispatched unit. This is the foundation for locational marginal pricing (LMP) in market dispatch.

## Day 1: economic dispatch toy

Three generators with different marginal costs and capacity limits dispatch to meet hourly demand. Solve with cvxpy, inspect dispatch and LMP duals.

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

```powershell
# Module CLI — prints LMP summary and saves dispatch CSV
python -m src.dispatch

# Or open the walkthrough notebook
jupyter notebook notebooks/01_economic_dispatch_toy.ipynb
```

## Day 2: battery arbitrage + regret harness

Day-ahead battery arbitrage LP with three forecast-driven approaches:

1. **Perfect foresight** — oracle upper bound (regret = 0). Uses actual day-ahead prices in the LP.
2. **MAE linear** — OLS on lag/calendar features (minimizes forecast error).
3. **SPO linear** — same linear model and features, fine-tuned with SPO+ subgradient on the battery LP (minimizes decision regret, not MAE).

Both forecasters feed the **same** arbitrage LP; schedules are executed against actual prices. Perfect foresight revenue minus forecast revenue is **regret in EUR** — the headline decision-quality metric.

### SPO vs MAE (core idea)

Traditional forecasting minimizes MAE/MSE. **Smart Predict-then-Optimize** (Elmachtoub & Grigas) trains on the loss induced in the downstream optimizer. For battery arbitrage, overpredicting a price spike vs missing one changes charge/discharge timing asymmetrically — SPO+ targets that dollar impact via subgradients from the LP.

### Run backtest

```powershell
# 90-day rolling backtest — MAE vs SPO linear (uses last 90 complete days in OPSD cache)
python -m src.eval --days 90

# Include legacy LightGBM baseline for comparison
python -m src.eval --days 90 --include-lightgbm

# Custom window (OPSD frozen at 2020-10-06; data ends ~2020-09-30)
python -m src.eval --days 60 --start 2020-07-01 --output results/regret.csv
```

First run downloads OPSD DE-LU day-ahead prices (~120 MB) to `data/opsd/`.

### Output: `results/regret.csv`

| Column | Description |
|---|---|
| `date` | Evaluation day (Europe/Berlin) |
| `approach` | `perfect_foresight`, `mae_linear`, `spo_linear`, or `rl_ppo` (with `--include-rl`) |
| `pf_revenue` | Perfect foresight revenue (EUR) |
| `approach_revenue` | Revenue for this approach (EUR) |
| `regret` | `pf_revenue - approach_revenue` (0 for perfect foresight) |
| `cumulative_regret` | Running sum of daily regret (forecast rows) |
| `forecast_mae` | Mean absolute price forecast error (EUR/MWh) |

Headline result to look for: **SPO loses on MAE, wins on cumulative regret (EUR).**

### Sanity tests

```powershell
pytest tests/ -q
```

## Mid-week: RL agent (Gymnasium + PPO)

Model-free RL primer on the same battery arbitrage problem. The agent acts hourly (charge / hold / discharge) with observations built from SoC, hour-of-day, and causal price lags — same feature pattern as the forecasters.

### Train

```powershell
# Modest budget (~5 min CPU) — saves to models/rl_ppo.zip (gitignored)
python -m src.rl.train --timesteps 50000 --output models/rl_ppo.zip
```

Training uses 60 complete OPSD days immediately before the default 90-day eval window. Requires OPSD cache (see Day 2).

### Evaluate in regret harness

```powershell
# 90-day capstone with RL as approach rl_ppo
python -m src.eval --days 90 --include-rl

# Custom model path
python -m src.eval --days 90 --include-rl --rl-model models/rl_ppo.zip
```

### RL vs forecast LP (interpret carefully)

| Approach | Pipeline | Information at decision time |
|---|---|---|
| MAE / SPO | Forecast 24h → LP → execute on actuals | Day-start forecast only |
| RL PPO | Sequential policy rollout | History + **realized within-day prices** (lags) |
| Perfect foresight | Oracle LP on actuals | Full-day actuals |

RL revenue is evaluated with `realized_revenue()` like the LP approaches. See [WEEK_07_REFLECTION.md](WEEK_07_REFLECTION.md) for the honest capstone read: RL beats forecast LPs here partly because of richer intra-day information, but still trails perfect foresight by ~EUR 267k over 90 days.

## Capstone: four-way showdown

Compare all four approaches on the same 90-day backtest (Jul–Sep 2020):

1. **LP + perfect foresight** — oracle upper bound
2. **LP + MAE forecast** — OLS on lag/calendar features
3. **RL agent (PPO)** — sequential policy with intra-day price lags
4. **LP + SPO forecast** — MAE init + SPO+ bias fine-tuning

### Run

```powershell
# Train RL agent first (~5 min)
python -m src.rl.train --timesteps 50000 --output models/rl_ppo.zip

# Full capstone: backtest + master table + regime + figures (~35–45 min)
python -m src.eval.capstone --days 90 --include-rl

# Skip 3×3 battery sensitivity grid (saves ~4–6 hours)
python -m src.eval.capstone --days 90 --include-rl --skip-sensitivity

# Replot from existing results/regret.csv
python -m src.eval.capstone --skip-backtest --skip-sensitivity
```

### Output: `results/`

| File | Description |
|---|---|
| `regret.csv` | Daily rows for all four approaches |
| `master_table.csv` | Total revenue, dollar regret, regret % of perfect foresight, avg MAE |
| `regime_robustness.csv` | Volatile vs calm week aggregates |
| `battery_sensitivity.csv` | 3×3 LP grid (efficiency × energy capacity) |
| `cumulative_regret.png` | **Money plot** — cumulative regret over backtest |
| `forecast_mae_vs_regret.png` | Daily forecast MAE vs daily regret scatter |
| `regime_robustness.png` | Regret by market regime |
| `battery_sensitivity.png` | SPO regret advantage heatmap |

Headline result: **SPO loses on MAE, wins on cumulative regret (EUR) vs MAE.** The scatter plot shows forecast accuracy and decision quality are weakly coupled — SPO's curve dips below MAE's on the money plot despite worse average MAE.

## Thursday: IEEE 14-bus AC OPF + load forecast error

pandapower AC OPF on the IEEE 14-bus case — connects Day-1 cvxpy duals to **locational** LMPs and line congestion.

1. Load `case14()`, run power flow, then AC OPF with polynomial generator costs
2. Extract bus LMPs from OPF shadow prices (`LAM_P`) and line `loading_percent`
3. Scale all bus loads with a synthetic day-ahead **load forecast vs actual** profile (lag + calendar OLS, same feature pattern as price forecasting)
4. Compare forecast-load OPF vs actual-load OPF — including **hidden congestion** (line binds in actual case but not forecast case)

### Run

```powershell
# Single-case OPF + LMP CSV
python -m src.dispatch.ieee14_opf

# 24-hour forecast vs actual scenario backtest (~1–2 min)
python -m src.eval.opf_scenario --hours 24

# Narrative notebook with plots
jupyter notebook notebooks/02_ieee14_opf_forecast_error.ipynb
```

### Output

| File | Description |
|---|---|
| `results/ieee14_bus_lmps.csv` | Bus LMPs from baseline OPF |
| `results/ieee14_line_loadings.csv` | Line loadings from baseline OPF |
| `results/ieee14_opf_scenarios.csv` | Hourly bus LMPs under `forecast_load` / `actual_load` |
| `results/ieee14_opf_scenarios_summary.csv` | Hidden congestion flags and LMP spread per hour |

## Project structure

```
battery-spo-rl/
├── data/               # Cached OPSD price CSV (gitignored)
├── models/             # Trained SB3 checkpoints (gitignored)
├── notebooks/          # Exploratory walkthroughs
├── src/
│   ├── dispatch/       # LP modules + IEEE 14-bus AC OPF (pandapower)
│   ├── data/           # OPSD price loader
│   ├── forecast/       # LightGBM + linear MAE/SPO forecasters
│   ├── rl/             # Gymnasium env + PPO training
│   └── eval/           # Regret backtest + capstone reporting CLI
└── results/            # Outputs (figures, CSVs)
```

## What this means in dispatch / market terms

When load rises past the cheap plant's capacity, the next marginal MW comes from a more expensive generator — and the market price (LMP) jumps to that generator's marginal cost. Storage arbitrage exploits these price differences: charge when LMP is low, discharge when LMP is high. The optimization layer converts forecasts into dollars; that is why we measure regret, not MAE.
