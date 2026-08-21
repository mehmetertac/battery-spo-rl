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

Day-ahead battery arbitrage LP with two approaches:

1. **Perfect foresight** — oracle upper bound (regret = 0). Uses actual day-ahead prices in the LP.
2. **Point forecast** — LightGBM lag/calendar forecaster feeds the same LP; schedule is executed against actual prices.

Perfect foresight revenue minus forecast revenue is **regret in EUR** — the headline decision-quality metric.

### Run backtest

```powershell
# 90-day rolling backtest (uses last 90 complete days in OPSD cache)
python -m src.eval --days 90

# Custom window (OPSD frozen at 2020-10-06; data ends ~2020-09-30)
python -m src.eval --days 60 --start 2020-07-01 --output results/regret.csv
```

First run downloads OPSD DE-LU day-ahead prices (~120 MB) to `data/opsd/`.

### Output: `results/regret.csv`

| Column | Description |
|---|---|
| `date` | Evaluation day (Europe/Berlin) |
| `approach` | `perfect_foresight` or `point_forecast` |
| `pf_revenue` | Perfect foresight revenue (EUR) |
| `approach_revenue` | Revenue for this approach (EUR) |
| `regret` | `pf_revenue - approach_revenue` (0 for perfect foresight) |
| `cumulative_regret` | Running sum of daily regret (point forecast rows) |
| `forecast_mae` | Mean absolute price forecast error (EUR/MWh) |

### Sanity tests

```powershell
pytest tests/test_battery_arbitrage.py -q
```

## Project structure

```
battery-spo-rl/
├── data/               # Cached OPSD price CSV (gitignored)
├── notebooks/          # Exploratory walkthroughs
├── src/
│   ├── dispatch/       # LP modules (economic dispatch, battery arbitrage)
│   ├── data/           # OPSD price loader
│   ├── forecast/       # LightGBM point forecaster
│   └── eval/           # Regret backtest CLI
└── results/            # Outputs (figures, CSVs)
```

## What this means in dispatch / market terms

When load rises past the cheap plant's capacity, the next marginal MW comes from a more expensive generator — and the market price (LMP) jumps to that generator's marginal cost. Storage arbitrage exploits these price differences: charge when LMP is low, discharge when LMP is high. The optimization layer converts forecasts into dollars; that is why we measure regret, not MAE.
