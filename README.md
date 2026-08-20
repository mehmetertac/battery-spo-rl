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

## Project structure

```
battery-spo-rl/
├── notebooks/          # Exploratory walkthroughs
├── src/dispatch/       # Reusable optimization modules
└── results/            # Outputs (figures, CSVs)
```

## What this means in dispatch / market terms

When load rises past the cheap plant's capacity, the next marginal MW comes from a more expensive generator — and the market price (LMP) jumps to that generator's marginal cost. Storage arbitrage exploits these price differences: charge when LMP is low, discharge when LMP is high. The optimization layer converts forecasts into dollars; that is why we measure regret, not MAE.
