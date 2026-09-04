# We improved the forecast's MAE and lost money — why decision-focused learning matters in energy markets

*Seed for a Week 12 long-form article. Repo: [battery-spo-rl](https://github.com/mehmetertac/battery-spo-rl)*

---

On a 90-day backtest of German day-ahead battery arbitrage (Jul–Sep 2020), we trained two price forecasters on the same linear model and the same lag/calendar features. One minimized **mean absolute error (MAE)**. The other used **Smart Predict-then-Optimize (SPO+)** — fine-tuning on the loss induced in the downstream battery LP, not on forecast accuracy.

The MAE forecaster won on accuracy: **9.38 vs 9.40 EUR/MWh**.

The SPO forecaster won on money: **EUR 9,198 less cumulative regret** vs perfect foresight over 90 days.

We improved the forecast's MAE and would have lost money if we'd stopped at the accuracy metric.

## Predict-then-optimize vs train-through-the-LP

Energy storage operators don't submit forecasts to a leaderboard. They submit **dispatch schedules** derived from forecasts through an optimizer. The relevant question is not "how close was the forecast?" but "how much revenue did the schedule earn vs what was possible?"

That gap is **regret**:

```
regret = perfect_foresight_revenue − approach_revenue
```

Perfect foresight is unattainable — it sets the ceiling. MAE and SPO measure how much **forecast error costs you in dollars** when fed through the same arbitrage LP and executed against actual prices.

Traditional forecasting treats every EUR/MWh error as symmetric. Battery arbitrage is asymmetric: overpredicting a price spike vs missing one changes when you charge and discharge. SPO+ (Elmachtoub & Grigas) trains on the LP's response to forecast perturbations — the envelope subgradient reads off optimal net export as sensitivity. Same model architecture, same features, different objective → different dispatch → different dollars.

In our backtest, SPO's advantage was strongest in **volatile weeks** (−EUR 23,711 vs MAE). In calm weeks, MAE won (+EUR 14,514). The 90-day total still favored SPO — but the regime split is the story for practitioners: the value of decision-focused learning shows up when price swings make timing errors expensive.

## How operators are actually judged

Real storage traders and asset optimizers are graded on:

- **Realized P&L** vs budget, forward curve, or peer fleet
- **Opportunity cost** vs perfect information (regret / value of perfect foresight)
- **Benchmark tracking error** — vs a simple heuristic or the firm's official optimizer

They are not graded on forecast MAE.

Forecast quality is a diagnostic. **Decision regret is the scoreboard.** SPO aligns training with that scoreboard for the predict-then-optimize path — the pipeline that day-ahead markets actually use: forecast → optimize → submit schedule.

## What this repo demonstrates

[battery-spo-rl](https://github.com/mehmetertac/battery-spo-rl) is a end-to-end capstone:

1. **Battery arbitrage LP** — cvxpy dispatch with LMP duals
2. **MAE vs SPO linear forecasters** — same features, different training objectives
3. **Four-way regret comparison** — perfect foresight, MAE, SPO, and RL (PPO)
4. **IEEE 14-bus AC OPF** — load forecast error shifts locational LMPs and can hide congestion

The RL agent beats both forecast LPs on cumulative regret — but it sees **intra-day realized prices** via lag features, while MAE/SPO commit a full-day schedule at day start. For day-ahead market submission, SPO vs MAE is the honest production comparison. RL is a useful primer on information timing; SPO + LP is the senior-signal result.

## Reproduce it

```powershell
pip install -r requirements.txt
python -m src.rl.train --timesteps 50000
python -m src.eval.capstone --days 90 --include-rl --skip-sensitivity
```

~45 minutes on CPU. Full instructions in the [README](https://github.com/mehmetertac/battery-spo-rl#reproduce).

---

*This post is the outline for a longer Week 12 piece on decision-focused learning in energy markets — SPO+ theory, the MAE–regret decoupling, regime robustness, and the OPF congestion story.*
