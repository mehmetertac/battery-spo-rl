# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-05

### Added

- Capstone **regret harness**: MAE vs SPO+ vs RL PPO vs perfect foresight on 90-day DE-LU battery arbitrage.
- IEEE 14-bus AC OPF with load-forecast scenarios and hidden congestion detection.
- Public narrative README, CI badge, `pyproject.toml`, committed capstone figures.
- `docs/figures/cumulative_regret.gif` — animated dispatch-quality demo.
- `CHANGELOG.md` and GitHub release v1.0.0.

### Headline result

SPO loses on MAE (+0.02 EUR/MWh) and wins **EUR 9,198** cumulative regret vs MAE on the default 90-day backtest.

## [0.2.0] - 2026-10-XX (Week 07)

### Added

- Gymnasium battery env, PPO training CLI, capstone reporting pipeline.
- SPO+ linear forecaster and regret backtest vs MAE baseline.

## [0.1.0] - 2026-09-XX (Week 07 Day 1–2)

### Added

- Repo scaffold, economic dispatch LP toy with LMP duals, battery arbitrage LP, OPSD price loader.

[Unreleased]: https://github.com/mehmetertac/battery-spo-rl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mehmetertac/battery-spo-rl/releases/tag/v1.0.0
[0.2.0]: https://github.com/mehmetertac/battery-spo-rl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mehmetertac/battery-spo-rl/releases/tag/v0.1.0
