# AGENT.md — rules for AI agents

Guidance for agents working in **battery-spo-rl**. Read this file first, then follow the linked docs for domain context.

---

## Documentation map

| Doc | Purpose |
|---|---|
| [README.md](README.md) | Project goal, setup, Day 1/Day 2 run commands, regret metric |
| [handover.md](handover.md) | Current status, repo layout, module API, week roadmap |
| [pytest.ini](pytest.ini) | Test discovery (`tests/`, `pythonpath = .`) |
| [requirements.txt](requirements.txt) | Dependencies |
| [src/dispatch/](src/dispatch/) | LP modules + IEEE 14-bus AC OPF (pandapower) |
| [src/forecast/](src/forecast/) | Price + load forecasters, SPO loss |
| [src/eval/](src/eval/) | Regret backtest + OPF scenario CLI |
| [tests/](tests/) | Unit tests (synthetic prices; no OPSD in CI) |
| [notebooks/](notebooks/) | Exploratory walkthroughs |

Headline metric for this repo: **regret in EUR vs perfect foresight** — not forecast MAE.

---

## Rules

### 1. File size limit

- **No file should exceed 1,000 lines.**
- If a file approaches or exceeds that limit, **stop and suggest a refactor** before adding more code (split modules, extract helpers, move tests/notebooks out).
- Prefer small, focused modules over monoliths.

### 2. Documentation before every push

- **Update documentation before every push** to the repository.
- At minimum, check whether these need updates for your change:
  - [README.md](README.md) — run commands, structure, user-facing behavior
  - [handover.md](handover.md) — done/next steps, API table, artifacts
- If you add CLI flags, modules, or outputs, document them in the relevant file above.

### 2a. Update handover.md on every push (required)

- **[handover.md](handover.md) must be updated before every push**, even for small changes.
- Treat it as the live project status doc for the next person (or agent).
- On each push, at minimum refresh:
  - **Last updated** date at the top
  - **What is done** table (add/mark items)
  - **Repo layout** if files or directories changed
  - **Core module API** if public symbols, CLI flags, or defaults changed
  - **Suggested next step** if priorities shifted
  - **Key commit** hash and one-line summary for the work being pushed
- If nothing functional changed (e.g. typo-only), still bump **Last updated** and note “no functional change.”
- Do not push without reviewing [handover.md](handover.md) — stale handover is a blocker.

### 3. Tests — always, at least minimal

- **Always create at least minimal unit tests**, even for small changes.
- Add **integration** tests when wiring multiple modules (e.g. loader → forecaster → LP → regret).
- Add **functional** tests when the project supports them (CLI smoke tests, end-to-end with tiny fixtures).
- Existing pattern: [tests/test_battery_arbitrage.py](tests/test_battery_arbitrage.py) uses synthetic data so CI does not depend on OPSD downloads.
- New optimization or forecast logic should get numeric/assertion checks, not only “runs without error.”

### 4. Run tests before commit or push

- **Run the test suite before commit or push**, depending on what the project uses:
  ```powershell
  pytest tests/ -q
  ```
- For changes touching the backtest or CLI, also smoke-test:
  ```powershell
  python -m src.eval --days 3
  ```
  (Use a small `--days` value; full 90-day runs are for validation, not every commit.)

**Git hooks:** This repo does **not** yet have pre-commit or pre-push hooks. When you touch workflow or the user asks for automation, **create hooks** (e.g. `.pre-commit-config.yaml` with `pytest` and optional format checks) and document install steps in [README.md](README.md).

### 5. Keep reading in-repo docs

- Do not guess API or roadmap from memory — use [handover.md](handover.md) for status and [README.md](README.md) for how to run.
- Match existing conventions in [src/dispatch/economic_dispatch.py](src/dispatch/economic_dispatch.py) (cvxpy, dataclasses, solver fallback) when extending dispatch code.
- Regret evaluation must remain comparable across approaches (perfect foresight upper bound vs forecast-driven LP executed on actual prices).

---

## Quick checklist (before push)

- [ ] No file > 1,000 lines (or refactor proposed)
- [ ] [README.md](README.md) updated if behavior or layout changed
- [ ] [handover.md](handover.md) updated (required on every push — date, status, API, key commit)
- [ ] New/changed logic has tests in [tests/](tests/)
- [ ] `pytest tests/ -q` passes
- [ ] Relevant smoke command run if CLI or backtest changed
