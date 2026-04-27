---
id: contributing
sidebar_position: 1
title: Contributing
---

# Contributing

## Development setup

```bash
git clone https://github.com/dynamicslab/KoopmanRL.git
cd KoopmanRL
uv sync --group dev
```

The `dev` group installs `pytest`, `pre-commit`, and linting tools. Activate the pre-commit hooks:

```bash
uv run pre-commit install
```

## Running the tests

```bash
uv run pytest
```

Tests live in `tests/`. Each environment has a smoke test that checks the `reset`/`step` interface and verifies that the cost function returns the correct shape.

## Code style

The repository enforces formatting and linting via pre-commit:

- **Ruff** — linting and import sorting
- **Black** — code formatting
- A GitHub Actions workflow runs `pre-commit` on every pull request

All checks must pass before merging. Run them locally with:

```bash
uv run pre-commit run --all-files
```

## Adding a new environment

1. Create `koopmanrl/environments/<name>.py` following the structure of an existing environment (e.g. `fluid_flow.py`).
2. Register the environment with `gym.envs.registration.register` inside the module.
3. Import and re-export the class in `koopmanrl/environments/__init__.py`.
4. Add a `configurations/<algo>_<name>_hparams.json` for each algorithm once you have run the optimization pipeline.
5. Add a documentation page in `docs/docs/environments/<name>.md`.

### Required interface

Every environment must implement:

| Method / attribute | Type | Description |
|--------------------|------|-------------|
| `observation_space` | `gym.spaces.Box` | State bounds |
| `action_space` | `gym.spaces.Box` | Action bounds |
| `reset(seed)` | `→ np.ndarray` | Reset to a random initial state |
| `step(action)` | `→ (obs, reward, done, info)` | Advance one timestep |
| `cost_fn(state, action)` | `→ float` | Quadratic cost for LQR/evaluation |
| `reward_fn(state, action)` | `→ float` | Negative cost |
| `vectorized_cost_fn(states, actions)` | `→ torch.Tensor` | Batched cost for SAKC critic |
| `f(state, action)` | `→ np.ndarray` | Ground-truth one-step transition |
| `continuous_A`, `continuous_B` | `np.ndarray` | Linearised dynamics (for LQR) |
| `reference_point` | `np.ndarray` | Target state $x^*$ |

## Adding a new algorithm

1. Implement the training loop in `koopmanrl/<name>.py`, following the `tap`-based argument pattern used in `soft_koopman_value_iteration.py`.
2. Add an optimization wrapper in `koopmanrl/opt_wrappers.py` if Optuna search is needed.
3. Add an optimization script `koopmanrl/<name>_optuna_opt.py`.
4. Document the algorithm in `docs/docs/algorithms/<name>.md`.

## Project layout

```
koopmanrl/              Core algorithms and environments
koopmanrl_utils/        Post-processing: trajectories, figures, GIFs
configurations/         Best-found hyperparameter JSON files
tests/                  Pytest test suite
docs/                   Docusaurus documentation site
figures/                Output figures (gitignored)
video_frames/           Output trajectory data (gitignored)
```
