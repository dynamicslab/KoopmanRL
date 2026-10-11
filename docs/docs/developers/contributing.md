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

The `dev` group installs `pre-commit`, `pytest` and `pytest-cov`; the linters themselves are installed by pre-commit. Activate the pre-commit hooks:

```bash
uv run pre-commit install
```

## Running the tests

```bash
uv run pytest
```

Tests live in `tests/`:

- `test_rl.py` runs each of the five algorithms for a short budget on all four environments, and checks that runs with the same seed repeat;
- `test_hparam_opt.py` runs the SKVI and SAKC optimization scripts with a single short trial on all four environments;
- `test_koopman_tensor_regressors.py` covers the ridge, SINDy and reduced-rank regressors;
- the remaining modules test the scripts of `koopmanrl_utils/` (launchers, result processing, validation and t-SNE scripts) and the helpers of the Snakemake workflows in `workflow/`.

The training and optimization tests start real (short) runs as subprocesses and take a while.

## Code style

The repository enforces formatting and linting via pre-commit:

- **pre-commit-hooks** — trailing whitespace, end-of-file, YAML syntax, large files, merge conflicts, debug statements
- **Ruff** — linting with `--fix` (`ruff`) and formatting (`ruff-format`), line length 120
- **isort** — import sorting with `--profile black`
- **Vulture** — dead-code detection on `koopmanrl/` with `--min-confidence 80`

In CI, `.github/workflows/lint.yml` runs `pre-commit run --all-files` on every pull request and push to `main`, and `.github/workflows/test-deploy-docs.yml` builds this documentation site on pull requests to `main`.

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
| `reset(seed=None)` | `→ np.ndarray` | Reset to a random initial state (`FluidFlow.reset(state=None, seed=None)` also accepts an initial state) |
| `step(action)` | `→ (obs, reward, done, info)` | Advance one timestep |
| `cost_fn(state, action)` | `→ float` | Quadratic cost for LQR/evaluation |
| `reward_fn(state, action)` | `→ float` | Negative cost |
| `vectorized_cost_fn(states, actions)` | `→ torch.Tensor` | Batched cost for SKVI's Bellman backup |
| `f(state, action)` | `→ np.ndarray` | Ground-truth one-step transition |
| `continuous_A`, `continuous_B` | `np.ndarray` | Linearised continuous-time dynamics (for LQR); `LinearSystem` instead has the discrete `A`, `B`, which LQR falls back to |
| `dt` | `float` | Time step (continuous-time environments; LQR and SKVI treat a missing `dt` as 1) |
| `reference_point` | `np.ndarray` | Target state $x^*$ |

## Adding a new algorithm

1. Implement the training loop in `koopmanrl/<name>.py`, following the `tap`-based argument pattern used in `soft_koopman_value_iteration.py`.
2. Add an optimization wrapper in `koopmanrl/opt_wrappers.py` if Optuna search is needed.
3. Add an optimization script `koopmanrl/<name>_optuna_opt.py`.
4. Document the algorithm in `docs/docs/algorithms/<name>.md`.

## Project layout

```
koopmanrl/              Core algorithms, environments, Koopman tensor, HPO scripts
koopmanrl_utils/        Experiment launchers, result processing, Koopman/SKVI validation scripts, movies/
configurations/         Best-found hyperparameter JSON files and settings of the reproduction scripts
workflow/               Snakemake workflows that reproduce the results of the paper
tests/                  Pytest test suite
docs/                   Docusaurus documentation site
```

Outputs are written to gitignored directories: `runs/` (TensorBoard logs), `saved_models/` (checkpoints), `figures/` and `video_frames/` (movies), `results/` and `.snakemake/` (Snakemake), and the `*_results/` folders of the `koopmanrl_utils/` scripts (e.g. `episodic_returns_results/`, `ablation_results/`, `tsne_koopman_tensor_results/`).
