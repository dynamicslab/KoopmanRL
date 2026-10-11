# Koopman Tensor Guide

Subdirectory containing the logic for the Koopman tensor generation: the tensor classes `torch_tensor.py` and `numpy_tensor.py`, and `regressors.py`, the regression shared by them and by the copies of the class inlined in `soft_koopman_value_iteration.py` and `soft_actor_koopman_critic.py` (see `koopmanrl/AGENTS.md`).

## Running Core Functionality

While the tensors themselves are not executable, the generation routine can be run on the individual environments with

```bash
uv run -m koopmanrl.koopman_tensor.generate_tensor
```

This will per default generate a Koopman tensor for the linear system. The options here are

* `LinearSystem-v0`
* `FluidFlow-v0`
* `Lorenz-v0`
* `DoubleWell-v0`

which for the example of the `FluidFlow-v0` would be executed as

```bash
uv run -m koopmanrl.koopman_tensor.generate_tensor --env_id FluidFlow-v0
```

## Directory Structure

```
koopman_tensor/
├── observables/        # Subdirectory containing the NumPy, and Torch observables
├── __init__.py         # Initialization of the subdirectory
├── AGENTS.md           # This file
├── generate_tensor.py  # Generating the Koopman tensor for a specific environment
├── numpy_tensor.py     # Koopman tensor implementation in pure NumPy
├── regressors.py       # Ridge, SINDy and reduced-rank regression of the tensor (scaled increment)
├── torch_tensor.py     # Koopman tensor implementation in PyTorch
└── utils.py            # Utilities for loading and storing Koopman tensors
```

## Regressors

`generate_tensor`, the tensor classes and SKVI/SAKC take `--regressor {ols,ridge,sindy,rrr}`; `generate_tensor` also takes `--rank`, `--penalty` and `--threshold` (plus `--save_model` and `--animate`).

* `ols` is least squares; with the torch class it is the path behind every result of the paper.
* `ridge`, `sindy` and `rrr` regress the increment φ(x′) − φ(x) on standardised columns, so the penalty, threshold or rank acts on the departure from persistence. When the hyperparameter is not given, it is chosen from a short grid (`PENALTIES`, `THRESHOLDS`, `RANK_FRACTIONS` in `regressors.py`) by fitting on the first 80% of the transitions and scoring on the last 20%, then refitted on all of them; the value used is stored as `tensor.regressor_parameter`.
* `ols` in `torch_tensor.py` keeps torch's default least-squares driver on purpose: `gelsd`, which SKVI and SAKC use, is less accurate on the linear system, where `tests/test_koopman_prediction_validation.py` bounds the one-step error of the exact model.
* Tests: `tests/test_koopman_tensor_regressors.py` (estimators, hyperparameter selection, agreement of the four class copies). `koopmanrl_utils/koopman_regressor_comparison.py` compares 13 regressors and keeps frozen copies of the regressors as they were before #17, so do not "fix" those copies.

See the root `AGENTS.md` for setup, testing and the working checklist.
