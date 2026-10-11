# KoopmanRL Source Code Guide

Core source code of the KoopmanRL package: the two KARL algorithms `soft_koopman_value_iteration` (SKVI) and `soft_actor_koopman_critic` (SAKC), the LQR and SAC baselines, and the Ray Tune hyperparameter optimizations of SKVI and SAKC (`skvi_optuna_opt`, `sakc_optuna_opt`). The Koopman tensor logic is factored out into the `koopman_tensor` subdirectory, and the control environments into the `environments` subdirectory.

## Running Core Functionality

Every algorithm is a script with a typed (`tap`) argument parser, run from the repository root:

- `uv run -m koopmanrl.soft_koopman_value_iteration`
- `uv run -m koopmanrl.soft_actor_koopman_critic`
- `uv run -m koopmanrl.linear_quadratic_regulator`
- `uv run -m koopmanrl.sac_continuous_action`
- `uv run -m koopmanrl.value_based_sac_continuous_action`
- `uv run -m koopmanrl.skvi_optuna_opt` / `uv run -m koopmanrl.sakc_optuna_opt` (hyperparameter optimization)

All of them take `--env_id` (`LinearSystem-v0`, `FluidFlow-v0`, `Lorenz-v0`, `DoubleWell-v0`) and list their options with `--help`. SKVI and SAKC take `--config_file configurations/<algo>_<env_slug>_hparams.json`; `utils.py::load_and_apply_config` applies it with the precedence command line > file > fallback, mapping the kebab-case keys of the file through each script's key map.

## Directory Structure

```
koopmanrl/
├── environments/                                 # The four control environments (see environments/AGENTS.md)
├── koopman_tensor/                               # Koopman tensor classes, shared regressors, generate_tensor (see koopman_tensor/AGENTS.md)
├── __init__.py                                   # Initialization of the Python package
├── AGENTS.md                                     # This file
├── interpretability_discrete_value_iteration.py  # Interpretability experiment of the discrete value iteration; broken: imports a module `analysis` that does not exist
├── koopman_observables.py                        # Copy of koopman_tensor/observables/torch_observables.py; the one SKVI and SAKC import
├── linear_quadratic_regulator.py                 # Baseline: discounted, entropy-regularised LQR
├── opt_wrappers.py                               # Training loops of SKVI and SAKC as functions for Ray Tune; imports the classes of the two algorithm files
├── sac_continuous_action.py                      # Baseline: Q-based soft actor-critic. From CleanRL.
├── sakc_optuna_opt.py                            # Ray Tune (Optuna search) hyperparameter optimization of SAKC
├── skvi_optuna_opt.py                            # Ray Tune (Optuna search) hyperparameter optimization of SKVI
├── soft_actor_koopman_critic.py                  # Soft Actor Koopman-Critic, with its own copy of the Koopman tensor class
├── soft_koopman_value_iteration.py               # Soft Koopman Value Iteration, with its own copy of the Koopman tensor class
├── utils.py                                      # make_env and load_and_apply_config
└── value_based_sac_continuous_action.py          # Baseline: soft actor-critic with a V-network, built on CleanRL's SoftQNetwork
```

## Critical Patterns

* **Four copies of the Koopman tensor class.** `soft_koopman_value_iteration.py` and `soft_actor_koopman_critic.py` each define their own `KoopmanTensor` and do not import the one in `koopman_tensor/`; `koopman_tensor/torch_tensor.py` and `koopman_tensor/numpy_tensor.py` are the other two. All four call `koopman_tensor/regressors.py::tensor_regression`. A change to the tensor must be made in all four, and `tests/test_koopman_tensor_regressors.py` checks that they agree.
* **Least-squares driver.** SKVI and SAKC call `torch.linalg.lstsq(..., driver="gelsd")` in their `ols`, and SKVI also in the solve of its value-function weights. The default CPU driver, `gelsy`, does not return the same result in every call, which made same-seed SAKC runs log different returns. Keep `gelsd` there. `koopman_tensor/torch_tensor.py` deliberately keeps the default driver (see `koopman_tensor/AGENTS.md`).
* **Hyperparameter optimization.** `*_optuna_opt` run `opt_wrappers.{skvi,sakc}_tuning_wrapper` under Ray Tune with `--num_samples 50`. The defaults `--cpu_cores_per_trial 28` and `--storage_dir` (a path on the authors' machine, where `<output_file>.json` is written at the end of the study) must be overridden on other machines. The tuned results are the `*_hparams.json` files in `configurations/`. When implementing new hyperparameter optimization logic, use Ray Tune.
* **Baselines.** `linear_quadratic_regulator`, `sac_continuous_action`, and `value_based_sac_continuous_action` are inherited from other libraries and should be left untouched, and not be considered when designing new functionality. The one departure is their `--seed` argument: a run uses the given seed, and draws one at random only when none is given, so that `koopmanrl_utils/run_optimized_experiments.py` can repeat a baseline run (`tests/test_rl.py` checks it).
* **Observables.** `koopman_observables.py` and `koopman_tensor/observables/torch_observables.py` are identical files; SKVI, SAKC and `koopmanrl_utils/skvi_policy_checks.py` import the former. Keep them identical when changing either.

See the root `AGENTS.md` for setup, testing and the working checklist.
