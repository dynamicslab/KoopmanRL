# KoopmanRL Source Code Guide

Core source code of the KoopmanRL package culminating in the 4 core algorithm `soft_actor_koopman_critic`, `soft_koopman_value_iteration`, `sakc_optuna_opt`, and `skvi_optuna_opt` with the core Koopman tensor logic factored out into the `koopman_tensor` subdirectory, and the control environments factored out into the `environments` subdirectory.

## Running Core Functionality

KoopmanRL's core algorithms can be executed as scripts, namely the main 4 algorithms are:

- `uv run python -m koopmanrl.soft_actor_koopman_critic`
- `uv run python -m koopmanrl.soft_koopman_value_iteration`
- `uv run python -m koopmanrl.sakc_optuna_opt`
- `uv run python -m koopmanrl.skvi_optuna_opt`

All 4 provide numerous command line options, which can be explored with

`uv run python -m koopmanrl.soft_actor_koopman_critic --help`

## Directory Structure

```
koopmanrl/
├── environments/                                 # Implementations of the reinforcement learning environments
├── koopman_tensor/                               # Koopman tensor logic
├── __init__.py                                   # Initialization of the Python package
├── AGENTS.md                                     # This file
├── interpretability_discrete_value_iteration.py  # Implementation of the interpretability of the KoopmanRL-based algorithms
├── koopman_observables.py                        # Observable (polynomial) structures of the Koopman operator
├── linear_quadratic_regulator.py                 # Implementation of the linear quadratic regulator
├── opt_wrappers.py                               # Function-based implementations of the Koopman algorithms
├── sac_continuous_action.py                      # Q-based formulation of the soft actor-critic algorithm. From CleanRL.
├── sakc_optuna_opt.py                            # Optuna hyperparameter optimization of the Soft Actor Koopman-Critic
├── skvi_optuna_opt.py                            # Optuna hyperparameter optimization of the Soft Koopman Value Iteration
├── soft_actor_koopman_critic.py                  # Implementation of the Soft Actor Koopman-Critic algorithm
├── soft_koopman_value_iteration.py               # Implementation of the Soft Koopman Value Iteration algorithm
├── utils.py                                      # Utilities for the reinforcement learning algorithms
└── value_based_sac_continuous_action.py          # Value-based formulation of the soft actor-critic algorithm. From CleanRL.
```

## Critical Patterns

* The main algorithms of the KoopmanRL package are `sakc_optuna_opt`, `skvi_optuna_opt`, `soft_actor_koopman_critic`, and `soft_koopman_value_iteration`.
* If possible, when working with the Koopman tensor make sure to inherit as much as possible from either the `opt_wrappers`, or the `koopman_tensor` directory. All the core logic is represented in the `opt_wrappers`.
* When implementing new hyperparameter optimization logic, utilize Ray tune functionality as well as possible.
* `linear_quadratic_regulator`, `sac_continuous_action`, and `value_based_sac_continuous_action` are inherited from other libraries and should be left untouched, and not be considered when designing new functionality.

## Working Checklist

1. Review the relevant AGENTS guide(s) and existing tests/examples for the functionality you touch.
2. Prototype changes in single files or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (tests/test_*.py) alongside code changes.
4. Run the scoped pytest command (uv run test -m ...) before submitting.
5. Keep documentation edits minimal and aligned.
