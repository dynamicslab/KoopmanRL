---
id: api
sidebar_position: 1
title: API Reference
---

# API Reference

KoopmanRL is organised into two top-level packages.

## `koopmanrl` — core algorithms and environments

| Module | Contents |
|--------|----------|
| `koopmanrl.environments` | Four benchmark Gym environments |
| `koopmanrl.soft_koopman_value_iteration` | SKVI training script |
| `koopmanrl.soft_actor_koopman_critic` | SAKC training script |
| `koopmanrl.linear_quadratic_regulator` | LQR baseline |
| `koopmanrl.sac_continuous_action` | SAC (Q-value) baseline |
| `koopmanrl.value_based_sac_continuous_action` | SAC (value-function) baseline |
| `koopmanrl.koopman_observables` | Observable (lifting) functions |
| `koopmanrl.koopman_tensor` | Koopman tensor construction and fitting |
| `koopmanrl.opt_wrappers` | Wrappers for Optuna/Ray Tune integration |
| `koopmanrl.utils` | Shared utilities (config loading, seeding) |
| `koopmanrl.sakc_optuna_opt` | SAKC hyperparameter optimization |
| `koopmanrl.skvi_optuna_opt` | SKVI hyperparameter optimization |

## `koopmanrl_utils` — post-processing and visualisation

| Module | Contents |
|--------|----------|
| `koopmanrl_utils.movies.generate_trajectories` | Roll out policies and save trajectory `.npy` files |
| `koopmanrl_utils.movies.generate_trajectory_figure` | Static PNG trajectory plots with optional vector field |
| `koopmanrl_utils.movies.generate_gifs` | Animated GIF generation from saved trajectories |
| `koopmanrl_utils.run_optimized_experiments` | Re-run best configs across seeds |
| `koopmanrl_utils.plot_csv_from_tensorboards` | Plot training curves from TensorBoard CSVs |

## Environments

All four environments follow the [OpenAI Gym](https://gymnasium.farama.org/) interface (`gym==0.23.1`). They are registered at import time and can be instantiated with:

```python
import gym
import koopmanrl.environments  # registers all environments

env = gym.make("FluidFlow-v0")
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
```

### Environment IDs

| ID | Class | Source |
|----|-------|--------|
| `LinearSystem-v0` | `LinearSystem` | `koopmanrl/environments/linear_system.py` |
| `FluidFlow-v0` | `FluidFlow` | `koopmanrl/environments/fluid_flow.py` |
| `Lorenz-v0` | `Lorenz` | `koopmanrl/environments/lorenz.py` |
| `DoubleWell-v0` | `DoubleWell` | `koopmanrl/environments/double_well.py` |

## Koopman tensor

The `koopmanrl.koopman_tensor` module provides the core Koopman tensor fitting routine used by both SKVI and SAKC. It accepts batches of transition tuples $(x_t, u_t, x_{t+1})$ and returns a tensor $\mathcal{K}$ such that

$$
\phi(x_{t+1}) \approx \mathcal{K}(u_t) \, \phi(x_t)
$$

where $\phi$ is the observable (lifting) function chosen via `koopmanrl.koopman_observables`.

## Config loading

`koopmanrl.utils.load_and_apply_config` provides layered configuration merging: a JSON file sets defaults, and any CLI flag explicitly provided takes precedence.

```python
from koopmanrl.utils import load_and_apply_config

args = MyArgs().parse_args()
args = load_and_apply_config(args, "configurations/sakc_fluid_flow_hparams.json")
```
