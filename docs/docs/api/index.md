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
| `koopmanrl.koopman_tensor` | Stand-alone Koopman tensor classes, regressors and the `generate_tensor` CLI |
| `koopmanrl.opt_wrappers` | Wrappers for Optuna/Ray Tune integration |
| `koopmanrl.utils` | Shared utilities (config loading, seeded environment construction, folder creation) |
| `koopmanrl.sakc_optuna_opt` | SAKC hyperparameter optimization |
| `koopmanrl.skvi_optuna_opt` | SKVI hyperparameter optimization |

## `koopmanrl_utils` — post-processing and visualisation

| Module | Contents |
|--------|----------|
| `koopmanrl_utils.movies.generate_trajectories` | Roll out policies and save trajectory `.npy` files |
| `koopmanrl_utils.movies.generate_trajectory_figure` | Static PNG trajectory plots with optional vector field |
| `koopmanrl_utils.movies.generate_gifs` | Animate saved trajectories as trajectory and cost-ratio GIFs |
| `koopmanrl_utils.run_optimized_experiments` | Re-run best configs and the baselines across seeds (episodic-return figures) |
| `koopmanrl_utils.run_ablations` | Run the SKVI and SAKC hyperparameter grids behind the two ablation figures |
| `koopmanrl_utils.run_skvi_optimization` | Run `skvi_optuna_opt` on all four benchmarks (no CLI; starts the studies when run) |
| `koopmanrl_utils.run_sakc_optimization` | Run `sakc_optuna_opt` on all four benchmarks (no CLI; starts the studies when run) |
| `koopmanrl_utils.dataframe_creator` | Collect the episodic returns of a folder of TensorBoard runs into a JSON data frame |
| `koopmanrl_utils.process_episodic_returns` | Reduce an episodic-return data frame to IQM curves with bootstrap confidence bands (table for plotting) |
| `koopmanrl_utils.process_skvi_ablations` | Reduce an SKVI ablation data frame to the IQM surface table |
| `koopmanrl_utils.process_sakc_ablations` | Reduce an SAKC ablation data frame to the IQM surface table |
| `koopmanrl_utils.plot_csv_from_tensorboards` | Plot training curves from TensorBoard logs |
| `koopmanrl_utils.tsne_koopman_tensor` | t-SNE embedding of Koopman tensors of the four benchmarks in a common dictionary basis |
| `koopmanrl_utils.koopman_prediction_validation` | Prediction error of the fitted Koopman tensor on held-out random-agent data |
| `koopmanrl_utils.koopman_regressor_comparison` | Held-out accuracy of the Koopman tensor under different regression algorithms |
| `koopmanrl_utils.skvi_policy_checks` | Read SKVI's deployed policy off the Koopman tensor and check it against references |
| `koopmanrl_utils.skvi_sensitivity_checks` | Tensor accuracy along the trained SKVI policy, and sensitivity of SKVI's control |
| `koopmanrl_utils.interpret_koopman` | Load and print stored SKVI value-function weights (SAKC not implemented) |

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

SKVI and SAKC each define their own `KoopmanTensor` class inside their training script (`koopmanrl.soft_koopman_value_iteration.KoopmanTensor`, `koopmanrl.soft_actor_koopman_critic.KoopmanTensor`). Both fit a tensor $\mathcal{K}$ from batches of transition tuples $(x_t, u_t, x_{t+1})$ such that

$$
\phi(x_{t+1}) \approx \mathcal{K}(u_t) \, \phi(x_t)
$$

where $\phi$ is the monomial observable (lifting) function `koopmanrl.koopman_observables.monomials`. What the two scripts share is that observable module and the regularised regressors of `koopmanrl.koopman_tensor.regressors`.

The `koopmanrl.koopman_tensor` package holds stand-alone versions for use outside the training scripts:

| Module | Contents |
|--------|----------|
| `koopmanrl.koopman_tensor.torch_tensor` | `KoopmanTensor` (PyTorch) |
| `koopmanrl.koopman_tensor.numpy_tensor` | `KoopmanTensor` (NumPy) |
| `koopmanrl.koopman_tensor.regressors` | `ridge`, `sindy`, `rrr`, `fit` and `tensor_regression`: ridge, sequentially thresholded least squares (SINDy) and reduced-rank regression of the tensor, with the hyperparameter chosen on held-out transitions when not given |
| `koopmanrl.koopman_tensor.utils` | `save_tensor` / `load_tensor` (pickle under `./koopman_tensor/saved_models/<env_id>/`) |
| `koopmanrl.koopman_tensor.observables` | NumPy and PyTorch observable dictionaries (monomials, indicators, Gaussians) |
| `koopmanrl.koopman_tensor.generate_tensor` | CLI that collects random-agent data and fits a tensor |

The ordinary-least-squares fit (`--regressor ols`, the default) is solved with `torch.linalg.lstsq`; `ridge`, `sindy` and `rrr` go through `regressors.tensor_regression`. To fit a tensor from the command line:

```bash
uv run -m koopmanrl.koopman_tensor.generate_tensor --env_id Lorenz-v0 --regressor ridge
```

`--rank` (for `rrr`), `--penalty` (for `ridge`) and `--threshold` (for `sindy`) are chosen on held-out transitions when omitted.

## Config loading

`koopmanrl.utils.load_and_apply_config(args, key_map, fallbacks)` provides layered configuration merging with the precedence **CLI flag > config file > fallback default**. It reads the JSON file named by `args.config_file` (if set), copies each value whose hyphenated key is in `key_map` onto the attribute that is still `None` (i.e. not given on the CLI), and then fills any attribute that is still `None` from `fallbacks`.

```python
from koopmanrl.utils import load_and_apply_config

key_map = {"seed": "seed", "num-paths": "num_paths"}  # JSON key -> args attribute
fallbacks = {"seed": 1, "num_paths": 100}  # used when neither CLI nor file sets a value

args = MyArgs().parse_args()  # MyArgs declares config_file and Optional[...] = None fields
args = load_and_apply_config(args, key_map, fallbacks)
```
