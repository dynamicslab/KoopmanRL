---
id: building
sidebar_position: 1
title: Building a Tensor
---

# Building a Koopman Tensor

The Koopman tensor $\mathcal{K}$ is identified from the regression

$$
\phi(x') \approx M \,\big(\psi(u) \otimes \phi(x)\big),
$$

with one column per transition, where $\phi$ and $\psi$ are monomial dictionaries of the state and the action. $M$ is reshaped into $\mathcal{K}$. A decoder $B$ is fitted from $x \approx B^\top \phi(x)$. The implementation is in `koopmanrl/koopman_tensor/` (see its [AGENTS.md](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl/koopman_tensor/AGENTS.md)).

## Generating a tensor

```bash
uv run -m koopmanrl.koopman_tensor.generate_tensor --env_id FluidFlow-v0
```

The script collects random-agent trajectories, fits the tensor and prints the average and maximum one-step prediction errors, of the state and of the dictionary, relative to the average norm, on the identification transitions.

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | `LinearSystem-v0`, `FluidFlow-v0`, `Lorenz-v0` or `DoubleWell-v0` |
| `--num_paths` | `100` | Number of random-agent trajectories |
| `--num_steps_per_path` | `300` | Steps per trajectory |
| `--state_order` | `2` | Order of the state monomials |
| `--action_order` | `2` | Order of the action monomials |
| `--seed` | `123` | Seed |
| `--regressor` | `ols` | `ols`, `ridge`, `sindy` or `rrr` |
| `--rank` | chosen | Rank for `rrr` |
| `--penalty` | chosen | Ridge penalty, in standardised units |
| `--threshold` | chosen | SINDy threshold, in standardised units |
| `--save_model` | off | Pickle the tensor to `./koopman_tensor/saved_models/<env_id>/path_based_tensor.pickle` |
| `--animate` | off | Animate the first trajectory |

For example, a reduced-rank tensor with its rank chosen on held-out data:

```bash
uv run -m koopmanrl.koopman_tensor.generate_tensor --env_id Lorenz-v0 --regressor rrr
```

When a regularised regressor is used, the hyperparameter it ended up with is printed.

## Regressors

`ols` solves the regression directly with `torch.linalg.lstsq`; it is the regressor used for the results of the paper. The regularised estimators `ridge`, `sindy` (sequentially thresholded least squares, 10 threshold-and-refit rounds) and `rrr` (reduced-rank regression) are implemented in [`koopmanrl/koopman_tensor/regressors.py`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl/koopman_tensor/regressors.py). They are not applied to the regression in its raw form, for two reasons given in that module:

- **The identity dominates $M$.** With a time step of 0.01, $M$ is close to $[I\;0\;\cdots\;0]$ and the coefficients that carry the dynamics are small, so a penalty or threshold on the raw coefficients removes the dynamics and keeps persistence.
- **The columns differ in scale** by many orders of magnitude, so a fixed penalty, threshold or rank means something different on every system.

Two changes of coordinates remove both, and leave ordinary least squares unchanged:

1. **Increment target.** The target is $\phi(x') - \phi(x)$. Since $\psi_0(u) = 1$, $\phi(x)$ is the first block of the regressors, so $M = M_\text{inc} + [I\;0\;\cdots\;0]$ and the penalty, threshold or rank acts on the departure from persistence. (This applies to the tensor, not to the generator form or to the decoder.)
2. **Column standardisation.** Every regressor column and every target column is divided by its root-mean-square value, so the penalty and the threshold are dimensionless.

### Choosing the hyperparameter

If `--rank`, `--penalty` or `--threshold` is not given, it is chosen from a short grid by fitting on the first 80% of the transitions and scoring the squared error on the **last 20%**, then the model is refitted on all transitions. The data are stored path by path, so the held-out 20% are the last 20% of the trajectories. The decoder's hyperparameter is always chosen this way.

| Regressor | Grid |
|-----------|------|
| `ridge` | penalties `1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2`, relative to the standardised Gram matrix |
| `sindy` | thresholds `1e-4, 1e-3, 1e-2, 3e-2, 1e-1` on the standardised coefficients |
| `rrr` | ranks of 25%, 50%, 75%, 90% and 100% of the number of outputs |

## Regressors in SKVI and SAKC

Both KARL algorithms take the same choice:

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id Lorenz-v0 --regressor ridge
uv run -m koopmanrl.soft_actor_koopman_critic --env_id Lorenz-v0 --regressor sindy
```

The default is `ols`. The two scripts have no `--rank`, `--penalty` or `--threshold` flags, so a regularised regressor always chooses its hyperparameter on the held-out transitions. How the regressors compare on held-out data is measured by the [regressor comparison](./validation.md#regressor-comparison).
