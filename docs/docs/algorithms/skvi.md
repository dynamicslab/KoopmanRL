---
id: skvi
sidebar_position: 1
title: Soft Koopman Value Iteration (SKVI)
---

# Soft Koopman Value Iteration (SKVI)

Soft Koopman Value Iteration (SKVI) is the first of the two KARL algorithms. It replaces the standard Bellman backup in soft value iteration with one that exploits a learned Koopman tensor representation of the environment's transition dynamics.

## Algorithm overview

SKVI operates in discrete value-iteration fashion:

1. **Koopman tensor construction** — collect trajectories from the environment and fit a Koopman tensor $\mathcal{K}$ that maps observable functions of the current state-action pair to observables of the next state.
2. **Lifted value iteration** — define the soft Bellman backup over the lifted (observable) space rather than the raw state space, exploiting the linearity of $\mathcal{K}$.
3. **Policy extraction** — derive the policy from the soft value function in the observable space, then project back to action space.

The key benefit is that the linear structure of the Koopman operator allows the Bellman backup to be solved analytically, avoiding the regression step needed by model-free methods.

## Running SKVI

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id LinearSystem-v0
```

All supported environment IDs:

| Environment | `--env_id` |
|-------------|-----------|
| Linear System | `LinearSystem-v0` |
| Fluid Flow | `FluidFlow-v0` |
| Lorenz | `Lorenz-v0` |
| Double Well | `DoubleWell-v0` |

## Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to train on |
| `--seed` | `1` | Random seed |
| `--total_timesteps` | `100000` | Training budget |
| `--num_trajectories` | `500` | Trajectories for Koopman tensor fitting |
| `--observable_dim` | Env-specific | Dimension of the observable (lifted) space |

Run `--help` to see the full list:

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --help
```

## Using a pre-optimised config

```bash
uv run python -m koopmanrl.soft_koopman_value_iteration \
    --config_file configurations/skvi_lorenz_hparams.json
```

## Source

`koopmanrl/soft_koopman_value_iteration.py`
