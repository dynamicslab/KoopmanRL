---
id: sakc
sidebar_position: 2
title: Soft Actor Koopman-Critic (SAKC)
---

# Soft Actor Koopman-Critic (SAKC)

Soft Actor Koopman-Critic (SAKC) is the second KARL algorithm. It extends the value-based Soft Actor-Critic (SAC) framework by replacing the neural state-value network with one that is linear in the Koopman observables, and by computing the critic's bootstrap target through the Koopman tensor representation of the transition dynamics.

## Algorithm overview

SAKC follows the actor-critic paradigm:

1. **Koopman tensor construction** — as in SKVI, trajectories are collected with a random agent and a Koopman tensor $\mathcal{K}$ is fitted to the environment's dynamics, $\phi(x') \approx K(u)\,\phi(x)$.
2. **Koopman value function** — the state-value network $V(x) = w^\top \phi(x)$ (`SoftKoopmanVNetwork`) is linear in the observables. Its weights are trained by gradient descent (Adam, learning rate `--v_lr`) towards the soft target $\min_i Q_i(x, a) - \alpha \log \pi(a \mid x)$ with $a \sim \pi$.
3. **Q-networks with a Koopman target** — two neural Q-networks (`SoftQNetwork`, learning rate `--q_lr`) are kept as in SAC. Their regression target uses the Koopman tensor to take the expectation over next states in closed form: $r + \gamma\, \bar{w}^\top K(u)\,\phi(x)$, where $\bar{w}$ are the weights of a target copy of the value network.
4. **Actor update** — the stochastic actor is updated with the SAC loss $\alpha \log \pi(a \mid x) - \min_i Q_i(x, a)$ (learning rate `--policy_lr`).
5. **Entropy regularisation** — a soft maximum-entropy objective is retained, balancing exploration and exploitation; $\alpha$ is tuned automatically by default.

## Running SAKC

```bash
uv run -m koopmanrl.soft_actor_koopman_critic --env_id FluidFlow-v0
```

## Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to train on |
| `--seed` | `1` | Random seed |
| `--total_timesteps` | `50000` | Training budget (environment steps) |
| `--num_paths` | `100` | Random-agent trajectories for Koopman tensor fitting |
| `--num_steps_per_path` | `300` | Steps per trajectory for Koopman tensor fitting |
| `--state_order` | `2` | Monomial order of the state observables $\phi(x)$ |
| `--action_order` | `2` | Monomial order of the action dictionary |
| `--regressor` | `ols` | Regressor for the Koopman tensor: `ols`, `ridge`, `sindy` or `rrr` |
| `--v_lr` | `1e-3` | Learning rate of the Koopman value network |
| `--q_lr` | `1e-3` | Learning rate of the Q-networks |
| `--policy_lr` | `3e-4` | Actor learning rate |
| `--autotune` | `True` | Automatic tuning of the entropy coefficient; passing the bare `--autotune` flag turns it **off** |
| `--alpha` | `0.2` | Fixed entropy regularisation coefficient, used only when autotuning is off |

Precedence is CLI flag > `--config_file` > default: `--env_id`, `--seed`, `--total_timesteps`, `--num_paths`, `--num_steps_per_path`, `--state_order`, `--action_order`, `--v_lr` and `--q_lr` are read from the config file when they are not given on the command line, and fall back to the defaults above when neither sets them. The other flags always use their CLI value or default.

Run `--help` to see the full list:

```bash
uv run -m koopmanrl.soft_actor_koopman_critic --help
```

## Using a pre-optimised config

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_fluid_flow_hparams.json
```

Override individual flags even when using a config:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_double_well_hparams.json \
    --seed 42
```

## Source

`koopmanrl/soft_actor_koopman_critic.py`
