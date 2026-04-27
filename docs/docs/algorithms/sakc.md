---
id: sakc
sidebar_position: 2
title: Soft Actor Koopman Critic (SAKC)
---

# Soft Actor Koopman Critic (SAKC)

Soft Actor Koopman Critic (SAKC) is the second KARL algorithm. It extends the standard Soft Actor-Critic (SAC) framework by replacing the learned critic network with a critic derived from the Koopman tensor representation of the transition dynamics.

## Algorithm overview

SAKC follows the actor-critic paradigm:

1. **Koopman tensor construction** — as in SKVI, trajectories are collected and a Koopman tensor $\mathcal{K}$ is fitted to the environment's dynamics.
2. **Koopman critic** — instead of learning $Q(s, a)$ from scratch via a neural network, SAKC computes the critic analytically from the Koopman tensor, exploiting its linear structure.
3. **Actor update** — a standard stochastic policy gradient update is applied to the actor network using the Koopman-derived critic as the advantage signal.
4. **Entropy regularisation** — a soft maximum-entropy objective is retained, balancing exploration and exploitation.

The Koopman critic replaces thousands of gradient steps of critic regression with a single closed-form computation, reducing sample complexity while maintaining the expressiveness of the actor.

## Running SAKC

```bash
uv run -m koopmanrl.soft_actor_koopman_critic --env_id FluidFlow-v0
```

## Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to train on |
| `--seed` | `1` | Random seed |
| `--total_timesteps` | `100000` | Training budget |
| `--num_trajectories` | `500` | Trajectories for Koopman tensor fitting |
| `--actor_lr` | `3e-4` | Actor learning rate |
| `--alpha` | `0.2` | Entropy regularisation coefficient |

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
