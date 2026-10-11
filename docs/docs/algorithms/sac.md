---
id: sac
sidebar_position: 4
title: Soft Actor-Critic Baselines
---

# Soft Actor-Critic Baselines

KoopmanRL ships two CleanRL-style SAC baselines that provide model-free comparators for the KARL algorithms.

## Q-value SAC

Standard Soft Actor-Critic with a Q-function critic, adapted from CleanRL.

```bash
uv run -m koopmanrl.sac_continuous_action --env_id Lorenz-v0
```

**Source:** `koopmanrl/sac_continuous_action.py`

## Value-based SAC

A variant that adds a learned state-value network $V(s)$ (learning rate `--v_lr`) to the two Q-networks: the Q-networks regress onto $r + \gamma V_{\text{target}}(s')$ instead of a soft Q target. It imports CleanRL's `SoftQNetwork`, but the value network and its training loop are KoopmanRL's own; it is the model-free counterpart of SAKC, which replaces this $V$ with one linear in the Koopman observables.

```bash
uv run -m koopmanrl.value_based_sac_continuous_action --env_id DoubleWell-v0
```

**Source:** `koopmanrl/value_based_sac_continuous_action.py`

## Seeding

Both baselines take `--seed`. Without it, a seed is drawn at random and recorded in the run name (`<env_id>__<exp_name>__<seed>__<timestamp>`), so a run can be repeated with `--seed`.

## Purpose

These baselines are the direct model-free counterparts to SAKC. Comparing SAKC against them on the same environment and seed budget quantifies the benefit of the Koopman critic.
