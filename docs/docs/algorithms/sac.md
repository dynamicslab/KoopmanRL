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

A variant that uses a value function $V(s)$ rather than $Q(s,a)$, also from CleanRL.

```bash
uv run -m koopmanrl.value_based_sac_continuous_action --env_id DoubleWell-v0
```

**Source:** `koopmanrl/value_based_sac_continuous_action.py`

## Purpose

These baselines are the direct model-free counterparts to SAKC. Comparing SAKC against them on the same environment and seed budget quantifies the benefit of the Koopman critic.
