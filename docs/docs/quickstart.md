---
id: quickstart
sidebar_position: 1
title: Quick Start
---

# Quick Start

This guide gets you from zero to a running experiment in under five minutes.

## Prerequisites

- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

Clone the repository and sync the environment:

```bash
git clone https://github.com/dynamicslab/KoopmanRL.git
cd KoopmanRL
uv sync
```

## Your First Experiment

Run the Linear Quadratic Regulator on the default Linear System environment:

```bash
uv run -m koopmanrl.linear_quadratic_regulator
```

Switch to a different environment with `--env_id`:

```bash
uv run -m koopmanrl.linear_quadratic_regulator --env_id FluidFlow-v0
```

## Running a KARL Algorithm

Run **Soft Koopman Value Iteration** on the Lorenz attractor:

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id Lorenz-v0
```

Run **Soft Koopman Actor-Critic** on the Double-Well potential:

```bash
uv run -m koopmanrl.soft_actor_koopman_critic --env_id DoubleWell-v0
```

## Using Pre-optimised Hyperparameters

The `configurations/` directory ships with best-found hyperparameter JSON files for every algorithm-environment pair:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_fluid_flow_hparams.json
```

Individual flags always override values from the config file:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_fluid_flow_hparams.json \
    --seed 42 \
    --total_timesteps 100000
```

## Discovering All Options

Every module exposes a typed argument parser. Query it with `--help`:

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --help
```

## Next Steps

- [Full installation options](./installation) — virtual environments, pip installs.
- [Algorithm documentation](./algorithms/skvi) — SKVI and SAKC in detail.
- [Environment reference](./environments/linear-system) — all four benchmark environments.
- [Hyperparameter optimization](./hyperparameter-optimization/overview) — Optuna tuning pipelines.
