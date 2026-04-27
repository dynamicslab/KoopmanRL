---
id: overview
sidebar_position: 1
title: Overview
---

# Hyperparameter Optimization

KoopmanRL includes automated hyperparameter search pipelines for both KARL algorithms, built on **Optuna** as the search backend and **Ray Tune** for distributed trial management.

## How it works

Each optimization script:

1. Defines a search space over the algorithm's tunable hyperparameters.
2. Spawns `max_concurrent` parallel trials via Ray, each running a full training run.
3. Uses **Tree-structured Parzen Estimator (TPE)** via Optuna to propose the next configuration based on completed trials.
4. At the end, writes the best-found configuration to a JSON file in `configurations/`.

The optimization metric is the average episodic return over the last `average_window` episodes of training.

## SAKC optimization

```bash
uv run -m koopmanrl.sakc_optuna_opt \
    --env_id FluidFlow-v0 \
    --num_samples 50 \
    --max_concurrent 4 \
    --total_timesteps 50000 \
    --output_file configurations/sakc_fluid_flow_hparams
```

### Search space

| Hyperparameter | Distribution |
|----------------|-------------|
| `seed` | `randint(0, 10000)` |
| `v-lr` | `loguniform(0.0001, 0.1)` |
| `q-lr` | `loguniform(0.0001, 0.1)` |
| `num-paths` | `randint(50, 300)` |
| `num-steps-per-path` | `randint(50, 300)` |
| `state-order` | `randint(2, 5)` |
| `action-order` | `randint(2, 5)` |

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to optimise for |
| `--num_samples` | `50` | Number of Optuna trials |
| `--max_concurrent` | `4` | Parallel trials at a time |
| `--total_timesteps` | `50000` | Training budget per trial |
| `--cpu_cores_per_trial` | `28` | CPU allocation per Ray worker |
| `--output_file` | `sakc_linear_system_params` | Output JSON filename (no extension) |

## SKVI optimization

```bash
uv run -m koopmanrl.skvi_optuna_opt \
    --env_id Lorenz-v0 \
    --num_samples 50 \
    --max_concurrent 4 \
    --total_timesteps 50000 \
    --output_file configurations/skvi_lorenz_hparams
```

### Search space

| Hyperparameter | Distribution |
|----------------|-------------|
| `seed` | `randint(0, 10000)` |
| `learning-rate` | `loguniform(0.00001, 0.01)` |
| `number-of-train-epochs` | `randint(50, 200)` |
| `num-paths` | `randint(50, 300)` |
| `num-steps-per-path` | `randint(50, 300)` |
| `state-order` | `randint(2, 6)` |
| `action-order` | `randint(1, 4)` |

SKVI enforces a minimum data size: if `num-paths × num-steps-per-path < 2^14`, the batch size is rounded down to the nearest power of two.

## Pre-optimised configurations

The `configurations/` directory ships with best-found JSON files for all algorithm-environment pairs:

```
configurations/
├── sakc_double_well_hparams.json
├── sakc_fluid_flow_hparams.json
├── sakc_lorenz_hparams.json
├── skvi_double_well_hparams.json
├── skvi_fluid_flow_hparams.json
└── skvi_lorenz_hparams.json
```

Pass any of these directly to the training scripts:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_lorenz_hparams.json
```

Individual flags always override config file values, so you can start from a pre-optimised config and vary a single parameter:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_lorenz_hparams.json \
    --seed 42
```

## Running optimised experiments

After optimization, `run_optimized_experiments.py` re-runs the best configurations across multiple seeds for final performance evaluation:

```bash
uv run python -m koopmanrl_utils.run_optimized_experiments
```
