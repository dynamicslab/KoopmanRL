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
4. At the end, writes the best-found configuration to `<--storage_dir>/<--output_file>.json`.

The optimization metric is the average episodic return over the last `average_window` episodes of training.

:::caution
The default of `--storage_dir` is a hard-coded path from the authors' machine, so always pass `--storage_dir` (e.g. `.`), and choose an `--output_file` that does not overwrite the shipped `configurations/*_hparams.json`. `--cpu_cores_per_trial` defaults to `28`; Ray cannot schedule a trial that asks for more cores than the machine has, so set it to at most your core count.
:::

## SAKC optimization

```bash
uv run -m koopmanrl.sakc_optuna_opt \
    --env_id FluidFlow-v0 \
    --num_samples 50 \
    --max_concurrent 4 \
    --total_timesteps 50000 \
    --cpu_cores_per_trial 4 \
    --storage_dir . \
    --output_file my_sakc_fluid_flow_hparams
```

### Search space

| Hyperparameter | Distribution |
|----------------|-------------|
| `seed` | `randint(0, 10000)` |
| `v-lr` | `loguniform(0.0001, 0.1)` |
| `q-lr` | `loguniform(0.0001, 0.1)` |
| `num-paths` | `choice([50, 75, 100, 125, 150, 175, 200])` |
| `num-steps-per-path` | `choice([75, 100, ..., 300])` (steps of 25) |
| `state-order` | `choice([1, 2, 3, 4])` |
| `action-order` | `choice([1, 2, 3, 4])` |

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to optimise for |
| `--num_samples` | `50` | Number of Optuna trials |
| `--max_concurrent` | `4` | Parallel trials at a time |
| `--total_timesteps` | `50000` | Training budget per trial |
| `--cpu_cores_per_trial` | `28` | CPU allocation per Ray worker; must not exceed the machine's cores |
| `--num_envs` | `1` | Recorded in the trial configuration (not used by the training wrappers) |
| `--average_window` | `5` | Number of final episodes the metric is averaged over |
| `--storage_dir` | author's local path | Directory the result JSON is written to; always set it |
| `--output_file` | `sakc_linear_system_params` | Output JSON filename (no extension) |

The SKVI script takes the same flags (default `--output_file`: `skvi_linear_system_params`).

## SKVI optimization

```bash
uv run -m koopmanrl.skvi_optuna_opt \
    --env_id Lorenz-v0 \
    --num_samples 50 \
    --max_concurrent 4 \
    --total_timesteps 50000 \
    --cpu_cores_per_trial 4 \
    --storage_dir . \
    --output_file my_skvi_lorenz_hparams
```

### Search space

| Hyperparameter | Distribution |
|----------------|-------------|
| `seed` | `randint(0, 10000)` |
| `learning-rate` | `loguniform(0.0003, 0.003)` |
| `number-of-train-epochs` | `choice([100, 125, 150, 175])` |
| `num-paths` | `choice([50, 75, 100, 125, 150, 175, 200])` |
| `num-steps-per-path` | `choice([75, 100, ..., 300])` (steps of 25) |
| `state-order` | `choice([1, 2, 3, 4])` |
| `action-order` | `choice([1, 2, 3, 4])` |

SKVI enforces a minimum data size: if `num-paths × num-steps-per-path < 2^14`, the batch size is reduced from `2^14` to the largest power of two below that product.

## Pre-optimised configurations

The `configurations/` directory ships with best-found JSON files for SKVI and SAKC on all four environments, next to the settings files of the reproduction scripts:

```
configurations/
├── ablations.json                    # grids and seeds of the ablation runs (run_ablations, Snakemake)
├── episodic_returns.json             # algorithms, benchmarks and seeds of the episodic-return runs
├── sakc_double_well_hparams.json     # best SAKC configuration, Double Well
├── sakc_fluid_flow_hparams.json      # best SAKC configuration, Fluid Flow
├── sakc_linear_system_hparams.json   # best SAKC configuration, Linear System
├── sakc_lorenz_hparams.json          # best SAKC configuration, Lorenz
├── skvi_double_well_hparams.json     # best SKVI configuration, Double Well
├── skvi_fluid_flow_hparams.json      # best SKVI configuration, Fluid Flow
├── skvi_linear_system_hparams.json   # best SKVI configuration, Linear System
├── skvi_lorenz_hparams.json          # best SKVI configuration, Lorenz
└── tsne.json                         # arguments of the t-SNE workflow (tsne_koopman_tensor)
```

The eight `*_hparams.json` files are the results of `koopmanrl_utils/run_skvi_optimization.py` and `koopmanrl_utils/run_sakc_optimization.py`, which run `skvi_optuna_opt` and `sakc_optuna_opt` one after the other on the four benchmarks with the default flags of the optimization scripts, writing `<algorithm>_<benchmark>_hparams.json` to `--storage_dir`. These drivers take no arguments and start the full studies as soon as they are run.

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

`koopmanrl_utils/run_optimized_experiments.py` re-runs the best configurations, and the LQR and SAC baselines next to them, across multiple seeds for final performance evaluation. Run without arguments it launches all 495 runs of 50,000 steps each, so start with a dry run and select a part:

```bash
uv run -m koopmanrl_utils.run_optimized_experiments --dry_run   # print the commands only
uv run -m koopmanrl_utils.run_optimized_experiments \
    --algorithms skvi sakc --environments Lorenz-v0 --seeds 1 2 3 --num_workers 2
```

`--environments`, `--algorithms` and `--seeds` filter the campaign, and `--num_workers` sets how many runs are made at the same time. See [`koopmanrl_utils/EPISODIC_RETURNS.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/EPISODIC_RETURNS.md) and the README section [Reproducing the Results of the Paper](https://github.com/dynamicslab/KoopmanRL#reproducing-the-results-of-the-paper) for the full procedure.
