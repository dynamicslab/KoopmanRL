---
id: episodic-returns
sidebar_position: 2
title: Episodic Returns
---

# Episodic Returns

The episodic-return figures compare LQR, SAC (Q), SAC (V), SKVI and SAKC on the linear system, the fluid flow, the Lorenz system and the double well. The full walkthrough, including the data-frame format and pgfplots snippets, is in [`koopmanrl_utils/EPISODIC_RETURNS.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/EPISODIC_RETURNS.md).

## Step 1 — Runs

```bash
uv run -m koopmanrl_utils.run_optimized_experiments --dry_run          # print the commands only
uv run -m koopmanrl_utils.run_optimized_experiments --num_workers 8    # eight runs at a time
```

This is 495 runs: five algorithms on four benchmarks, 25 seeds per benchmark (24 on the linear system). SKVI and SAKC use the tuned configurations in `configurations/`; the baselines use the defaults of their scripts. Algorithms, benchmarks and seeds are read from `configurations/episodic_returns.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `--environments` | all four | Benchmarks, e.g. `Lorenz-v0` |
| `--algorithms` | `lqr sac_q sac_v skvi sakc` | Algorithms to run |
| `--seeds` | seeds in the run list | Seeds to run on every selected benchmark |
| `--total_timesteps` | 50,000 | Environment steps per run |
| `--output_dir` | `episodic_returns_results` | Working directory of the runs |
| `--num_workers` | `1` | Runs made at the same time (Ray) |
| `--dry_run` | off | Print the commands without running them |

A part of the experiments is selected with the filters, for example:

```bash
uv run -m koopmanrl_utils.run_optimized_experiments --algorithms lqr sac_q --environments Lorenz-v0
```

Logs go to `<output_dir>/runs/<run name>/` for the baselines and to `runs/SKVI/` and `runs/SAKC/` for the two KARL algorithms. The largest run, SKVI on the fluid flow, takes about 3.5 GB of memory. A failed run does not stop the others; failed runs are listed at the end, and their incomplete logs have to be removed before step 2. Use an output directory that holds no other runs, since step 2 takes every run it finds.

## Step 2 — Data frames

```bash
mkdir -p episodic_returns_results/frames

uv run -m koopmanrl_utils.dataframe_creator \
    --mode Episodic_Returns \
    --system FluidFlow-v0 \
    --rl_algo soft_actor_koopman_critic \
    --target_dir episodic_returns_results/runs/SAKC \
    --storage_dir episodic_returns_results/frames \
    --output_file SAKC_fluid_flow.json
```

`--rl_algo` is the algorithm's name in the run folders: `linear_quadratic_regulator`, `sac_continuous_action`, `value_based_sac_continuous_action` (all three under `runs/`), `soft_koopman_value_iteration` (`runs/SKVI`) or `soft_actor_koopman_critic` (`runs/SAKC`). `--storage_dir` has to exist.

## Step 3 — Tables

```bash
mkdir -p episodic_returns_results/csv/fluid_flow

uv run -m koopmanrl_utils.process_episodic_returns \
    --root_dir episodic_returns_results/frames \
    --data_frame SAKC_fluid_flow \
    --output_dir episodic_returns_results/csv/fluid_flow \
    --output_name SAKC_fluid_flow.csv \
    --deterministic_bootstrap
```

At every logged step the script computes, with [rliable](https://github.com/google-research/rliable), the inter-quartile mean over the runs and its confidence band (`--confidence_band`, default `0.95`) from a stratified bootstrap. `--deterministic_bootstrap` seeds the bootstrap so that a data frame always gives the same table.

The table has the columns `timesteps,episodic_returns,lower_confidence_bound,upper_confidence_bound`. An `--output_name` ending in `.csv` gives a comma-separated file; any other name (e.g. `.dat`) gives the space-separated format that the figure sources of the paper read. Both have a header row.

A loop over all twenty algorithm–benchmark pairs is in the [guide](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/EPISODIC_RETURNS.md#all-tables-at-once).

## With Snakemake

```bash
uvx --python 3.12 snakemake -n episodic_returns
```

See [Snakemake workflows](./snakemake.md).
