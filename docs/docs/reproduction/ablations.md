---
id: ablations
sidebar_position: 3
title: Ablations
---

# Ablations

The two ablation figures show the return of SKVI over the number of actions and training epochs, and of SAKC over the learning rates of its value network and Q-networks, on all four benchmarks. The full walkthrough, including the data published for the paper and what is known about its runs, is in [`koopmanrl_utils/ABLATIONS.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/ABLATIONS.md).

## Step 1 — Runs

```bash
uv run -m koopmanrl_utils.run_ablations --dry_run          # print the commands only
uv run -m koopmanrl_utils.run_ablations --num_workers 8
```

The campaign is 1,440 runs: two algorithms on four benchmarks, at the 36 points of a 6 × 6 grid, with five seeds per grid point (1, 21, 41, 61, 81). It takes several days of computing time. The grids and seeds are read from `configurations/ablations.json`; every run takes its other hyperparameters from the tuned configuration of its benchmark.

| Algorithm | Swept flag | Default values |
|-----------|------------|----------------|
| SKVI | `--num_actions` | 71 81 91 101 111 121 |
| SKVI | `--num_training_epochs` | 75 100 125 150 175 200 |
| SAKC | `--v_lr` | 0.0001 0.0005 0.001 0.005 0.01 0.05 |
| SAKC | `--q_lr` | 0.0001 0.0005 0.001 0.005 0.01 0.05 |

The actor's learning rate (`--policy_lr`) is not swept and stays at the SAKC default.

The launcher takes `--environments`, `--algorithms` (`skvi`, `sakc`), `--seeds`, the four grid flags above, `--total_timesteps`, `--output_dir` (default `ablation_results`), `--num_workers` and `--dry_run`. A short test in a directory of its own:

```bash
uv run -m koopmanrl_utils.run_ablations --algorithms skvi --environments LinearSystem-v0 \
    --seeds 1 21 --num_actions 71 81 --num_training_epochs 75 --total_timesteps 2200 \
    --output_dir ablation_results/test
```

Running the same selection again without `--total_timesteps` and `--output_dir` makes the runs that are missing after an interruption. Logs go to `ablation_results/runs/SKVI/` and `ablation_results/runs/SAKC/`. These folder names are the same as those of the episodic-return runs, so keep the two campaigns in different output directories.

## Step 2 — Data frames

```bash
mkdir -p ablation_results/frames/ablation_skvi

uv run -m koopmanrl_utils.dataframe_creator \
    --mode SKVI_Ablations \
    --system LinearSystem-v0 \
    --target_dir ablation_results/runs/SKVI \
    --storage_dir ablation_results/frames/ablation_skvi \
    --output_file linear_system.json
```

Use `--mode SAKC_Ablations` with `--target_dir ablation_results/runs/SAKC` for SAKC. The grid point and the seed of a run are read from the name of its folder.

## Step 3 — Tables

```bash
mkdir -p ablation_results/csv/SKVI/window_3

uv run -m koopmanrl_utils.process_skvi_ablations \
    --root_dir ablation_results/frames/ablation_skvi \
    --data_frame linear_system \
    --output_dir ablation_results/csv/SKVI/window_3 \
    --output_name linear_system_ablation.csv \
    --smoothing_window 3
```

`koopmanrl_utils.process_sakc_ablations` takes the same flags. For every grid point the script pools the last `--smoothing_window` returns of every run at that point and writes their inter-quartile mean. The figures of the paper use a window of 3.

The table has 36 rows, one per grid point, in six blocks of six. The format follows the file name:

| `--output_name` | Layout |
|-----------------|--------|
| `*.csv` | Comma-separated with a header row: `num_actions,num_training_epochs,episodic_return` (SKVI) or `v_lr,q_lr,episodic_return` (SAKC). Read in pgfplots with `col sep=comma` and `mesh/rows=6`. |
| anything else, e.g. `*.dat` | The layout of the tables of the paper: header `x y z`, space-separated, an empty line after every block of six rows. The figure sources of the paper read it unchanged. |

A grid point without runs gets `nan`, so a table made before the campaign is complete still has all 36 rows. A loop over both algorithms and all benchmarks, and the pgfplots code of both figures, are in the [guide](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/ABLATIONS.md#all-tables-at-once).

## With Snakemake

```bash
uvx --python 3.12 snakemake -n ablations
```

Without `-n` the target `ablations` starts the full campaign. See [Snakemake workflows](./snakemake.md).
