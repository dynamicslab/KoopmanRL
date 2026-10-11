---
id: snakemake
sidebar_position: 5
title: Snakemake Workflows
---

# Snakemake Workflows

`workflow/` holds [Snakemake](https://snakemake.readthedocs.io) workflows for the three pipelines. A workflow knows which results exist, makes the missing ones, and makes a run again when it was interrupted. Every run gets a directory of its own, so the campaigns cannot mix. The full reference — settings, resuming, what triggers a rerun, verified commands — is [`workflow/README.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/workflow/README.md).

## Requirements

Snakemake is not a project dependency: the project is pinned to Python 3.10, and Snakemake 8+ needs Python 3.11 or newer. Run it as a separate tool with `uvx`; its jobs call the project code through the project environment with `uv run --no-sync`, so sync it first:

```bash
uv sync                                   # once: creates .venv/
uvx --python 3.12 snakemake --version
```

To pin a version on every machine, write `snakemake@<version>` in place of `snakemake`. Run all commands from the repository root.

## Targets

| Target | Jobs | Result |
|--------|------|--------|
| `episodic_returns` | 495 runs, 20 data frames, 20 tables | [Episodic-return](./episodic-returns.md) tables |
| `ablations` | 1,440 runs, 8 data frames, 8 tables | [Ablation](./ablations.md) tables |
| `tsne` | 4 identification jobs, 1 embedding job | [t-SNE](./tsne.md) tables |
| `all` | all of the above (1,935 RL runs) | everything |

A call without a target makes nothing and lists the targets; `all` has to be named explicitly. Always dry-run first:

```bash
uvx --python 3.12 snakemake -n episodic_returns
uvx --python 3.12 snakemake -n ablations
uvx --python 3.12 snakemake -n tsne
```

## Running locally

```bash
uvx --python 3.12 snakemake --cores 8 --resources mem_mb=16000 --keep-going episodic_returns
uvx --python 3.12 snakemake --cores 1 tsne
```

A run job takes one core. `--resources mem_mb=...` caps the memory that running jobs declare (4000 MB for an SKVI run, 2000 MB for other runs). `--resources` takes every word that follows it, so put the target before it or another option in between.

Results go to `results/<workflow>/` (`runs/`, `frames/`, `tables/`, `logs/`; `tensors/` for the t-SNE). Give the same command again to resume: finished jobs are not repeated, and failed or interrupted runs are made again. Start every call for one set of results from the same directory, since Snakemake keeps its records in `.snakemake/` there.

## Settings

The run lists come from `configurations/episodic_returns.json`, `configurations/ablations.json` and `configurations/tsne.json`, which the launchers read as well. Override them with `--config` or `--configfile`, for example a short test:

```bash
uvx --python 3.12 snakemake --cores 2 episodic_returns --config results_dir=/tmp/smoke \
    'episodic_returns={only_benchmarks: [LinearSystem-v0], only_algorithms: [lqr, sac_q, sac_v],
                       seeds: {LinearSystem-v0: [4430, 2738]}, total_timesteps: 2200}'
```

The settings are checked before any job is made: an unknown key inside a workflow's block is refused (a misspelled block name itself is not noticed). The keys of each workflow are listed in the README.

## Clusters

With [snakemake-executor-plugin-slurm](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) every job is submitted as a Slurm job with the memory it declares:

```bash
uvx --python 3.12 --with snakemake-executor-plugin-slurm snakemake episodic_returns \
    --executor slurm --jobs 100 \
    --default-resources slurm_account=<account> slurm_partition=<partition> runtime=<minutes>
```

This command has only been dry-run, not run on a cluster. The repository, its `.venv/`, `uv` and Snakemake must be reachable under the same paths on the compute nodes. See the [README](https://github.com/dynamicslab/KoopmanRL/blob/main/workflow/README.md) for the details and for the ablations.
