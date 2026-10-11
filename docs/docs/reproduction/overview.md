---
id: overview
sidebar_position: 1
title: Overview
---

# Reproducing the Paper

Three pipelines produce the tables that the figures of the paper are drawn from. Each one can be run script by script, or as a [Snakemake workflow](./snakemake.md) that tracks which results exist and remakes the missing ones.

| Result | Launcher | Runs | Run list | Guide |
|--------|----------|------|----------|-------|
| [Episodic returns](./episodic-returns.md) of LQR, SAC (Q), SAC (V), SKVI and SAKC | `koopmanrl_utils.run_optimized_experiments` | 495 | `configurations/episodic_returns.json` | [`EPISODIC_RETURNS.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/EPISODIC_RETURNS.md) |
| [Ablations](./ablations.md) of SKVI and SAKC over 6 × 6 grids | `koopmanrl_utils.run_ablations` | 1,440 | `configurations/ablations.json` | [`ABLATIONS.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/ABLATIONS.md) |
| [t-SNE](./tsne.md) of the Koopman tensors (supplementary material) | `koopmanrl_utils.tsne_koopman_tensor` | no training runs | `configurations/tsne.json` | [`TSNE.md`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/TSNE.md) |

Each RL run is 50,000 environment steps. The ablation campaign alone takes several days of computing time, so start with a dry run (`--dry_run` for the launchers, `-n` for Snakemake) and a short selection.

All commands are run from the root of the repository.

## Shape of the RL pipelines

The episodic-return and ablation pipelines share three steps:

```
1. launcher            one TensorBoard log per run (charts/episodic_return)
2. dataframe_creator   the logs of one algorithm on one benchmark -> JSON data frame
3. process_*           a data frame -> table for pgfplots (.csv, or .dat as in the paper sources)
```

The JSON run lists in `configurations/` are read by both the launchers and the Snakemake workflow, so the algorithms, benchmarks, seeds and grids are defined in one place.

## Published data frames

The data frames of the SKVI and SAKC runs of the paper (episodic returns and ablations) are published at [huggingface.co/datasets/dynamicslab/KoopmanRL-v2](https://huggingface.co/datasets/dynamicslab/KoopmanRL-v2). They have the format of step 2, so step 3 can be run on them directly. The data frames of the baseline runs are not published. The guides above describe the files and what has been checked against the tables of the paper.

## What repeats exactly

Every run is seeded. Per the launchers' docstrings, two runs with the same seed log the same returns only on the same machine, with the same library versions and the **same number of numerical threads** (`OMP_NUM_THREADS`, which the launchers leave to the environment). SKVI and SAKC solve their least-squares problems with the LAPACK driver `gelsd` so that the Koopman tensor is the same bit for bit in every process; see [Least-squares driver](../koopman-tensor/least-squares-driver.md).

The seeds of the baseline runs of the paper were drawn at random and are not part of `configurations/episodic_returns.json`; by default the baselines are run with the seeds of SKVI and SAKC, so their runs are new runs rather than repetitions.
