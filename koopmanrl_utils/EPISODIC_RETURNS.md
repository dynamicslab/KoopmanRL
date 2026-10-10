# Episodic Returns: From Raw Data to pgfplots-Ready Tables

This document describes how the tables behind the episodic-return figures of the paper are
produced: the comparison of LQR, SAC (Q), SAC (V), SKVI and SAKC on the fluid flow, the
Lorenz system and the double well, and the same comparison on the linear system.

---

## Overview

```
Step 1 – run_optimized_experiments.py
    Runs every algorithm on every benchmark, once per seed.
    Raw data: one TensorBoard log per run, with the return of every episode.

Step 2 – dataframe_creator.py
    Collects the logs of one algorithm on one benchmark into a JSON data frame.

Step 3 – process_episodic_returns.py
    Aggregates a data frame over the seeds and writes a table for pgfplots:
    inter-quartile mean and 95% confidence band of the return at every logged step.
```

All commands are run from the root of the repository. One curve of a figure is one
algorithm on one benchmark, so steps 2 and 3 are run once per algorithm and benchmark.

---

## Step 1 — Raw data

**Module:** `koopmanrl_utils.run_optimized_experiments`

```bash
uv run -m koopmanrl_utils.run_optimized_experiments --num_workers 8
```

This makes 495 runs of 50,000 environment steps: five algorithms on four benchmarks, with
25 seeds per benchmark (24 on the linear system). SKVI and SAKC use the tuned
configurations in `configurations/`; the baselines use the defaults of their scripts. The
seeds are those of the SKVI and SAKC runs of the paper. The baseline runs of the paper drew
their seeds at random, and those seeds are not recorded in this repository, so the baselines
are run with the seeds of SKVI and SAKC: their runs are new runs, not repetitions of the
runs of the paper. The options for running a part of the experiments, and the conditions
under which a run repeats exactly, are described at the top of the script.

The runs write into `episodic_returns_results/` (`--output_dir`):

| Algorithm | Logs |
|---|---|
| LQR, SAC (Q), SAC (V) | `episodic_returns_results/runs/<run name>/` |
| SKVI | `episodic_returns_results/runs/SKVI/<run name>/` |
| SAKC | `episodic_returns_results/runs/SAKC/<run name>/` |

The name of a run folder carries the benchmark, the algorithm, the seed and the time at
which the run started:

```
<benchmark>__linear_quadratic_regulator__<seed>__<time>
<benchmark>__sac_continuous_action__<seed>__<time>
<benchmark>__value_based_sac_continuous_action__<seed>__<time>
<benchmark>__soft_koopman_value_iteration__<number of actions>__<training epochs>__<seed>__<time>
<benchmark>__soft_actor_koopman_critic__<seed>__<v_lr>__<q_lr>__<time>
```

Each log holds the scalar `charts/episodic_return`, written at the end of every episode
against the index of the environment step. An episode has 2,000 steps (200 on the linear
system), so a full run logs 25 returns (250 on the linear system).

Three things to check before going on:

- **Incomplete runs.** A run that failed or was interrupted leaves a log with fewer
  returns. The launcher lists the runs that failed; move their folders out of `runs/`
  before step 2, since step 3 needs every run of a data frame to have the same steps.
- **Repeated runs.** A run that was made twice with the same seed has two folders, and
  step 3 counts both.
- **Other runs.** Step 2 takes every run of the chosen algorithm and benchmark that it
  finds, so the folder must hold the runs of this experiment only. The runs of the
  ablation studies in particular must be kept in a different directory.

### Starting from the published data frames

The data frames of the SKVI and SAKC runs of the paper are published at
<https://huggingface.co/datasets/dynamicslab/KoopmanRL-v2>, as
`data/episodic_returns_skvi/<benchmark>.json` and `data/episodic_returns_sakc/<benchmark>.json`.
They have the format of step 2, so for these two algorithms steps 1 and 2 can be skipped:
step 3 reads `<benchmark>.json` from the folder given as `--root_dir`, with
`--data_frame <benchmark>`. The data frames of the baseline runs of the paper are not
published.

From the published data frames, step 3 gives the inter-quartile means of the tables of the
paper to the four decimals of the tables (checked for SKVI on the fluid flow and the double
well and for SAKC on the double well, at the first and the last logged step). The bounds of
the confidence bands agree with those of the paper only up to the noise of the bootstrap,
which was not seeded when the tables of the paper were made.

---

## Step 2 — Data frames

**Module:** `koopmanrl_utils.dataframe_creator`

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

| Flag | Description |
|---|---|
| `--mode` | `Episodic_Returns` for this pipeline. |
| `--system` | Benchmark: `LinearSystem-v0`, `FluidFlow-v0`, `Lorenz-v0` or `DoubleWell-v0`. |
| `--rl_algo` | Algorithm, by the name it has in the run folders (table below). |
| `--target_dir` | Folder that holds the run folders (table below). |
| `--storage_dir` | Folder the data frame is written into. It has to exist. |
| `--output_file` | File name of the data frame. |

| Algorithm | `--rl_algo` | `--target_dir` |
|---|---|---|
| LQR | `linear_quadratic_regulator` | `episodic_returns_results/runs` |
| SAC (Q) | `sac_continuous_action` | `episodic_returns_results/runs` |
| SAC (V) | `value_based_sac_continuous_action` | `episodic_returns_results/runs` |
| SKVI | `soft_koopman_value_iteration` | `episodic_returns_results/runs/SKVI` |
| SAKC | `soft_actor_koopman_critic` | `episodic_returns_results/runs/SAKC` |

The data frame has one entry per run, keyed by the name of the run folder:

```json
"FluidFlow-v0__soft_actor_koopman_critic__5412__0.009423359172870875__0.0017865746944645956__1765986950": {
    "environment": "FluidFlow-v0",
    "rl_algorithm": "soft_actor_koopman_critic",
    "seed": 5412,
    "v_lr": 0.009423359172870875,
    "q_lr": 0.0017865746944645956,
    "episodic_returns": [-79874.671875, -77027.21875, "..."],
    "steps": [1999, 3999, "..."],
    "time": 1765986950
}
```

`v_lr` and `q_lr` are present for SAKC only. `steps` are the indices of the environment
steps at which the episodes ended, counted from zero, and `episodic_returns` the returns
of those episodes.

---

## Step 3 — Tables for pgfplots

**Module:** `koopmanrl_utils.process_episodic_returns`

```bash
mkdir -p episodic_returns_results/csv/fluid_flow

uv run -m koopmanrl_utils.process_episodic_returns \
    --root_dir episodic_returns_results/frames \
    --data_frame SAKC_fluid_flow \
    --output_dir episodic_returns_results/csv/fluid_flow \
    --output_name SAKC_fluid_flow.csv \
    --deterministic_bootstrap
```

| Flag | Description |
|---|---|
| `--root_dir` | Folder that holds the data frame. |
| `--data_frame` | Name of the data frame, without `.json`. |
| `--output_dir` | Folder the table is written into. It has to exist. |
| `--output_name` | File name of the table. A name ending in `.csv` gives a comma-separated file; any other name, such as `SAKC_fluid_flow.dat`, gives a space-separated file, which is the format of the tables that the figure sources of the paper read. Both have a header row with the names of the columns. |
| `--confidence_band` | Level of the confidence band (default `0.95`). |
| `--deterministic_bootstrap` | Seed the bootstrap, so that the same data frame gives the same table on every call. Without it the bounds of the band change slightly from call to call. |

At every logged step the script takes the returns of all runs of the data frame and
computes, with [rliable](https://github.com/google-research/rliable),

- the inter-quartile mean over the runs, and
- its confidence band, as the percentile interval of a stratified bootstrap with 50,000
  resamples.

The table has a header row and one row per logged step, with four decimals:

```
timesteps,episodic_returns,lower_confidence_bound,upper_confidence_bound
```

| Column | Content |
|---|---|
| `timesteps` | Index of the environment step at which the episodes ended, counted from zero. |
| `episodic_returns` | Inter-quartile mean of the return over the runs. |
| `lower_confidence_bound`, `upper_confidence_bound` | Bounds of the confidence band. |

The bootstrap takes about a minute per table on the benchmarks with 25 logged steps, and
longer in proportion on the linear system, which logs 250.

---

## All tables at once

```bash
OUT=episodic_returns_results
mkdir -p $OUT/frames

for system in LinearSystem-v0:linear_system FluidFlow-v0:fluid_flow Lorenz-v0:lorenz DoubleWell-v0:double_well; do
    for algorithm in \
        LQR:linear_quadratic_regulator:runs \
        SAC_Q:sac_continuous_action:runs \
        SAC_V:value_based_sac_continuous_action:runs \
        SKVI:soft_koopman_value_iteration:runs/SKVI \
        SAKC:soft_actor_koopman_critic:runs/SAKC; do
        IFS=: read env_id benchmark <<< "$system"
        IFS=: read name rl_algo folder <<< "$algorithm"
        mkdir -p $OUT/csv/$benchmark

        uv run -m koopmanrl_utils.dataframe_creator \
            --mode Episodic_Returns --system $env_id --rl_algo $rl_algo \
            --target_dir $OUT/$folder --storage_dir $OUT/frames \
            --output_file ${name}_${benchmark}.json

        uv run -m koopmanrl_utils.process_episodic_returns \
            --root_dir $OUT/frames --data_frame ${name}_${benchmark} \
            --output_dir $OUT/csv/$benchmark --output_name ${name}_${benchmark}.csv \
            --deterministic_bootstrap
    done
done
```

This writes the twenty tables `episodic_returns_results/csv/<benchmark>/<ALGORITHM>_<benchmark>.csv`,
for example `csv/fluid_flow/SAKC_fluid_flow.csv`. The figure sources of the paper read
their tables from `data/episodic_returns/<benchmark>/<ALGORITHM>_<benchmark>.dat`, so the
`csv/` folder has the layout of that folder.

---

## Reading the tables in pgfplots

The figure sources of the paper address the columns by name. A curve with its band, with
the axis settings of those sources:

```latex
\documentclass[tikz]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.17}

\begin{document}
\begin{tikzpicture}
  \begin{axis}[
      xmode=log, xmin=1999, xmax=32000,
      xtick={2000, 4000, 8000, 16000, 32000},
      xticklabels={2000, 4000, 8000, 16000, 32000},
      xlabel={Steps in Environment}, ylabel={Episodic Returns (IQM)},
      legend pos=south east,
    ]
    \addplot[name path=lower, draw=none, forget plot]
      table[col sep=comma, x=timesteps, y=lower_confidence_bound] {SAKC_fluid_flow.csv};
    \addplot[name path=upper, draw=none, forget plot]
      table[col sep=comma, x=timesteps, y=upper_confidence_bound] {SAKC_fluid_flow.csv};
    \addplot[blue, opacity=0.2, forget plot] fill between[of=lower and upper];
    \addplot[blue, thick]
      table[col sep=comma, x=timesteps, y=episodic_returns] {SAKC_fluid_flow.csv};
    \addlegendentry{SAKC}
  \end{axis}
\end{tikzpicture}
\end{document}
```

- `col sep=comma` is what a comma-separated table needs in addition to the column names.
  A space-separated `.dat` table is read without it, as in the figure sources of the paper.
- `timesteps` counts from zero, so the first episode of 2,000 steps ends at 1999; the axis
  therefore starts at `xmin=1999`.
- The tables cover all 50,000 steps. The figures of the paper show the steps up to 32,000,
  and the steps from 28,000 to 32,000 again in a second panel (`xmin=28000`, `xmax=32000`).
