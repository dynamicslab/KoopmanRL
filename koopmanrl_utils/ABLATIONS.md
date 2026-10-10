# Ablations: From Raw Data to pgfplots-Ready Tables

This document describes how the tables behind the two ablation figures of the paper are
produced: the return of SKVI over the number of actions and the number of training epochs,
and the return of SAKC over the learning rates of its value network and of its Q-networks,
each on the linear system, the fluid flow, the Lorenz system and the double well.

---

## Overview

```
Step 1 – run_ablations.py
    Runs each algorithm at every point of its 6 x 6 grid on every benchmark, once per seed.
    Raw data: one TensorBoard log per run, with the return of every episode.

Step 2 – dataframe_creator.py
    Collects the logs of one ablation on one benchmark into a JSON data frame.

Step 3 – process_skvi_ablations.py, process_sakc_ablations.py
    Aggregates a data frame over the seeds and writes a table for pgfplots:
    the inter-quartile mean of the last returns at every grid point.
```

All commands are run from the root of the repository. One surface of a figure is one
algorithm on one benchmark, so steps 2 and 3 are run once per algorithm and benchmark.

---

## Step 1 — Raw data

**Module:** `koopmanrl_utils.run_ablations`

```bash
uv run -m koopmanrl_utils.run_ablations --num_workers 8
```

This makes 1,440 runs of 50,000 environment steps: two algorithms on four benchmarks, at
the 36 points of a grid, with five seeds per grid point (1, 21, 41, 61, 81). The campaign
takes several days of computing time. `--dry_run` prints the commands without running them.

| Algorithm | Swept flag | Values | Axis label in the figure of the paper |
|---|---|---|---|
| SKVI | `--num_actions` | 71, 81, 91, 101, 111, 121 | No. of Actions |
| SKVI | `--num_training_epochs` | 75, 100, 125, 150, 175, 200 | Training Epochs |
| SAKC | `--v_lr` | 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05 | Value Network Learning Rate |
| SAKC | `--q_lr` | 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05 | Policy Network Learning Rate |

`--v_lr` is the learning rate of the optimiser of the value network and `--q_lr` the
learning rate of the optimiser of the Q-networks. The actor has an optimiser of its own.
Its learning rate, `--policy_lr`, is not swept: the launcher does not pass it, and it is
not among the entries that SAKC reads from a configuration file, so it stays at the default
of the script, 0.0003. Every run takes its other hyperparameters from the tuned
configuration of its benchmark in `configurations/`. What is known about the
hyperparameters of the runs of the paper, and the conditions under which a run repeats
exactly, are described at the top of the script.

A part of the campaign is selected with `--algorithms`, `--environments`, `--seeds` and the
values of the grid. A short test, with a directory of its own so that its short runs stay
apart from the runs of the campaign:

```bash
uv run -m koopmanrl_utils.run_ablations --algorithms skvi --environments LinearSystem-v0 \
    --seeds 1 21 --num_actions 71 81 --num_training_epochs 75 --total_timesteps 2200 \
    --output_dir ablation_results/test
```

The same selection without `--total_timesteps` and `--output_dir` makes the runs that are
missing after an interruption.

The runs of the campaign write into `ablation_results/` (`--output_dir`):

| Algorithm | Logs |
|---|---|
| SKVI | `ablation_results/runs/SKVI/<run name>/` |
| SAKC | `ablation_results/runs/SAKC/<run name>/` |

The name of a run folder carries the benchmark, the algorithm, the grid point, the seed and
the time at which the run started:

```
<benchmark>__soft_koopman_value_iteration__<number of actions>__<training epochs>__<seed>__<time>
<benchmark>__soft_actor_koopman_critic__<seed>__<v_lr>__<q_lr>__<time>
```

Step 2 reads the grid point and the seed of a run from this name. Each log holds the scalar
`charts/episodic_return`, written at the end of every episode against the index of the
environment step. An episode has 2,000 steps (200 on the linear system), so a full run logs
25 returns (250 on the linear system).

Three things to check before going on:

- **Incomplete runs.** A run that failed or was interrupted leaves a log with fewer
  returns, and step 3 would take the last returns of that log as they are. The launcher
  lists the runs that failed; move their folders out of `runs/` before step 2.
- **Repeated runs.** A grid point that was run twice with the same seed has two folders,
  and step 3 counts both.
- **Other runs.** Step 2 takes every run of the chosen benchmark that it finds in a folder,
  whatever its algorithm, so `runs/SKVI` and `runs/SAKC` must hold the runs of the ablations
  only. The runs of the episodic-return figures use the same two folder names and must be
  kept in a different directory.

### Starting from the published data frames

The data frames of the ablations of the paper are published at
<https://huggingface.co/datasets/dynamicslab/KoopmanRL-v2>, as
`data/ablation_skvi/<benchmark>.json` and `data/ablation_sakc/<benchmark>.json`. They have
the format of step 2: placed in `ablation_results/frames/ablation_skvi/` and
`ablation_results/frames/ablation_sakc/`, they take the place of steps 1 and 2. Their runs
were made in May 2025, and the data frames do not record the hyperparameters that are not
swept.

The data frame of SKVI on the double well holds repeated runs.
`data/ablation_skvi/double_well.json` is about twice the size of the data frames of SKVI on
the fluid flow and on the Lorenz system (578,112 bytes against 293,079 and 269,127), and
grid points with two runs per seed were found in it, the two runs started 16.6 hours apart
in each of the five cases seen.
Step 3 counts every run of a grid point. From the size of the file we infer, without having
counted the runs, that most or all grid points were run twice; the table of the paper for
this benchmark then pools about twice as many runs per grid point as five seeds give. The
launcher makes one run per grid point and seed, so a rerun does not compute the same
estimate for SKVI on the double well.

---

## Step 2 — Data frames

**Module:** `koopmanrl_utils.dataframe_creator`

```bash
mkdir -p ablation_results/frames/ablation_skvi

uv run -m koopmanrl_utils.dataframe_creator \
    --mode SKVI_Ablations \
    --system LinearSystem-v0 \
    --target_dir ablation_results/runs/SKVI \
    --storage_dir ablation_results/frames/ablation_skvi \
    --output_file linear_system.json
```

| Flag | Description |
|---|---|
| `--mode` | `SKVI_Ablations` or `SAKC_Ablations` (table below). |
| `--system` | Benchmark: `LinearSystem-v0`, `FluidFlow-v0`, `Lorenz-v0` or `DoubleWell-v0`. |
| `--target_dir` | Folder that holds the run folders (table below). |
| `--storage_dir` | Folder the data frame is written into. It has to exist. |
| `--output_file` | File name of the data frame. |

| Ablation | `--mode` | `--target_dir` |
|---|---|---|
| SKVI | `SKVI_Ablations` | `ablation_results/runs/SKVI` |
| SAKC | `SAKC_Ablations` | `ablation_results/runs/SAKC` |

The data frame has one entry per run, keyed by the name of the run folder. For SKVI:

```json
"LinearSystem-v0__soft_koopman_value_iteration__71__75__1__1791586319": {
    "environment": "LinearSystem-v0",
    "rl_algorithm": "soft_koopman_value_iteration",
    "seed": 1,
    "num_actions": 71,
    "num_training_epochs": 75,
    "episodic_returns": [-964.5636596679688, -810.7860107421875, "..."],
    "steps": [199, 399, "..."],
    "time": 1791586319
}
```

For SAKC:

```json
"LinearSystem-v0__soft_actor_koopman_critic__1__0.0001__0.0005__1791586044": {
    "environment": "LinearSystem-v0",
    "rl_algorithm": "soft_actor_koopman_critic",
    "seed": 1,
    "v_lr": 0.0001,
    "q_lr": 0.0005,
    "episodic_returns": [-45919.09375, -72029.734375, "..."],
    "steps": [199, 399, "..."],
    "time": 1791586044
}
```

`steps` are the indices of the environment steps at which the episodes ended, counted from
zero, and `episodic_returns` the returns of those episodes.

---

## Step 3 — Tables for pgfplots

**Modules:** `koopmanrl_utils.process_skvi_ablations`, `koopmanrl_utils.process_sakc_ablations`

```bash
mkdir -p ablation_results/csv/SKVI/window_3

uv run -m koopmanrl_utils.process_skvi_ablations \
    --root_dir ablation_results/frames/ablation_skvi \
    --data_frame linear_system \
    --output_dir ablation_results/csv/SKVI/window_3 \
    --output_name linear_system_ablation.csv \
    --smoothing_window 3
```

The two scripts take the same flags:

| Flag | Description |
|---|---|
| `--root_dir` | Folder that holds the data frame. |
| `--data_frame` | Name of the data frame, without `.json`. |
| `--output_dir` | Folder the table is written into. It has to exist. |
| `--output_name` | File name of the table. A name ending in `.csv` gives a comma-separated table with named columns; any other name, such as one ending in `.dat`, gives the layout of the tables of the paper, which its figure sources read unchanged. |
| `--smoothing_window` | Number `w` of returns taken from the end of every run (default `1`). The figures of the paper use `3`. |

For every point of the grid the script takes the runs of the data frame that were made at
that point, and from each of them its last `w` returns. The returns of all these runs are
pooled, and the table holds their inter-quartile mean, computed with
[rliable](https://github.com/google-research/rliable): of the pooled returns, a quarter,
rounded down, is dropped at either end, and the rest is averaged. With five seeds this is

| `w` | Pooled returns | Dropped at either end | Averaged |
|---|---|---|---|
| 1 | 5 | 1 | 3 |
| 3 | 15 | 3 | 9 |
| 5 | 25 | 6 | 13 |

At a grid point with two runs per seed, as they were found in the published data frame of
SKVI on the double well (step 1), the pool is twice as large: 10, 30 and 50 returns, of
which 2, 7 and 12 are dropped at either end.

According to the notes kept with the tables of the paper, its figures use `w = 3`, and the
tables for `w = 1` and `w = 5` were kept to check how much the surfaces depend on this
choice. The window is a choice of the post-processing; it does not enter the runs.

The table has 36 rows, one per grid point, with four decimals. A `.csv` table starts with a
header row of column names:

```
num_actions,num_training_epochs,episodic_return      (SKVI)
v_lr,q_lr,episodic_return                            (SAKC)
```

| Column | Content |
|---|---|
| `num_actions`, `num_training_epochs` | Grid point of SKVI. |
| `v_lr`, `q_lr` | Grid point of SAKC. |
| `episodic_return` | Inter-quartile mean of the last `w` returns of the runs at the grid point. |

The rows come in six blocks of six: the first column is constant within a block, and the
second column runs through its six values. Both columns ascend in the table of SKVI; in the
table of SAKC `v_lr` ascends from block to block and `q_lr` descends within a block. This
is the order of the tables of the paper. A grid point without runs has `nan` as its return;
a table made before the campaign is complete therefore has all 36 rows.

A `.dat` table has the same rows in the layout of the tables of the paper: the header line
`x y z`, the three columns separated by spaces, and an empty line after every block of six
rows.

---

## All tables at once

```bash
OUT=ablation_results
WINDOW=3

for algorithm in skvi:SKVI sakc:SAKC; do
    IFS=: read name folder <<< "$algorithm"
    mkdir -p $OUT/frames/ablation_$name $OUT/csv/$folder/window_$WINDOW

    for system in LinearSystem-v0:linear_system FluidFlow-v0:fluid_flow Lorenz-v0:lorenz DoubleWell-v0:double_well; do
        IFS=: read env_id benchmark <<< "$system"

        uv run -m koopmanrl_utils.dataframe_creator \
            --mode ${folder}_Ablations --system $env_id \
            --target_dir $OUT/runs/$folder --storage_dir $OUT/frames/ablation_$name \
            --output_file $benchmark.json

        uv run -m koopmanrl_utils.process_${name}_ablations \
            --root_dir $OUT/frames/ablation_$name --data_frame $benchmark \
            --output_dir $OUT/csv/$folder/window_$WINDOW \
            --output_name ${benchmark}_ablation.csv \
            --smoothing_window $WINDOW
    done
done
```

This writes the eight tables `ablation_results/csv/<SKVI or SAKC>/window_3/<benchmark>_ablation.csv`,
which is the folder layout of the tables in the sources of the paper. `WINDOW=1` and
`WINDOW=5` give the tables of the other two windows. With `_ablation.dat` in place of
`_ablation.csv` the loop writes the tables as the figure sources of the paper read them.

---

## Reading the tables in pgfplots

One panel of the SKVI figure, with the axis options of the figure of the paper, compiled in
`ablation_results/csv/`:

```latex
\documentclass[crop,tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

\begin{document}
\begin{tikzpicture}
  \begin{axis}[
      width=10cm, height=9cm,
      grid=major, tick align=outside, tick pos=left, opacity=0.6,
      title={Linear System},
      xlabel={No. of Actions}, ylabel={Training Epochs}, zlabel={Episodic Return},
      view={210}{30},
      xtick={121, 111, 101, 91, 81, 71},
      ytick={200, 175, 150, 125, 100, 75},
      colormap/viridis,
    ]
    \addplot3+[scatter, surf, mesh/rows=6, unbounded coords=jump]
      table[col sep=comma, x=num_actions, y=num_training_epochs, z=episodic_return]
      {SKVI/window_3/linear_system_ablation.csv};
  \end{axis}
\end{tikzpicture}
\end{document}
```

One panel of the SAKC figure, on its logarithmic axes:

```latex
\documentclass[crop,tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

\begin{document}
\begin{tikzpicture}
  \begin{axis}[
      width=10cm, height=9cm,
      grid=major, tick align=outside, tick pos=left, opacity=0.6,
      minor tick length=0pt,
      title={Linear System},
      xlabel={\begin{tabular}{c} Value Network \\ Learning Rate \end{tabular}},
      ylabel={\begin{tabular}{c} Policy Network \\ Learning Rate \end{tabular}},
      zlabel={Episodic Return},
      view={210}{30},
      xmode=log, xmin=0.0001, xmax=0.05,
      ymode=log, ymin=0.0001, ymax=0.05,
      xtick={0.0001, 0.001, 0.01},
      xticklabels={$10^{-4}$, $10^{-3}$, $10^{-2}$},
      extra x ticks={0.0005, 0.005, 0.05},
      extra x tick style={xshift=7.5pt, yshift=-2pt},
      ytick={0.0005, 0.005, 0.05},
      yticklabels={$5 \cdot 10^{-4}$, $5 \cdot 10^{-3}$, $5 \cdot 10^{-2}$},
      extra y ticks={0.0001, 0.001, 0.01},
      extra y tick style={xshift=-8.5pt, yshift=-6pt},
      yticklabel style={xshift=3pt},
      colormap/viridis,
    ]
    \addplot3+[mesh, scatter, surf, mesh/rows=6, unbounded coords=jump]
      table[col sep=comma, x=v_lr, y=q_lr, z=episodic_return]
      {SAKC/window_3/linear_system_ablation.csv};
  \end{axis}
\end{tikzpicture}
\end{document}
```

- `mesh/rows=6` tells pgfplots the shape of the grid: the table is six blocks of six rows.
  Without it pgfplots joins the 36 points by a line and draws no surface. On this grid
  `mesh/cols=6` has the same effect, because the grid is square; for a table with six
  values in the first column and fewer in the second, `mesh/rows=6` is the one that holds.
- `unbounded coords=jump` makes pgfplots leave out the patches that have a corner with a
  `nan` return, in a table made before the campaign is complete. Without it pgfplots drops
  the rows with `nan` and takes the remaining points for a grid of another shape: it then
  joins points that are not neighbours, without an error, or draws the marks only, or
  stops with an error, depending on which rows are missing. On a complete table the option
  changes nothing.
- A `.dat` table needs neither the column names nor `mesh/rows`: pgfplots reads its columns
  by position and takes the shape of the grid from the empty lines. The figure sources of
  the paper read it with

  ```latex
  \addplot3+ [scatter,surf] table {SKVI/window_3/linear_system_ablation.dat};
  \addplot3+ [mesh,scatter,surf] table {SAKC/window_3/linear_system_ablation.dat};
  ```

  in place of the `\addplot3` commands above. A complete `.csv` table and the `.dat` table
  of the same data frame give the same picture. A `.dat` table with `nan` returns needs
  `unbounded coords=jump` as well, added to the options in the square brackets.
- The order of the rows does not change the surface, but it is the order in which pgfplots
  paints the patches, and the picture changes with it where the surface hides a part of
  itself. The tables have the row order of the tables of the paper, so they give its
  pictures. `z buffer=sort` among the axis options makes the picture independent of the
  order.
