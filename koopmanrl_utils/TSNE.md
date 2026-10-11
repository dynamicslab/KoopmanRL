# t-SNE of the Koopman Tensors: From Raw Data to pgfplots-Ready Tables

This document describes how the tables behind the figure "t-distributed stochastic neighbour
embedding of the Koopman tensors projected onto a common basis" of the electronic
supplementary material of the paper are produced: Koopman tensors of the linear system, the
fluid flow, the Lorenz system and the double well, embedded together in two dimensions. It
also records what is known about how the published figure was made, and what the embedding
does and does not show.

---

## Overview

```
Step 1 – tsne_koopman_tensor.py (identification)
    Identifies a set of Koopman tensors per benchmark from random-agent data.
    Raw data: tensors_<benchmark>.npz, the tensors as identified.

Step 2 – tsne_koopman_tensor.py (common basis)
    Writes every tensor in the monomial dictionaries of the largest orders of the set.

Step 3 – tsne_koopman_tensor.py (embedding)
    Embeds all tensors together by t-SNE and writes the tables for pgfplots:
    one row per tensor with its two coordinates.
```

One call runs the three steps. All commands are run from the root of the repository. The
numbers in this document are those of the runs described in it, on a shared two-core
machine with torch 2.9.1 (LAPACK of MKL 2024.2) and scikit-learn 1.7.2; where a number
depends on the number of numerical threads, the range over one and two threads is given.

The identification and the embedding can also be run as one Snakemake workflow, which keeps
them apart: one job per benchmark identifies and stores its tensors (a call of the script
with `--identify_only`, which ends before the embedding), and one job embeds the stored
tensors of all benchmarks (`--embed_only`). Another embedding is then made from the same
tensors, and only the missing tensors are identified:

```bash
uvx --python 3.12 snakemake --cores 1 tsne
```

The commands, the settings and the layout of its results are described in
[`workflow/README.md`](../workflow/README.md). Its jobs run with one thread, and with one
core it gave the tensors and the tables of one call of the script with `OMP_NUM_THREADS=1`,
byte for byte.

---

## Step 1 — Raw data

**Module:** `koopmanrl_utils.tsne_koopman_tensor`

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor
```

This identifies 161 tensors per benchmark (1 min 23 s on one core, peak memory 0.8 GB) and
goes on with steps 2 and 3. Everything is written into `tsne_koopman_tensor_results/`
(`--output_dir`), which git ignores.

- **Data.** For each benchmark one set of random-agent trajectories is collected with the
  calls of `koopmanrl.soft_koopman_value_iteration.generate_koopman_tensor`. Every
  identification budget of the sweep uses the first steps of the first trajectories of that
  set.
- **Identification.** Each tensor `K[i, j, z]` (predicted state monomial `i`, state monomial
  `j`, action monomial `z`) is the ordinary-least-squares regression of the package's
  `KoopmanTensor` class with monomial dictionaries. The regression is solved by
  `torch.linalg.lstsq` with the LAPACK driver `gelsd` (`--lstsq_driver`), which returns the
  same tensor in every call. The package calls the solver without naming a driver, which
  is `gelsy`; that call is available as `--lstsq_driver gelsy` and does not return the same
  tensor in every call (see "Reproducibility").
- **Raw data.** The tensors of a benchmark are stored in `tensors_<benchmark>.npz`, with
  their rows of the sweep, the size of the data set and the driver of the solver, as soon
  as the benchmark is done. `--resume` picks up an interrupted run: benchmarks whose
  tensors are already stored for the same sweep and the same driver are read instead of
  identified.

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor --resume
```

Two sweeps are available (`--sweep`):

| Sweep | Tensors per benchmark |
|---|---|
| `inferred_layout` (default) | 161 from one data seed: 121 with state order 2 and action order 2 on a grid of 11 numbers of trajectories (10 to 110) by 11 numbers of steps per trajectory (50 to 550), 25 on a second such grid of 5 by 5, and 15 on the grid of state orders 1 to 4 by action orders 1 to 4 without (4, 4), at 100 trajectories of 300 steps. This is the layout that the published coordinates point to; see "How the published figure was made". |
| `orders` | 128: state orders 1 to 4 by action orders 1 to 4 by 8 data seeds, at the tuned SKVI identification budget of the benchmark. This is the reading of the text of the paper. |

The `orders` sweep (5 minutes, peak memory 0.9 GB) is run into its own directory with

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor --sweep orders --output_dir tsne_koopman_tensor_results/orders
```

---

## Step 2 — Common basis

Tensors of different dictionary orders have different shapes. Each tensor is written in the
monomial dictionaries of the largest state order and the largest action order of the sweep
(35 state monomials and 5 action monomials for orders 4 and 4): every coefficient goes to
the positions of its three monomials, and all other coefficients are zero. Evaluated through
the common dictionaries, the tensor gives the same prediction for its own dictionary
functions; the tests check this.

The double well has two state coordinates and the other benchmarks three. Its coordinates
are taken as the first two of a state whose third coordinate does not appear
(`--double_well_coordinates`), so all its coefficients of monomials with the third
coordinate are zero.

The tensors in the common basis are flattened to 6,125 coefficients each.

---

## Step 3 — Embedding and tables

All tensors of all benchmarks are embedded together by `sklearn.manifold.TSNE` with its
defaults (perplexity 30, PCA initialisation, Barnes-Hut) and `random_state=42`. Steps 2 and
3 can be repeated from the stored tensors, with other settings if wanted:

```bash
uv run -m koopmanrl_utils.tsne_koopman_tensor --embed_only
uv run -m koopmanrl_utils.tsne_koopman_tensor --embed_only --units box
```

`--embed_only` rewrites the files of `--output_dir`: the second command replaces the default
embedding by the one in box units, and the first one restores it. The arguments of the
identification (`--sweep` and its arguments, `--linear_system_seed`, `--lstsq_driver`) have
to be those the tensors were identified with, since they are written into `settings.json`
next to the embedding: stored tensors of another sweep, of another linear system or of
another driver are refused with a message that names the difference, and nothing is
written.

| File | Content |
|---|---|
| `linear_system_tsne.csv`, `fluid_flow_tsne.csv`, `lorenz_tsne.csv`, `double_well_tsne.csv` | One row per tensor of the benchmark. These are the file names that the figure source of the paper reads. |
| `tsne.csv` | All rows, with the columns `benchmark` and `class` (0 linear system, 1 fluid flow, 2 Lorenz, 3 double well) in front. |
| `t_sne_figure.tex` | Standalone pgfplots figure that reads the four files next to it. |
| `tsne_preview.pdf`, `tsne_preview.png` | The scatter drawn with matplotlib, and the same points shaded by state order. |
| `separation.csv` | The scores defined below, per benchmark and for all tensors. |
| `settings.json` | All arguments (among them the driver of the solver), the size of the common basis, the numbers of numerical threads, run time, peak memory and the scores. |
| `tensors_<benchmark>.npz` | The raw data of step 1, with the rows of the sweep, the size of the data set and the driver of the solver. |

The per-benchmark tables have a header row, commas as separators and six decimals:

```
index,x-val,y-val,state_order,action_order,num_paths,num_steps_per_path,seed,grid
```

| Column | Content |
|---|---|
| `index` | Number of the tensor within the benchmark, from 0. |
| `x-val`, `y-val` | The two coordinates of the embedding. |
| `state_order`, `action_order` | Orders of the monomial dictionaries of the tensor. |
| `num_paths`, `num_steps_per_path` | Identification budget of the tensor. |
| `seed` | Data seed. |
| `grid` | Number of the grid of the sweep that the tensor belongs to: 0 and 1 for the two grids of budgets and 2 for the grid of orders of `inferred_layout`; always 0 for `orders`. |

The first three columns are the columns of the tables of the paper (`index,x-val,y-val`).

### Scores

`separation.csv` measures how well the benchmarks separate, with the benchmarks as the
classes. The columns ending in `_embedding` are computed from the two coordinates, those
ending in `_tensors` from the flattened tensors themselves.

- **Silhouette.** For one tensor, with `a` its mean distance to the other tensors of its
  benchmark and `b` its mean distance to the tensors of the nearest other benchmark, the
  silhouette is `(b - a) / max(a, b)`. The score is the mean over the tensors: near 1 for
  compact clusters far apart, near 0 for overlapping ones, negative when tensors are on
  average closer to another benchmark than to their own.
- **Purity.** The share of the 10 nearest neighbours of a tensor that are of its benchmark,
  averaged over the tensors. Without separation it is about 0.25.
- **Purity without replicates** (`purity_other_configuration`). The same after the tensors
  of the same benchmark with the same orders and budget have been removed from the
  candidates. It differs from the purity when a configuration has several data seeds, whose
  tensors are nearly identical.

---

## Reading the tables in pgfplots

pgfplots addresses a column by its name in the header row; the other columns are ignored.
`t_sne_figure.tex`, written by the script, is the complete figure:

```latex
\documentclass[crop,tikz]{standalone}

\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

% Define colors for the individual clusters
\definecolor{linear_sys_color}{RGB}{251, 86, 7}
\definecolor{fluid_flow_color}{RGB}{255, 0, 110}
\definecolor{lorenz_color}{RGB}{131, 56, 236}
\definecolor{double_well_color}{RGB}{58, 134, 255}

\begin{document}

\begin{tikzpicture}
    \begin{axis}[
        enlarge y limits=true,
        enlarge x limits=true,
        yticklabel={\empty},
        xticklabel={\empty},
        legend columns=4,
        axis lines=left,
        width=20cm,
        height=10cm,
        legend image post style={scale=1.5},
        legend style={
            at={(0.5,1.15)},
            anchor=north,
            column sep=0.3cm,
            font=\large,
            draw=white
        }
    ]

    \addplot[only marks, thick, linear_sys_color, mark=diamond*] table [x=x-val, y=y-val, col sep=comma] {linear_system_tsne.csv};
    \addlegendentry{Linear System}

    \addplot[only marks, thick, fluid_flow_color, mark=*] table [x=x-val, y=y-val, col sep=comma] {fluid_flow_tsne.csv};
    \addlegendentry{Fluid Flow}

    \addplot[only marks, thick, lorenz_color, mark=square*] table [x=x-val, y=y-val, col sep=comma] {lorenz_tsne.csv};
    \addlegendentry{Lorenz 1963}

    \addplot[only marks, thick, double_well_color, mark=triangle*] table [x=x-val, y=y-val, col sep=comma] {double_well_tsne.csv};
    \addlegendentry{Stochastic Double Well}

    \end{axis}
\end{tikzpicture}

\end{document}
```

It is compiled inside the output directory:

```bash
cd tsne_koopman_tensor_results
pdflatex t_sne_figure.tex
```

- `col sep=comma` is what a comma-separated table needs in addition to the column names.
- The file is the figure source of the paper (`figures/sources/t_sne_figure.tex` of the
  paper project) with three changes. The tables are read from the directory of the file
  instead of `data/tsne/`. The axis limits `xmin=-60`, `xmax=55` and `ymin=-50`, which the
  paper sets for its own coordinates, are left out: the coordinates of a t-SNE have no fixed
  scale and differ from run to run. The comment lines around the four plots are left out.
- To use the tables in the paper project, copy the four `<benchmark>_tsne.csv` files into
  its `data/tsne/`. The unchanged source of the paper compiles against them; its three axis
  limits then have to be removed or adjusted to the new coordinates.

The combined table gives the same figure from one file, with the mark and the colour chosen
by the column `class`. Saved as `tsne_classes.tex` next to `tsne.csv`, the following is
compiled with `pdflatex tsne_classes.tex`:

```latex
\documentclass[crop,tikz]{standalone}

\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

\definecolor{linear_sys_color}{RGB}{251, 86, 7}
\definecolor{fluid_flow_color}{RGB}{255, 0, 110}
\definecolor{lorenz_color}{RGB}{131, 56, 236}
\definecolor{double_well_color}{RGB}{58, 134, 255}

\begin{document}

\begin{tikzpicture}
    \begin{axis}[
        enlarge y limits=true,
        enlarge x limits=true,
        yticklabel={\empty},
        xticklabel={\empty},
        legend columns=4,
        axis lines=left,
        width=20cm,
        height=10cm,
        legend image post style={scale=1.5},
        legend style={at={(0.5,1.15)}, anchor=north, column sep=0.3cm, font=\large, draw=white},
        scatter/classes={
            0={mark=diamond*, linear_sys_color},
            1={mark=*, fluid_flow_color},
            2={mark=square*, lorenz_color},
            3={mark=triangle*, double_well_color}
        }
    ]

    \addplot[scatter, only marks, thick, scatter src=explicit symbolic]
        table [x=x-val, y=y-val, meta=class, col sep=comma] {tsne.csv};
    \legend{Linear System, Fluid Flow, Lorenz 1963, Stochastic Double Well}

    \end{axis}
\end{tikzpicture}

\end{document}
```

---

## How the published figure was made

The code that produced the published coordinates was not found. The history of this
repository and of the upstream repository (`Pdbz199/koopman-rl`, all branches) hold one
t-SNE script, added on 27 May 2024, which embeds the rows of the regression matrix of a
single benchmark and opens a window. The revised manuscript says, in its electronic
supplementary material, that the tensors "were identified for each benchmark with state and
action dictionaries of several orders, projected onto a common dictionary basis and embedded
in two dimensions by t-distributed stochastic neighbour embedding". It gives no numbers and
no settings; neither does the arXiv version (2403.02290v2, section 5).

The published tables (`data/tsne/<benchmark>_tsne.csv` of the paper project; 161 rows for
the linear system, the Lorenz system and the double well, 157 for the fluid flow) have the
same structure in their row order:

| Rows | What the coordinates show |
|---|---|
| 0 to 120 | 11 blocks of 11. For the Lorenz system and the fluid flow the position along the cluster follows the place in the block: eight places in rising order, then the lowest, then the two highest. This is the order of eleven values such as 50, 100, ..., 550 when their names are sorted as text (100, ..., 450, 50, 500, 550). The linear system and the double well show no such order. |
| 121 to 145 | 5 blocks of 5. For the Lorenz system and the fluid flow the position rises within each block. |
| 146 to 149 | Four points per benchmark at one place that all four benchmarks share. |
| 150 to 153 | Four points inside the cluster of the benchmark. |
| 154 to 160 | Seven points away from the cluster, in tight groups (4 and 3 for the fluid flow and the double well). |

For the fluid flow one of the blocks of 11 holds 7 rows (six rising, then the lowest), and
all later rows come four places earlier: the four largest values of that block are absent,
which gives 157.

A count of 161 that is the same for the two-state double well and the three-state benchmarks
rules out rows or columns of a regression matrix. The sweep `inferred_layout` is the reading
that fits this structure:

- rows 0 to 120: tensors of state order 2 and action order 2 (the defaults of
  `generate_tensor`) on a grid of 11 by 11 identification budgets, the steps per trajectory
  being the variable with the text-sorted order;
- rows 121 to 145: a second such grid of 5 by 5;
- rows 146 to 160: the dictionary orders 1 to 4 by 1 to 4 without one combination, in the
  order state order 1 (four action orders), 2, 3 and 4 (three action orders).

Run with this layout, the module gives what the published figure shows: four clusters; the
Lorenz system and the fluid flow drawn out along the steps per trajectory, the linear system
and the double well without such an order; the tensors of state order 1 of all four
benchmarks at one common place; those of state order 2 inside the clusters; those of state
orders 3 and 4 in tight groups away from their cluster.

| | Published coordinates | Default sweep, on one thread and on two |
|---|---|---|
| Points per benchmark | 161 (fluid flow 157) | 161 |
| Silhouette per benchmark | 0.62 to 0.69 | 0.63 to 0.82 |
| Purity per benchmark | 0.95 to 0.96 | 0.95 to 0.98 |
| Points with fewer than half of their 10 nearest neighbours in their benchmark | 31 | 27 (25 when the tensors of one thread are embedded on two) |

This is an inference from the coordinates, not the original procedure. The following is
assumed and can be changed in `INFERRED_LAYOUT`:

- The second variable of the two budget grids is not determined by the coordinates. The
  number of trajectories is assumed.
- The values 50 to 550 for the steps per trajectory fit the text-sorted order; other value
  sets with the same sorting would fit as well. The numbers of trajectories are 10 to 110 to
  keep the run short. With 50 to 550 for both variables (an exploratory run outside this
  module, with five times more data) the embedding had the same structure, with silhouettes
  of 0.60 to 0.82.
- The tensors of the second grid are not copies of tensors of the first one in the published
  tables. Here the second grid uses the last instead of the first trajectories of the data.
- Which combination of orders is absent is not known; (4, 4) is assumed, as the group sizes
  4, 4, 4, 3 suggest.
- One data seed (123, the default of `generate_tensor`) is assumed. The fluid flow is run
  with all 161 tensors.

---

## What the embedding shows

The scores are those defined in step 3. Unless stated otherwise they are the silhouette and
the purity of all tensors in the embedding, for the tensors of the default sweep and of the
`orders` sweep identified with the default driver on one thread, embedded on one thread and
on two; a range is given where the two embeddings differ.

### Default sweep

The tensors of the four benchmarks form four clusters (silhouette 0.73, purity 0.96). The
clusters are the 146 tensors per benchmark with orders (2, 2), which differ in the amount
of data only. The 15 tensors per benchmark of the grid of orders do not join them:

- Embedded without the grid of orders, nearly every tensor has all its 10 nearest neighbours
  in its benchmark (purity 0.999).
- Embedded alone, the 60 tensors of the grid of orders do not separate by benchmark
  (silhouette 0.16 and purity 0.59 at perplexity 10, 0.11 and 0.58 at perplexity 5).
- The tensors of state order 1 of the fluid flow, the Lorenz system and the double well are
  closer to each other (distances 0.3 to 1.0) than to the tensors of orders (2, 2) of their
  own benchmark (1.6 to 19).

The separation of the clusters does not depend on the open choices:

| Setting | Silhouette | Purity |
|---|---|---|
| default | 0.73 | 0.96 |
| `--perplexity 5` | 0.41 | 0.96 |
| `--perplexity 10` | 0.60 | 0.97 |
| `--perplexity 50` | 0.76 to 0.77 | 0.96 |
| `--perplexity 100` | 0.59 | 0.96 |
| `--tsne_method exact` | 0.82 | 0.96 |
| `--tsne_init random` | 0.72 to 0.73 | 0.96 |
| `--scaling standard` | 0.71 | 0.98 |
| `--scaling unit_norm` | 0.75 to 0.76 | 0.98 |
| `--metric cosine` | 0.77 | 0.97 |
| `--units box` | 0.71 to 0.74 | 0.98 |
| `--subtract_persistence` | 0.72 to 0.73 | 0.96 |
| `--shared_coordinates_only` | 0.66 to 0.68 | 0.95 to 0.96 |
| `--double_well_coordinates 1 2` | 0.73 | 0.97 |
| `--common_basis smallest` | 0.55 | 0.99 |

The scores of single benchmarks vary more than these: at perplexity 5 the silhouette of the
fluid flow is 0.32 on one thread and 0.23 on two. `--tsne_seed` has no effect with the PCA
initialisation (seeds 0, 1, 2 and 42 gave the same scores).

### Sweep over orders

With state and action orders 1 to 4 and 8 data seeds the tensors form one cluster per
benchmark and state order (see the right panel of `tsne_preview.png`), not four clusters:
the silhouette is 0.10 (0.12 for the tensors identified on two threads). The purity is 0.99,
because the 8 seeds of a configuration lie together; without replicates it is 0.94, so the
nearest tensors of other orders are mostly of the same benchmark.

| Setting | Silhouette | Purity | Purity without replicates |
|---|---|---|---|
| default | 0.10 | 0.99 | 0.94 |
| `--perplexity 5` | -0.04 | 0.76 | 0.65 to 0.66 |
| `--perplexity 10` | 0.03 | 0.84 | 0.76 |
| `--perplexity 50` | 0.15 to 0.17 | 1.00 | 0.97 |
| `--perplexity 100` | 0.16 to 0.19 | 0.98 to 0.99 | 0.96 |
| `--tsne_method exact` | 0.07 | 0.98 | 0.94 |
| `--scaling standard` | 0.27 to 0.28 | 0.99 | 0.95 to 0.96 |
| `--scaling unit_norm` | 0.09 to 0.10 | 1.00 | 0.97 |
| `--metric cosine` | 0.03 | 0.93 | 0.89 |
| `--units box` | 0.28 to 0.31 | 1.00 | 0.98 |
| `--subtract_persistence` | 0.11 to 0.14 | 0.98 to 0.99 | 0.92 to 0.94 |
| `--shared_coordinates_only` | 0.07 to 0.08 | 0.98 | 0.94 to 0.95 |
| `--common_basis smallest` | 0.56 | 1.00 | 1.00 |
| `--linear_system_seed -1` (the linear system identified again) | 0.07 to 0.08 | 0.98 | 0.95 |
| 2 of the 8 seeds | 0.13 | 0.76 | 0.71 |
| 4 of the 8 seeds | 0.08 | 0.96 | 0.92 |
| state order 2 only, action orders 1 to 4 | 0.65 | 0.98 | 0.94 |
| action order 2 only, state orders 1 to 4 | 0.30 | 0.85 | 0.57 |

The state order is what splits a benchmark: with one state order the four benchmarks form
four clusters, with one action order and four state orders they do not (of the nearest
tensors of other configurations, 57% are of the same benchmark). With
`--common_basis smallest` only the block of orders (1, 1) is compared; the scores are
higher, but the picture consists of a few dense spots.

### Among the flattened tensors

The scores among the flattened tensors themselves (columns `_tensors`) are the same for the
tensors identified on one thread and on two, to the four decimals of `separation.csv`:

| | Linear system | Fluid flow | Lorenz | Double well | All |
|---|---|---|---|---|---|
| Default sweep, silhouette | -0.92 | 0.81 | -0.94 | 0.88 | -0.04 |
| Default sweep, purity | 0.96 | 0.98 | 0.95 | 0.98 | 0.97 |
| Default sweep without the grid of orders, silhouette | 1.00 | 0.99 | 0.28 | 0.98 | 0.81 |
| `orders` sweep, silhouette | -0.80 | 0.07 | -0.82 | 0.48 | -0.27 |
| `orders` sweep, purity | 0.96 | 1.00 | 0.96 | 1.00 | 0.98 |

The silhouettes of the linear system and of the Lorenz system are negative although their
purity is high. The silhouette averages the distances from a tensor to all tensors of its
benchmark, and a few tensors of the grid of orders lie far from everything in the units of
the environments: in the default sweep three tensors of state order 4 are at distances of
3,200 to 6,500 from the (2, 2) tensors of the linear system, and two tensors of state
order 4 at 68,000 and 77,000 from those of the Lorenz system (the third at 2,100). The mean
distance of a (2, 2) tensor to its own benchmark is therefore 85 for the linear system and
968 for the Lorenz system, against 4.3 and 29 to the nearest other benchmark. The purity only looks at the 10
nearest neighbours, which are (2, 2) tensors of the same benchmark. Without the grid of
orders the silhouettes are positive. The t-SNE works with neighbourhoods, which is why the
embedding shows clusters where the silhouette among the tensors is negative.

### What the separation rests on

Three points limit what can be read from the figure.

- **State dimension.** The double well differs from the other benchmarks by construction,
  because its tensors have no coefficients for monomials of the third coordinate. Tensors of
  "no change" (`K[i, i, 0] = 1`, no dynamics) in the layout of the default sweep separate
  the double well from the rest with a purity of 0.98; restricted to the shared coordinates
  they do not (0.004 embedded on two threads, 0.10 on one). In the units of the environments, 99% of the squared distance between
  the mean (2, 2) tensors of the fluid flow and of the double well lies outside the shared
  coordinates. The identified tensors still separate when only the shared coordinates are
  kept (`--shared_coordinates_only`: purity of the double well 0.95), so the state dimension
  is not the only difference.
- **Scale.** In the units of the environments the mean (2, 2) tensor of the Lorenz system
  has norm 25 and those of the other benchmarks 2 to 4, and about 80% of the squared distance
  from the Lorenz system to each other benchmark is in three coefficients. `--units box`
  divides every state coordinate and the action by the largest absolute value of its bound
  in the observation and action boxes of the environment (20, 50 and 50 for the three
  coordinates of the Lorenz system). In these units all four norms are 2 to 3, the distances
  between the mean tensors 0.9 to 2.9, and the clusters remain.
- **Spread within a benchmark.** For the (2, 2) tensors of the two budget grids, in the
  units of the environments:

  | | Linear system | Fluid flow | Lorenz | Double well |
  |---|---|---|---|---|
  | Norm of the mean tensor | 4.15 | 3.16 | 24.9 | 2.35 |
  | Root-mean-square distance from the mean | 0.0000 | 0.0201 | 18.9 | 0.0347 |
  | Largest distance from the mean | 0.0000 | 0.046 | 56.9 | 0.187 |

  The distances between the mean tensors of the benchmarks are 2.0 to 25. For the linear
  system (its dynamics are in the dictionary), the fluid flow and the double well the tensor
  at fixed orders is nearly the same for every amount of data, so any property that differs
  between the benchmarks separates them. For the Lorenz system this does not hold: its
  tensor depends strongly on the amount of data. Its norm falls from 62 at 50 steps per
  trajectory to 21 at 550 steps (means over the numbers of trajectories), and its spread is
  of the size of its distance to the other benchmarks. Its tensors form one drawn-out
  cluster because the other benchmarks lie elsewhere, not because they are close to each
  other. In box units the spread of the Lorenz tensors is 0.20, against a norm of 3.1.

The figure therefore shows that the least-squares construction returns a tensor that is
specific to the system, and that for three of the four benchmarks this tensor hardly depends
on the amount of data at fixed dictionary orders. It does not show this for the Lorenz
system, it does not show that tensors of different dictionary orders of one system are
close, and it does not measure predictive accuracy (for that, see
`koopman_prediction_validation.py`).

---

## Reproducibility

**Data.** The random-agent data are reproduced from the seeds: the transitions of a
benchmark are the same, value for value, in every process, whether the benchmark is
identified alone or after others.

**Tensors.** What the regression returns from these data depends on the LAPACK driver of
`torch.linalg.lstsq`, which `--lstsq_driver` sets and which is stored with the tensors and
in `settings.json`.

| Driver | Method | Result |
|---|---|---|
| `gelsd` (default) | Singular value decomposition (divide and conquer). | The same tensor in every call. |
| `gelss` | Singular value decomposition. | The same tensor in every call; equal to that of `gelsd` to a relative 4e-10 in the default sweep. |
| `gelsy` | Orthogonal factorisation with column pivoting. The driver of the solver call of the package, which names none. | Not the same tensor in every call. |

With `gelsd`, on one machine and with one number of numerical threads, a run reproduces the
tensors bit for bit and the tables byte for byte. This was compared for:

| Runs, each in processes of its own | Tensors bit-identical | Tables |
|---|---|---|
| Default sweep, two runs on one thread | 644 of 644 | byte-identical |
| Default sweep on one thread with one process per benchmark (four calls with `--environments <benchmark>`, then `--embed_only`), against the runs above | 644 of 644 | byte-identical |
| Default sweep, two runs on two threads | 644 of 644 | byte-identical |
| `orders` sweep, two runs on one thread | 512 of 512 | byte-identical |
| Default sweep with `gelss`, two runs on one thread | 644 of 644 | byte-identical |

"Tables" are the four `<benchmark>_tsne.csv`, `tsne.csv` and `separation.csv`;
`t_sne_figure.tex`, `tsne_preview.png` and the four `tensors_<benchmark>.npz` were equal as
files as well. `settings.json` differs in the output folder, the run time and the peak
memory, and `tsne_preview.pdf` differs as a file.

The number of threads is one of the conditions. Between a run on one thread and a run on
two, 38 of the 644 tensors of the default sweep were bit-identical; the others differed by a
relative 2e-13 in the median and by up to 4e-6, where the difference of two tensors is the
largest difference of a coefficient relative to the largest coefficient. The largest
differences are those of the worst-conditioned regressions, all of the Lorenz system: 4e-6
at orders (2, 4), 2e-6 at (4, 3), 1e-6 at (3, 4), 9e-7 at (4, 2), 4e-7 at (3, 3), and below
3e-8 for every other tensor. In the `orders` sweep no tensor was bit-identical between one
thread and two (median 3e-12, largest 8e-6, again the Lorenz system at (2, 4)). Another
machine, or another build of torch and of its LAPACK, was not tried; differences of this
kind have to be expected there.

Some regressions are rank-deficient to the solver. `torch.linalg.lstsq` estimates an
effective rank with a threshold of the machine precision times the larger dimension of the
regression matrix (6.7e-12 for 30,000 transitions, relative to the largest singular value)
and solves for that rank.

| Sweep | Regressions with singular values below the threshold | Within a quarter of a decade above it |
|---|---|---|
| default | Lorenz system at orders (3, 4) and (4, 3) (condition numbers 7e12 and 6e12, five singular values below the threshold each): 2 of 644 | Lorenz system at (2, 4), (3, 3), (4, 2) |
| `orders` | Linear system at (4, 4) and Lorenz system at (3, 4), (4, 3), (4, 4) on all 8 seeds, Lorenz system at (4, 2) on 3 seeds: 35 of 512 | Lorenz system at (2, 4) and (3, 3), and at (4, 2) on the other seeds |

All other regressions are more than one decade away from the threshold, except the linear
system at (4, 3) in the `orders` sweep (half a decade). For the regressions below the
threshold `gelsd` returns the solution of smallest norm at the effective rank: in the
default sweep rank 95 of 100 with norm 44.7 for the Lorenz system at (3, 4), and rank 135 of
140 with norm 2,058 at (4, 3). These are the same solutions on one thread and on two (to
the 1e-6 and 2e-6 above), and no tensor of the `orders` sweep differed by more than 9e-6
between one thread and two. Where singular values lie at the threshold, last-digit
differences of another machine or build can change the estimated rank, and with it the
solution.

**The driver of the package.** `--lstsq_driver gelsy` is the solver call of the package
(`koopmanrl.koopman_tensor` and the modules of the algorithms). Its result differs from
call to call, also within one process, on one thread and with identical data. torch hands
LAPACK's `GELSY` a pivot array that it has not initialised (torch 2.9.1, and the sources of
its releases from 1.9.0 to 2.13.0; the development branch of torch sets it to zero). LAPACK
keeps every column whose entry in this array is not zero out of the column pivoting, so the
factorisation follows what the memory held. The same LAPACK routine, called directly with
the array filled with non-zero values, gave a tensor of torch bit for bit for a
well-conditioned regression and the norms 81.7 and 76,319 below; with other contents of the
array it gave other results.

- For the regressions far from the threshold this changes the last digits. Between four
  runs of the default sweep on one thread the tensors differed by up to a relative 1.2e-12,
  and between 189 and 414 of the 644 tensors of two runs were bit-identical. (Three runs of
  the `orders` sweep on two threads agreed to 1e-10.)
- For the two regressions of the default sweep below the threshold it changes the estimated
  rank and the solution. In 12 identifications on one thread the tensor of the Lorenz system
  at (3, 4) had norm 506 six times, 81.7 four times, 1.47 and 0.41 once each, and the tensor
  at (4, 3) had norm 76,319 eleven times and 18,709 once. This happens in a single call of
  the script as it does with one process per benchmark. Called directly with a zeroed pivot
  array, the routine returns ranks 95 and 135 and norms 44.8 and 2,069, close to the
  solutions of `gelsd`.
- Against `gelsd`, the tensors of `gelsy` agree to a relative 6e-6 (median 2e-13) for the
  642 other regressions of the default sweep.

**Embedding.** The t-SNE is deterministic for identical tensors and an identical number of
numerical threads, and sensitive to anything else. The displacements are in percent of the
diagonal of the embedding, for the default sweep:

| What differs | Largest displacement of a point | Median |
|---|---|---|
| `gelsd`, two runs with the same number of threads | none | none |
| `gelsd`, one thread against two, same stored tensors | 11% | 1.3% |
| `gelsd`, tensors identified on one thread against two (up to 4e-6), embedded with the same number of threads | 12 to 14% | 3.6 to 4.2% |
| `gelsd`, a run on one thread against a run on two | 14% | 4.2% |
| `gelsd` against `gelss` (up to 4e-10), one thread | 14% | 3.5% |
| `gelsy`, 11 of the 644 tensors differ, by up to 3e-15 | 21% | 2.7% |
| `gelsy`, two runs on one thread with the same solutions for the two regressions below the threshold (up to 6e-13) | 22% | 3.0% |
| `gelsy`, two runs on one thread with another solution for one of them | 12% | 1.5% |
| `gelsy`, two runs on one thread with other solutions for both of them | 29 to 36% | 8% |
| `gelsd` against `gelsy`, one thread | 18 to 22% | 3 to 4% |

In the `orders` sweep with `gelsd`, a run on one thread against a run on two moved single
points by up to 23% (median 0.7%).

The clusters stay the same in all these cases. Over the embeddings of the default sweep
compared here (five with `gelsd` or `gelss`, seven with `gelsy`), the silhouette of all
tensors was 0.72 to 0.73 and their purity 0.96 to 0.97; per benchmark the silhouette was
0.62 to 0.82 and the purity 0.95 to 0.98.

**Exact reproduction.** With the default driver, a figure is reproduced from the seeds on
the same machine, with the same libraries and the same number of threads. From stored
tensors the embedding alone is repeated: `--embed_only` gave the same `tsne.csv` byte for
byte in separate processes, on one thread and on two, and different tables between the
two. The number of threads is fixed with the environment variable `OMP_NUM_THREADS`, for
the identification (checked for the Lorenz system) and for the embedding:

```bash
OMP_NUM_THREADS=1 uv run -m koopmanrl_utils.tsne_koopman_tensor
OMP_NUM_THREADS=1 uv run -m koopmanrl_utils.tsne_koopman_tensor --embed_only
```

`settings.json` records the numbers of threads and the driver of a run. Keep the four
`tensors_<benchmark>.npz` files (1.7 MB together for the default sweep) and the number of
threads with a figure, so that its coordinates can be regenerated without the
identification.

**What is not reproduced.** The coordinates of the published figure were not made by this
code (see "How the published figure was made"): the script reproduces their structure, not
their values. Tensors of `gelsy` are not reproduced from the seeds, only read from their
files. Between numbers of threads the tensors of `gelsd` agree to rounding, which is up to
9e-6 for the worst-conditioned regressions, and the coordinates of single points do not
agree. The same has to be expected between machines and between builds of the libraries,
for a run from the seeds and for an embedding of stored tensors; neither was compared.

---

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--sweep` | `inferred_layout` | `inferred_layout` or `orders`. |
| `--environments` | all four | Benchmarks, by the names of the tables. |
| `--state_orders`, `--action_orders` | `1 2 3 4` | Dictionary orders of the `orders` sweep. |
| `--num_paths`, `--num_steps_per_path` | tuned SKVI budget | Identification budgets of the `orders` sweep; several values give a grid. |
| `--seeds`, `--seed0` | `8`, `0` | Number of data seeds and first seed of the `orders` sweep. |
| `--linear_system_seed` | `0` | Seed of the matrix A that the linear system draws at construction. Negative: a new matrix per data seed, with the data of `generate_koopman_tensor`. |
| `--lstsq_driver` | `gelsd` | LAPACK driver of the least-squares solver of the identification. `gelsd` and `gelss` (singular value decompositions) return the same tensor in every call. `gelsy` is the driver of the solver call of the package; its tensors differ from call to call (see "Reproducibility"). |
| `--common_basis` | `largest` | `largest`: dictionaries of the largest orders, missing coefficients zero. `smallest`: only the block of the smallest orders. |
| `--double_well_coordinates` | `0 1` | Coordinates of the common state that the double well occupies. |
| `--units` | `raw` | `raw`: units of the environments. `box`: every state coordinate and the action divided by the largest absolute value of its bound in the boxes of the environment. |
| `--subtract_persistence` | off | Remove the coefficients `K[i, i, 0] = 1` of "no change". |
| `--shared_coordinates_only` | off | Drop the monomials of the coordinate that the double well does not have. |
| `--scaling` | `none` | `standard`: z-score of every coefficient. `unit_norm`: every tensor divided by its norm. |
| `--metric` | `euclidean` | `euclidean` or `cosine`. |
| `--perplexity`, `--tsne_seed`, `--tsne_init`, `--tsne_method` | `30`, `42`, `pca`, `barnes_hut` | Settings of the t-SNE. |
| `--neighbours` | `10` | Number of nearest neighbours of the purity scores. |
| `--output_dir` | `tsne_koopman_tensor_results` | Folder the files are written into. |
| `--resume` | off | Read the stored tensors of the benchmarks that an interrupted run of the same sweep and driver has finished. |
| `--embed_only` | off | Repeat steps 2 and 3 from the stored tensors, which have to be those of the sweep and the driver of the arguments. |
| `--identify_only` | off | End after step 1: store the tensors of `--environments` and write nothing else. Used by the Snakemake workflow, with one call per benchmark. |
