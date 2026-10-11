# Snakemake Workflows

This directory holds the [Snakemake](https://snakemake.readthedocs.io) workflows that
reproduce results of the paper. A workflow knows which results exist, makes the missing
ones, and makes a result again when a run it depends on was interrupted.

| Workflow | Target | Result | Guide |
|---|---|---|---|
| Episodic returns | `episodic_returns` | Tables of the episodic-return figures | `koopmanrl_utils/EPISODIC_RETURNS.md` |
| Ablations | `ablations` | Tables of the two ablation figures | `koopmanrl_utils/ABLATIONS.md` |
| t-SNE | `tsne` | Tables of the t-SNE figure of the Koopman tensors | `koopmanrl_utils/TSNE.md` |

---

## Requirements

Snakemake is not a dependency of the project: the project is pinned to Python 3.10, and
Snakemake 8 and newer need Python 3.11 or newer. It is run as a separate tool with `uvx`,
and its jobs call the code of the project through the project environment:

```bash
uv sync                                  # once: creates the project environment in .venv/
uvx --python 3.12 snakemake --version    # 9.27.0 at the time of writing
```

`uvx` takes the newest Snakemake when it installs it for the first time. To use one version
on every machine, name it in every call, as `snakemake@9.27.0` in place of `snakemake`:

```bash
uvx --python 3.12 snakemake@9.27.0 --version
```

A job starts the project code as `uv run --project <repository> --no-sync python -m <module>`.
`--no-sync` uses the environment as it is, so `uv sync` has to be run before the first job.

All commands below are run from the root of the repository. Snakemake finds
`workflow/Snakefile` and the settings in `workflow/profiles/default/` by itself, and keeps
its bookkeeping in `.snakemake/` in the directory it is started from.

---

## All workflows

### A call without a target

`snakemake` without a target makes nothing: the call stops and lists the targets
(*executed*, also as a dry run). Every call names one: `episodic_returns`, `ablations` or
`tsne` for a workflow, a path of a result, or `all` for the results of all three workflows.
A dry run of `all` (*executed*) lists 1,997 jobs:

```
episodic_returns_run     495     episodic_returns_frame   20     episodic_returns_table   20
ablations_run          1,440     ablations_frame           8     ablations_table           8
tsne_identify              4     tsne_embed                1
all                        1
```

1,935 of them are reinforcement-learning runs of 50,000 environment steps, 1,440 of these
for the ablations, which take several days of computing time. Use `-n` first: it lists the
jobs of a target without running them.

Options that take a list of words take the targets that follow them as well: after
`--forcerun`, `--resources` and `--default-resources`, a target is read as one more value
of the option, and the call is left without a target (*executed* with `--forcerun`: the
call stops with the list of targets). Name the targets first, or put another option in
between.

### Missing intermediate results

Snakemake makes a missing intermediate result only when something that depends on it has to
be made. A run directory, a data frame or a file of tensors that was deleted is therefore
not made again as long as the files made from it exist: the call ends with "Nothing to be
done". Name the intermediate result as a target as well, or with `--forcerun`; the jobs
after it are then made again too.

```bash
uvx --python 3.12 snakemake --cores 1 ablations results/ablations/frames/sakc_linear_system.json

uvx --python 3.12 snakemake --cores 1 ablations \
    --forcerun results/ablations/runs/sakc/linear_system/0.0001__0.0005/21

uvx --python 3.12 snakemake --cores 1 tsne results/tsne/tensors/tensors_double_well.npz
```

*Executed* for the ablations and the t-SNE, not for the episodic returns, whose rules are
of the same kind as those of the ablations:

- Ablations, a deleted data frame: nothing was done. The first command made the data frame
  and its tables again and nothing else (where three data frames and nine tables existed:
  one data frame and its three tables). The second command makes one run, its data frame and
  its tables again (as a dry run).
- t-SNE, a file of tensors moved away: nothing was done. The third command identified the
  tensors of that benchmark and made the embedding again; with `--forcerun` instead, a dry
  run listed the same two jobs. The tensors that are identified again are the ones from
  before, see [What repeats and what does not](#what-repeats-and-what-does-not).

### The directory Snakemake is started from

Snakemake keeps a record of every result: the job that made it, with its parameters and
its inputs, and whether that job finished. The records lie in `.snakemake/` of the
directory Snakemake is started from, under the path of the result as the call spells it.
Start every call for one set of results from the same directory, with the same
`results_dir`.

From another directory, with `results_dir` spelled differently (relative in one call,
absolute in the next), or for a results folder that was copied without `.snakemake/`,
Snakemake finds results without a record and goes by the files and their time stamps
alone (*executed* for the episodic returns, with made-up logs in place of the runs):

- a changed setting is not noticed: after a change of `total_timesteps` no run is made
  again, and the call ends with "Nothing to be done";
- the directory of a run that was interrupted counts as a finished run;
- a data frame or a table is made again when the script that builds it carries a later
  time stamp, whether the script changed or not.

The data frame job is then the only guard for the runs. It fails when the runs of a data
frame logged different steps, and names the runs that are shorter than the others; it does
not notice a data frame whose runs are all equally short.

### What makes a job again

A job is made again when its result is missing and needed, when one of its inputs is
newer than its result, when the set of its inputs changed, and when its parameters
changed. `rerun-triggers` in `workflow/profiles/default/config.yaml` lists these reasons,
and `-n` prints the reason of every job. For the episodic returns and the ablations
(*executed* for both on a copy of `workflow/` and `configurations/`):

- **Settings** are parameters and inputs of the jobs. Another `total_timesteps` makes the
  runs again and everything after them; one more seed makes the runs of that seed, and the
  data frames and tables they enter.
- **The scripts that build data frames and tables** are inputs of the jobs that call them:
  `koopmanrl_utils/dataframe_creator.py`, `process_episodic_returns.py`,
  `process_skvi_ablations.py` and `process_sakc_ablations.py`. A change of one of them
  makes its data frames or tables again, and the tables after a data frame. No run is
  made again. Snakemake compares the content of a script with its record, so a new time
  stamp alone (`touch`) makes nothing again. The scripts are inputs under their full
  paths: with the repository at another path, the data frames and tables are made again
  as well, and the runs are not.
- **Nothing else of the code** makes a job again: not a change of an algorithm in
  `koopmanrl/`, of a launcher (`run_optimized_experiments.py`, `run_ablations.py`), of a
  file `configurations/*_hparams.json`, or of a rule in `workflow/rules/`. Snakemake can
  also go by the text of a rule, which is switched off here: with it, one blank more in the
  shell command of a run rule makes every run of a campaign again.

After a change of the third kind, name what is to be made again with `--forcerun`, which
takes names of rules and paths of results; the jobs after them are made again as well. The
first command below lists every run of the episodic returns with its data frame and table,
the second one run with its data frame and table (*executed* on made-up logs); without `-n`
they are made:

```bash
uvx --python 3.12 snakemake -n episodic_returns --forcerun episodic_returns_run

uvx --python 3.12 snakemake -n episodic_returns \
    --forcerun results/episodic_returns/runs/skvi/lorenz/8801
```

What makes the jobs of the t-SNE again is described in its section.

---

## Episodic returns

```
episodic_returns_run     one run per algorithm, benchmark and seed    495 jobs
episodic_returns_frame   one data frame per algorithm and benchmark    20 jobs
episodic_returns_table   one table per algorithm and benchmark         20 jobs
```

These are the three steps of `koopmanrl_utils/EPISODIC_RETURNS.md`, with the same scripts.
A run job calls `koopmanrl_utils.run_optimized_experiments` for a single run, so the command
line of a run is built in one place. The algorithms, the benchmarks and the seeds are read
from `configurations/episodic_returns.json`, which that script reads as well.

Results are written to `results/episodic_returns/`:

| Path | Content |
|---|---|
| `runs/<algorithm>/<benchmark>/<seed>/` | Working directory of one run: its TensorBoard log (`runs/...`), console output (`logs/`) and checkpoints (`saved_models/`; LQR writes none). |
| `frames/<algorithm>_<benchmark>.json` | Data frame of the runs of one algorithm on one benchmark. |
| `tables/<benchmark>/<ALGO>_<benchmark>.csv` or `.dat` | Table for pgfplots, see [Format of the tables](#format-of-the-tables). |
| `logs/run/`, `logs/frame/`, `logs/table/` | Output of the jobs. |

`<algorithm>` is `lqr`, `sac_q`, `sac_v`, `skvi` or `sakc`; `<ALGO>` is the same name in
capitals, as in the file names of the paper; `<benchmark>` is `linear_system`, `fluid_flow`,
`lorenz` or `double_well`.

Every run has a directory of its own. Snakemake removes the directory of a run that fails,
and makes a run again whose job was interrupted, so a data frame is only built from runs
that finished. The data frame job also checks its data frame against the run directories
it was built from, and fails unless

- every run directory holds one run folder, which is a run of the algorithm and the
  benchmark of the data frame;
- the seed in the name of the run folder is the seed of its directory;
- every run logged a return, and all runs at the same steps.

The message names the run directories that do not fit, for the last point those that
deviate from the steps of most runs, and is appended to the log of the job in
`logs/frame/`.

Snakemake removes the data frame of a job that failed. Tables that were made from an
earlier data frame stay where they are, and the same call again ends with "Nothing to be
done", since a missing data frame alone does not make its tables again (see
[Missing intermediate results](#missing-intermediate-results)); the tables then still
describe the earlier data frame (*executed* on made-up logs, with a selection that was
changed and a run folder in the wrong directory). Once the runs are corrected, name the
data frame as a target together with the tables, or remove the tables. The message of the
job says so.

### Format of the tables

A table is written in the format of its file name, by `process_episodic_returns.py`:

- `.csv`, the default, is comma-separated. It is the format for new figures: pgfplots
  reads it with `col sep=comma` and takes the columns by their names, as
  `EPISODIC_RETURNS.md` shows.
- `.dat` is space-separated, as the tables that the figure sources of the paper read.
  `tables/` has the layout of `data/episodic_returns/` of the paper sources, so its `.dat`
  tables can be put there and the figure sources stay as they are (*executed* with tables
  of made-up returns: `linear_system_episodic_returns.tex` of the paper compiled unchanged).

The target `episodic_returns` makes the format of the setting `table_format`; a table of
either format can be named by its path, and both can lie next to each other
(*executed* as dry runs and on made-up returns):

```bash
uvx --python 3.12 snakemake --cores 4 episodic_returns --config 'episodic_returns={table_format: dat}'

uvx --python 3.12 snakemake --cores 4 results/episodic_returns/tables/lorenz/LQR_lorenz.dat
```

### Commands

Commands marked *executed* were run while this workflow was written, with Snakemake 9.27.0:
as a dry run where noted, and otherwise on the linear system with at most two seeds and at
most 2,200 steps per run (the settings under [Short runs](#short-runs)). SKVI and SAKC were
run through the workflow once each, with one seed and 400 steps. The full set of 495 runs
was not made with the workflow.

**Dry run** (*executed* with the default settings: 495, 20 and 20 jobs). Lists the jobs
without running them:

```bash
uvx --python 3.12 snakemake -n episodic_returns
```

**Local run with N cores** (*executed* with `--cores 2` on short runs). A run job takes one
core. `--resources mem_mb=...` limits the sum of the memory the running jobs declare, in MB:
4000 for a run of SKVI, 2000 for the other runs and 1000 for the other jobs. With
`--keep-going`, the jobs that do not depend on a failed job are still made.

```bash
uvx --python 3.12 snakemake --cores 8 --resources mem_mb=16000 --keep-going episodic_returns
```

Two things to know about `--resources` (*executed* as dry runs):

- The limit holds jobs back only if it is at least the largest declaration, the 4000 of a
  run of SKVI. A job that declares more than the limit is not refused: Snakemake lowers its
  declaration to the limit, so with `mem_mb=3000` a run of SKVI counts as 3000 and with
  `mem_mb=1500` every run counts as 1500.
- `--resources` takes every word that follows it. The target must not come directly after
  the value: `--resources mem_mb=16000 episodic_returns` stops with a `ValueError`. Put the
  target first or another option in between, as above.

The declared memory is above what the runs took in short runs on the linear system: 2.9 GB
for SKVI, and 0.7 to 0.9 GB for the other four algorithms. `run_optimized_experiments.py`
gives about 3.5 GB for a full run of SKVI on the fluid flow.

**Part of the experiments** (*executed* in both forms on the linear system; the two commands
below as dry runs only: 50 runs, 2 data frames and 2 tables each). Either name the tables that
are wanted,

```bash
uvx --python 3.12 snakemake --cores 4 \
    results/episodic_returns/tables/lorenz/LQR_lorenz.csv \
    results/episodic_returns/tables/lorenz/SAC_Q_lorenz.csv
```

or restrict the target `episodic_returns` (see [Settings](#settings)):

```bash
uvx --python 3.12 snakemake --cores 4 episodic_returns \
    --config 'episodic_returns={only_benchmarks: [Lorenz-v0], only_algorithms: [lqr, sac_q]}'
```

**Resuming** (*executed* for each case below). Give the same command again, from the same
directory (see [The directory Snakemake is started from](#the-directory-snakemake-is-started-from)).
Jobs whose results exist are not repeated, and a deleted table is made again from its data
frame.

- A run that fails, or whose process is killed: Snakemake removes its directory, and the
  next call makes the run again. Its console output is kept in
  `results/episodic_returns/logs/run/<algorithm>/<benchmark>/<seed>.log`.
- Ctrl-C: Snakemake stops the running jobs. Their directories stay, marked as incomplete,
  and the next call makes these runs again from the start.
- `kill` of Snakemake (SIGTERM): Snakemake waits for the running jobs and then stops.
- Snakemake killed outright (SIGKILL, power loss): the running runs go on without it and
  the directory stays locked. Wait for these runs to end or stop them, unlock, and give
  the command again; the runs that were going are made again.

  ```bash
  uvx --python 3.12 snakemake --unlock
  ```

**Slurm** (*not executed*: no Slurm was available. Only a dry run of the command below was
made, with version 2.8.0 of the plugin; it accepted the options and listed the 536 jobs,
535 of them with the memory they declare). With
[snakemake-executor-plugin-slurm](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html),
every job is submitted as a Slurm job with the memory it declares:

```bash
uvx --python 3.12 --with snakemake-executor-plugin-slurm snakemake episodic_returns \
    --executor slurm --jobs 100 \
    --default-resources slurm_account=<account> slurm_partition=<partition> runtime=<minutes>
```

Name the target before `--default-resources`, which takes every word that follows it, as
`--resources` does.
`runtime` is the time limit of a job in minutes; the duration of a full run on a given
cluster has to be measured first. The repository, its `.venv/`, `uv` and the installation of
Snakemake have to be reachable under the same paths on the compute nodes.

### Short runs

For a test of the workflow, the seeds, the number of steps and the results directory can
be set on the command line (*executed*, with another results directory):

```bash
uvx --python 3.12 snakemake --cores 2 episodic_returns --config results_dir=/tmp/smoke \
    'episodic_returns={only_benchmarks: [LinearSystem-v0], only_algorithms: [lqr, sac_q, sac_v],
                       seeds: {LinearSystem-v0: [4430, 2738]}, total_timesteps: 2200}'
```

`total_timesteps` has to cover at least one episode, which is 200 steps on the linear
system and 2,000 steps on the other three benchmarks. A return is logged at the end of an
episode; a shorter run logs none, and the data frame job fails with a `KeyError` of
`dataframe_creator.py` in its log (*executed* with 400 steps on the Lorenz system and
made-up logs).

### Settings

Settings are given with `--config` as above, or in a YAML or JSON file with `--configfile`.
Nested values are merged into the defaults; a list replaces the default list. The settings
are checked before any job is made, for all three workflows (*executed* as dry runs):

- A key that `configurations/episodic_returns.json` does not have is refused, with the
  keys that exist: a misspelled key of `episodic_returns`, a benchmark under `seeds` that is
  not written as `Lorenz-v0`, an algorithm under `algorithms`. A misspelled name of the
  block itself (`episodic_return={...}`) is not noticed: Snakemake takes it as a setting
  that nothing reads.
- The algorithms and the benchmarks are not settings, since the launcher reads them from
  the file: apart from `mem_mb`, an entry under `algorithms` or `benchmarks` that is not as
  the file lists it is refused (*executed* as dry runs).
- `null` on the command line is written as the word: `total_timesteps: null`
  (`none` is read the same way).
- A list that selects something holds at least one entry and none twice; seeds are whole
  numbers from 0, and `total_timesteps` a whole number from 1.

| Key | Default | Meaning |
|---|---|---|
| `results_dir` | `results` | Directory of the results of all workflows. |
| `python` | `uv run --project <repository> --no-sync python` | Command that starts the Python of the project environment. |
| `episodic_returns: total_timesteps` | `null` | Environment steps per run; `null` leaves the 50,000 of the algorithms. |
| `episodic_returns: only_algorithms` | `null` | List of algorithms the target `episodic_returns` is restricted to; `null` for all. |
| `episodic_returns: only_benchmarks` | `null` | List of benchmarks it is restricted to, as `LinearSystem-v0`, `FluidFlow-v0`, `Lorenz-v0`, `DoubleWell-v0`; `null` for all. |
| `episodic_returns: table_format` | `csv` | Format of the tables that the target `episodic_returns` makes: `csv` or `dat`, see [Format of the tables](#format-of-the-tables). |
| `episodic_returns: seeds: <benchmark>` | seeds of the paper | Seeds of the runs on a benchmark, under its name as `Lorenz-v0`. |
| `episodic_returns: algorithms: <algorithm>: mem_mb` | 2000, SKVI 4000 | Memory a run declares, in MB. |

### What repeats exactly

- **Number of threads.** Snakemake sets `OMP_NUM_THREADS` and the related variables to the
  number of threads of a job, which is one. In our check, a run of SAC (Q) past the start of
  its training logged the same returns on every call with these variables set to one, and
  the same returns on every call without them, but not the same in both cases. A run made by
  hand with `run_optimized_experiments.py` therefore repeats a run of the workflow only with
  `OMP_NUM_THREADS=1`, or on a machine with one core.
- **Tables.** With the same seeds and settings, the tables of LQR, SAC (Q) and SAC (V) from
  the workflow were identical, byte for byte, to tables made by hand with the commands of
  `EPISODIC_RETURNS.md`. This was checked with two seeds and 2,200 steps per run, which is
  before SAC starts to train (step 5,000), so it does not cover the training of SAC.
- **Order of the runs.** The confidence band of a table depends on the order of the runs in
  its data frame. `dataframe_creator.py` lists the runs in the order of the names of their
  folders, which start with the benchmark, the algorithm and the seed, so the order does
  not depend on the file system or on the times at which the runs started.
- **SAKC.** Two runs of SAKC with the same seed differ, as described at the top of
  `run_optimized_experiments.py`, so its tables agree between reruns in distribution only.
- **Changed settings.** Snakemake makes a job again when its parameters changed, for example
  the runs when `total_timesteps` changes. Changing `python` does not repeat any job. See
  [What makes a job again](#what-makes-a-job-again).

## Ablations

```
ablations_run     one run per algorithm, benchmark, grid point and seed     1,440 jobs
ablations_frame   one data frame per algorithm and benchmark                    8 jobs
ablations_table   one table per algorithm, benchmark and smoothing window       8 jobs
```

These are the three steps of `koopmanrl_utils/ABLATIONS.md`, with the same scripts. A run
job calls `koopmanrl_utils.run_ablations` for a single run, so the command line of a run is
built in one place. The two grids and the seeds are read from `configurations/ablations.json`,
which that script reads as well. The algorithms of that file carry the names they have in
`configurations/episodic_returns.json`: their modules, the folders of their logs, the memory
of their runs and the benchmarks are listed there only, and are looked up under these names.

**The full campaign is 1,440 runs of 50,000 environment steps (two algorithms on four
benchmarks, at the 36 points of a grid, with five seeds per grid point) and takes several
days of computing time.** The target `ablations` starts it, unless `-n` is given or the
settings restrict it, and so does the target `all`. It was not made with the workflow.

Results are written to `results/ablations/`:

| Path | Content |
|---|---|
| `runs/<algorithm>/<benchmark>/<first>__<second>/<seed>/` | Working directory of one run: its TensorBoard log (`runs/SKVI/...` or `runs/SAKC/...`), checkpoints (`saved_models/`) and console output (`logs/`). |
| `frames/<algorithm>_<benchmark>.json` | Data frame of the runs of one algorithm on one benchmark. |
| `tables/<ALGO>/window_<w>/<benchmark>_ablation.csv` or `.dat` | Table for pgfplots, see [Format of the ablation tables](#format-of-the-ablation-tables). |
| `logs/run/`, `logs/frame/`, `logs/table/` | Output of the jobs. |

`<algorithm>` is `skvi` or `sakc`, and `<ALGO>` the same name in capitals. `<first>__<second>`
is the grid point: the values of the two swept flags, in the order of the columns of the
table. For SKVI these are the number of actions and the number of training epochs
(`71__75`), for SAKC `v_lr` and `q_lr` (`0.0001__0.0005`). `<w>` is the smoothing window.
The folder `tables/` has the layout of the folder `csv/` of `ABLATIONS.md`, which that
guide gives as the layout of the tables in the sources of the paper.

The data frame job checks its data frame against the run directories it was built from,
as for the episodic returns, and here the two swept values as well: it fails when the
seed or a swept value in the name of a run folder is not the one of its directory, and
names the directory. A run folder in the directory of another grid point would otherwise
enter the table as a run of that grid point.

### Format of the ablation tables

A table is written in the format of its file name, by `process_skvi_ablations.py` or
`process_sakc_ablations.py`:

- `.csv`, the default, is comma-separated with a header row of column names. It is the
  format for new figures: pgfplots reads it with `col sep=comma`, takes the columns by
  their names and needs `mesh/rows=6`, as `ABLATIONS.md` shows.
- `.dat` is the layout of the tables that the figure sources of the paper read by the
  position of their columns: space-separated, with the header `x y z` and an empty line
  after every six rows. `tables/` has the layout of `data/ablations/` of the paper sources,
  so its `.dat` tables can be put there and the figure sources stay as they are
  (*executed* with tables of made-up returns on the full grids:
  `ablation_figure_skvi.tex` and `ablation_figure_sakc.tex` of the paper compiled
  unchanged).

The target `ablations` makes the format of the setting `table_format`; a table of either
format can be named by its path, and both can lie next to each other (*executed* as dry
runs and on made-up returns):

```bash
uvx --python 3.12 snakemake --cores 4 ablations --config 'ablations={table_format: dat}'

uvx --python 3.12 snakemake --cores 4 results/ablations/tables/SAKC/window_3/lorenz_ablation.dat
```

### Grid values in paths

A value of a grid has one spelling: the one Python prints for the number in
`configurations/ablations.json` (`0.0005`, not `5e-4` or `0.00050`). The workflow writes this
string into the path of the run and passes it to the launcher. The launcher reads it as a
number and prints the number into the command line of the algorithm, and SAKC prints it
into the name of its run folder; both give the same string again. `dataframe_creator.py`
reads the number from that name, and `process_sakc_ablations.py` finds the runs of a grid
point by comparing it with the numbers of the grid. `tests/test_run_ablations.py` checks for
every value of both grids that reading and printing returns the string, and that the number
read equals the number of the grid.

The wildcards of the grid point match these strings only: a path with `1e-4`, `0.00010` or
`71.0` has no rule (*executed* as dry runs). A path that joins an algorithm with a value of
the grid of the other algorithm is refused when the jobs are worked out (*executed* as a
dry run).

### Commands

Commands marked *executed* were run while this workflow was written, with Snakemake 9.27.0
and a results directory outside of the repository: as a dry run where noted, and otherwise
with the settings under [Short runs of the ablations](#short-runs-of-the-ablations), which
are eight runs of SAKC with 400 steps each (2 min 44 s with one core). SKVI was run through
the workflow once: one grid point (71 actions, 75 epochs) on the linear system, one seed,
400 steps (5 min 55 s).

**Dry run** (*executed* with the default settings: 1,440, 8 and 8 jobs). Lists the jobs
with their commands, without running them:

```bash
uvx --python 3.12 snakemake -n ablations
```

**Full run with N cores** (*not executed*). A run job takes one core and declares the
memory of its algorithm in `configurations/episodic_returns.json`, 4000 MB for SKVI and
2000 MB for SAKC; `--resources mem_mb=...` limits the sum over the running jobs:

```bash
uvx --python 3.12 snakemake --cores 8 --resources mem_mb=16000 --keep-going ablations
```

What is said for the episodic returns about a limit below 4000 and about the place of the
target after `--resources` holds here as well.

The process of the run of SKVI named above held 2,886,952 kB (2.8 GiB) at its peak, sampled
every two seconds. `run_ablations.py` gives about 3.5 GB
for SKVI on the fluid flow with the tuned configuration. The runs of SKVI with more than
71 actions were not measured, nor those of the ablation on the fluid flow and the Lorenz
system, so it is not known whether 4000 MB cover every grid point.

For Slurm, the command of the episodic returns applies with the target `ablations`
(*not executed*; a dry run with version 2.8.0 of the plugin listed the 1,457 jobs with
the memory they declare).

**Part of the campaign** (*executed*: the first form on short runs, the commands below as
dry runs only). Restrict the target `ablations` (see
[Settings of the ablations](#settings-of-the-ablations)); the first command below is 720 runs,
4 data frames and 4 tables, the second 20 runs, 1 data frame and 1 table:

```bash
uvx --python 3.12 snakemake --cores 4 ablations --config 'ablations={only_algorithms: [sakc]}'

uvx --python 3.12 snakemake --cores 4 ablations \
    --config 'ablations={only_algorithms: [sakc], only_benchmarks: [Lorenz-v0],
                         only_values: {v_lr: [0.0001, 0.0005], q_lr: [0.01, 0.05]}}'
```

or name what is wanted: a table (180 runs, 1 data frame, 1 table) or the directory of a
run (1 run):

```bash
uvx --python 3.12 snakemake --cores 4 results/ablations/tables/SAKC/window_3/lorenz_ablation.csv

uvx --python 3.12 snakemake --cores 1 results/ablations/runs/sakc/lorenz/0.0005__0.01/41
```

A data frame holds the runs of the selection of the call that made it. A call with another
selection of values or seeds makes the data frame and its tables again from the runs that
are selected then (*executed* as a dry run, with one seed fewer).

**Smoothing windows** (*executed* on short runs). The tables of the paper use a window of
three returns; the windows 1 and 5 were kept as a check of that choice. The window enters
the table job only:

```bash
uvx --python 3.12 snakemake --cores 4 ablations --config 'ablations={smoothing_windows: [1, 3, 5]}'
```

After a call with the default window, this call ran the table jobs of the windows 1 and 5
and no other job; runs, data frames and the tables of window 3 kept their time stamps. A
table of any window can also be named as a file, as above.

The tables of a window that is not listed any more stay where they are, and are not made
again when their data frame is: after a call with another selection, more seeds or other
steps they describe the data frame from before (*executed* on made-up logs: after a seed
was added with the default window only, the table of window 1 kept its time stamp and its
content). List the windows again, or delete the folders `window_<w>` that are not wanted.

**Resuming** (*executed* for the cases marked so). Give the same command again. Jobs whose
results exist are not repeated (*executed*: nothing was done, and no file changed).

- A run that fails: Snakemake removes its directory, and its console output is kept in
  `results/ablations/logs/run/<algorithm>/<benchmark>/<first>__<second>/<seed>.log`
  (*executed* with a seed that NumPy refuses).
- Interrupted and killed calls: the cases listed for the episodic returns are properties of
  Snakemake and of a rule of the same kind. They were *not executed* again for this workflow.
- A deleted table is made again from its data frame (*executed*).
- A deleted data frame or run directory is not made again as long as the tables made from
  it exist: see [Missing intermediate results](#missing-intermediate-results), with the
  commands that were *executed* for this workflow.

### Short runs of the ablations

A corner of a grid, two seeds, short runs and one benchmark, set on the command line
(*executed*, with another results directory):

```bash
uvx --python 3.12 snakemake --cores 1 ablations --config results_dir=/tmp/smoke \
    'ablations={only_algorithms: [sakc], only_benchmarks: [LinearSystem-v0],
                only_values: {v_lr: [0.0001, 0.0005], q_lr: [0.0001, 0.0005]},
                seeds: [1, 21], total_timesteps: 400}'
```

or in a file that is given with `--configfile` (*executed*; both forms give the same jobs):

```yaml
ablations:
  only_algorithms: [sakc]
  only_benchmarks: [LinearSystem-v0]
  only_values:
    v_lr: [0.0001, 0.0005]
    q_lr: [0.0001, 0.0005]
  seeds: [1, 21]
  total_timesteps: 400
```

The table of a part of a grid has all 36 rows, with `nan` at the grid points without runs;
`ABLATIONS.md` describes how pgfplots reads such a table. For both algorithms, give the
values of all four flags under `only_values` and leave `only_algorithms` out.

### Settings of the ablations

| Key | Default | Meaning |
|---|---|---|
| `ablations: total_timesteps` | `null` | Environment steps per run; `null` leaves the 50,000 of the algorithms. |
| `ablations: only_algorithms` | `null` | List of algorithms the target `ablations` is restricted to: `skvi`, `sakc`; `null` for both. |
| `ablations: only_benchmarks` | `null` | List of benchmarks it is restricted to, as `LinearSystem-v0`; `null` for all. |
| `ablations: table_format` | `csv` | Format of the tables that the target `ablations` makes: `csv` or `dat`, see [Format of the ablation tables](#format-of-the-ablation-tables). |
| `ablations: only_values: <flag>` | all six values | List of the values of a swept flag (`num_actions`, `num_training_epochs`, `v_lr`, `q_lr`) that are run and enter the data frames, spelled as in `configurations/ablations.json`; `null` for all six. |
| `ablations: seeds` | 1, 21, 41, 61, 81 | Seeds of the runs at every grid point. |
| `ablations: smoothing_windows` | `[3]` | Windows `w` for which the target `ablations` makes tables: the number of returns taken from the end of every run. |
| `episodic_returns: algorithms: <algorithm>: mem_mb` | SKVI 4000, SAKC 2000 | Memory a run declares, in MB; shared with the episodic returns. |

The settings are checked as those of the episodic returns, see [Settings](#settings): a
key that `configurations/ablations.json` does not have and a flag under `only_values` that
no grid has are refused, `null` is written as the word on the command line, and a list
holds at least one entry and none twice (*executed* as dry runs). `ablations: algorithms`
is refused when it is not as the file lists it (*executed* as dry runs with another grid
value and with one more algorithm).

The grids are not a setting. The launcher refuses values that are not in
`configurations/ablations.json`, and `process_skvi_ablations.py` and
`process_sakc_ablations.py` write the 36 rows of the grids of the paper, which they hold
themselves.

### Runs made without the workflow

Runs made with `run_ablations.py` can be handed to the workflow: copy each run folder to
`results/ablations/runs/<algorithm>/<benchmark>/<first>__<second>/<seed>/runs/<SKVI or SAKC>/<run name>/`.
No flag is needed. The data frame job fails when a run folder lies in the directory of
another seed or grid point than the one in its name. Snakemake notes that `ablations_run`
has jobs without metadata and does not make these runs again, also not when
`total_timesteps` is changed (*executed* with the logs of 20 earlier runs: 4 of SKVI on the
linear system, 16 of SAKC on the linear system and the double well, 2,200 steps each).

### What was compared

- **Data frames and tables.** From these 20 logs, the 3 data frames and the 9 tables of the
  windows 1, 3 and 5 of the workflow were identical, byte for byte, to those made by hand
  with the commands of `ABLATIONS.md`. Both paths read the same logs, so this compares the
  data frames and the tables, not the runs. A run on the double well logs one return in
  2,200 steps, so its three tables are equal to each other; those of the linear system
  differ between the windows.
- **Commands of the launcher.** `run_ablations.py --dry_run` prints the same 1,440 commands
  before and after the grids and seeds were moved to `configurations/ablations.json`.
- **Runs.** What is said under [What repeats exactly](#what-repeats-exactly) about the number
  of threads and about SAKC holds for these runs as well; it was not checked again.

## t-SNE

```
tsne_identify   the tensors of one benchmark, identified and stored       4 jobs
tsne_embed      one embedding of the stored tensors of all benchmarks     1 job
```

Both steps are calls of `koopmanrl_utils/tsne_koopman_tensor.py`, which
`koopmanrl_utils/TSNE.md` describes: an identification job calls it with `--identify_only`
for one benchmark, the embedding job with `--embed_only` for all of them. They are separate
jobs so that another embedding is made from the stored tensors, and the tensors of one
benchmark without those of the others: a changed setting of the embedding makes the
embedding job again and leaves the tensors as they are. With one core the workflow gives
the tensors and the tables of one call of the script on one thread, byte for byte (see
[What repeats and what does not](#what-repeats-and-what-does-not)).

The benchmarks are read from `configurations/episodic_returns.json`. The sweeps are defined
in the script. `configurations/tsne.json` holds no lists: it names the arguments of the
script that the workflow passes on, by the step that reads them.

Results are written to `results/tsne/`:

| Path | Content |
|---|---|
| `tensors/tensors_<benchmark>.npz` | The tensors of one benchmark, the result of an identification job. |
| `tables/<benchmark>_tsne.csv` | Table for pgfplots. Its first columns `index`, `x-val`, `y-val` are the columns that the TikZ source of the paper reads; `TSNE.md` lists the others. |
| `tables/tsne.csv`, `separation.csv`, `settings.json`, `t_sne_figure.tex`, `tsne_preview.pdf`, `tsne_preview.png` | The other files of the script: all rows in one table, the scores, the arguments of the embedding job, the pgfplots source of the figure and a preview. |
| `tables/tensors_<benchmark>.npz` | Copy of the tensors that the tables were made from. The target `tsne` asks for the copies too, so a missing copy makes the embedding job again (*executed*). |
| `logs/identify/<benchmark>.log`, `logs/embed/embed.log` | Output of the jobs. |

`<benchmark>` is `linear_system`, `fluid_flow`, `lorenz` or `double_well`. `tables/` is the
output directory of the script as `TSNE.md` describes it, so the figure is compiled there
with `pdflatex t_sne_figure.tex` (*executed* on a copy of the five files: one page). The
script reads the tensors from its output directory, which is why the embedding job copies
them there. `tables/` can be deleted and is made again from `tensors/` with the same
tables. `tensors/` is made again with the same tensors on the same machine with the same
installed libraries; on another machine the last digits of the tensors, and with them the
coordinates of single points, have to be expected to differ, so keep `tensors/` with a
figure.

Compiling the figure inside `tables/` leaves `t_sne_figure.pdf` and the other files of
`pdflatex` there. They are not results of the workflow: a later embedding replaces the
tables and leaves these files as they are, so the PDF shows the embedding from before
until it is compiled again (*executed* with a file of that name: it was still there after
the embedding job had run again).

### Commands

Commands marked *executed* were run while this workflow was written, with Snakemake 9.27.0,
one core and a results directory outside of the repository, on a machine with two cores.
The full default sweep and the full `orders` sweep were made with the workflow.

**Dry run** (*executed* with the default settings: 4 and 1 jobs):

```bash
uvx --python 3.12 snakemake -n tsne
```

**Default run** (*executed*: 2 min 12 s). This is the sweep
`inferred_layout` with the default embedding, as a call of the script without arguments
makes it:

```bash
uvx --python 3.12 snakemake --cores 1 tsne
```

An identification job took 18 to 37 s and the embedding job 17 s. The process of a job
held 815 to 875 MiB at its peak; a job declares 2000 MB. With more cores the identification
jobs of different benchmarks run at the same time (*not executed*).

**A different embedding from the same tensors** (*executed* with the command below on the
tensors of the default sweep, and with `embedding: {units: box, subtract_persistence: true}`
on those of the `orders` sweep):

```bash
uvx --python 3.12 snakemake --cores 1 tsne --config 'tsne={embedding: {perplexity: 50}}'
```

Only the embedding job ran; the files of `tensors/` kept their checksums and time stamps.
`tables/` holds one embedding, so its files are replaced. The command without the setting
makes the default embedding again, with the same tables as before, byte for byte
(*executed*). The names under `embedding` are those of the arguments of the script (see
[Settings of the t-SNE](#settings-of-the-t-sne)).

**The `orders` sweep, into a results directory of its own** (*executed*: 5 min 40 s, of
which 149 s for the fluid flow and 118 s for the Lorenz system; 805 to 904 MiB per job).
The directory below lies in `results/`, which git ignores:

```bash
uvx --python 3.12 snakemake --cores 1 tsne --config results_dir=results/orders \
    'tsne={identification: {sweep: orders}}'
```

The settings belong to the results directory and have to be given with every call for it.
The command without `'tsne={...}'` and with `results_dir=results/orders` would identify the
default sweep in that directory and replace its tensors (*executed* as a dry run: 4 and 1
jobs). A file given with `--configfile` keeps them together (*executed* as a dry run after
the run above: nothing to be done):

```yaml
results_dir: results/orders
tsne:
  identification:
    sweep: orders
```

**Part of the benchmarks** (*executed* with the command below: 2 and 1 jobs, 56 s):

```bash
uvx --python 3.12 snakemake --cores 1 tsne \
    --config 'tsne={only_benchmarks: [LinearSystem-v0, DoubleWell-v0]}'
```

The tensors of the selected benchmarks are embedded together, and the figure source reads
their tables only. The embedding job removes from `tables/` the tables and the copies of
the tensors of the benchmarks that are not selected, since an earlier embedding may have
left them there with other coordinates; `tensors/` is not touched. *Executed* in a directory
with the results of all four benchmarks: the call with two benchmarks ran the embedding job
only and left two tables, and the call for all four after it ran the embedding job only
and gave the first four tables again, byte for byte.

**Resuming** (*executed* for the cases marked so). Give the same command again. Jobs whose
results exist are not repeated (*executed*: nothing was done and no file changed, also
after a change of the setting `python`).

- A deleted table, or a deleted `tables/`: the embedding job is made again from `tensors/`
  (*executed* for one table and for the folder: the same tables as before, byte for byte).
- A deleted file of `tensors/`, together with the tables: the tensors of that benchmark are
  identified again and the embedding job is made again; the other benchmarks are not
  identified again (*executed* for the Lorenz system: the other three files kept their
  checksums and time stamps; the tensors of the Lorenz system and all tables were the ones
  from before, byte for byte).
- A deleted file of `tensors/` while the tables exist: nothing is done, see
  [Missing intermediate results](#missing-intermediate-results).
- A job that fails: Snakemake removes its outputs, and its log keeps the message of the
  script (*executed* with an argument under `embedding` that the script does not have).
  For the embedding job these are the files of `tables/`, which are made again from
  `tensors/`. For an identification job it is the file of tensors of its benchmark, which
  has to be identified again. The sweep is therefore checked before any job is made (see
  [Settings of the t-SNE](#settings-of-the-t-sne)).
- Interrupted and killed calls: the cases listed for the episodic returns are properties of
  Snakemake. They were *not executed* for this workflow.

### Settings of the t-SNE

| Key | Default | Meaning |
|---|---|---|
| `tsne: only_benchmarks` | `null` | List of benchmarks that are identified and embedded, as `LinearSystem-v0`. |
| `tsne: identification: sweep` | `inferred_layout` | The sweep: `inferred_layout` or `orders`. |
| `tsne: identification: <argument>` | `null` | The other arguments that the identification reads: `state_orders`, `action_orders`, `num_paths`, `num_steps_per_path`, `seeds` and `seed0`, which only the `orders` sweep uses, `linear_system_seed`, and `lstsq_driver`. |
| `tsne: embedding: <argument>` | none | Arguments of the embedding: `common_basis`, `double_well_coordinates`, `units`, `subtract_persistence`, `shared_coordinates_only`, `scaling`, `metric`, `perplexity`, `tsne_seed`, `tsne_init`, `tsne_method`, `neighbours`. |
| `tsne: mem_mb` | 2000 | Memory a job declares, in MB. |

The arguments, their values and their defaults are those of the table "Arguments" of
`TSNE.md`. An argument that is `null` or not listed is not passed, which leaves the default
of the script; on the command line `null` is written as the word. A list is written as
`double_well_coordinates: [1, 2]` and a switch as `subtract_persistence: true` (*executed*
as a dry run; `units`, `perplexity` and `subtract_persistence` in runs). The settings are
checked before any job is made (*executed* as dry runs): a key of `tsne` that
`configurations/tsne.json` does not have is refused, and `only_benchmarks` holds at least
one benchmark and none twice.

- The arguments under `identification` are part of the command of the identification jobs.
  Setting one makes these jobs again, also when the value is the default of the script
  (*executed* as a dry run with `seeds: 8`). They are passed to the embedding job as well,
  which does not read them but writes them into `settings.json`, so that file names the
  sweep of the tensors.
- Under `identification` only the arguments that `configurations/tsne.json` lists are taken.
  Any other name is refused when the jobs are worked out, also an argument of the
  embedding such as `perplexity`, which would identify every tensor again for nothing. The
  values of `sweep` (`inferred_layout`, `orders` or `null`) and of `lstsq_driver` (`gelsd`,
  `gelss`, `gelsy` or `null`) are checked there as well. A misspelled sweep or driver
  therefore stops the call before an identification job has removed the tensors it was to
  replace (*executed* as dry runs, and with `sweep: nonsense` in a directory with results:
  no file changed).
- The arguments under `embedding` are part of the command of the embedding job only.
- An argument of the identification under `embedding` is refused when the jobs are worked
  out, since it would make the embedding job again from the same tensors and change nothing
  but `settings.json`. The arguments that the rules set are refused under both keys:
  `environments`, `output_dir`, `identify_only`, `embed_only`, `resume` (*executed* as dry
  runs). `tests/test_tsne_koopman_tensor.py` checks that `identification` in
  `configurations/tsne.json` lists the arguments that the identification of the script
  reads, and that the benchmarks and their order are those of the script.
- Under `embedding`, a name that the script does not have is not noticed before the job
  runs: the embedding job fails with the message of the script in its log.

### What repeats and what does not

The measurements behind this section are in the section "Reproducibility" of `TSNE.md`;
what was run with the workflow is marked *executed*.

- **The workflow against a call of the script.** *Executed* for the default sweep: the
  workflow with one core, and one call of the script with `OMP_NUM_THREADS=1` and without
  further arguments, gave the same four files of tensors, the same four tables, and the same
  `tsne.csv`, `separation.csv`, `t_sne_figure.tex` and `tsne_preview.png`, byte for byte.
  `settings.json` differs between calls in the output folder, the run time and the peak
  memory, and `tsne_preview.pdf` differs as a file.
- **The identification.** The script solves the regression with the LAPACK driver `gelsd`,
  which returns the same tensor in every call, and seeds the data of every benchmark by
  itself. A job for one benchmark therefore identifies the tensors that a call for all
  benchmarks identifies, and an identification that is made again gives the file from
  before (*executed*: the tensors of the Lorenz system were removed with the tables and
  made again, and all four files of tensors were made again after a change of the script;
  the files and the tables had the checksums from before both times). With
  `lstsq_driver: gelsy`, the solver call of the package, this does not hold: its result
  differs from call to call, in the last digits for most tensors and altogether for two
  tensors of the Lorenz system, and the tables with it. `TSNE.md` has the numbers.
- **The embedding.** From the same files of `tensors/`, the embedding job gives the same
  tables on every call (*executed*: after a change of a setting and back, with
  `--forcerun tsne_embed`, and after `tables/` was removed).
- **Number of threads.** Snakemake sets `OMP_NUM_THREADS` and the related variables to the
  number of threads of a job, which is one for both rules. Tensors and tables are exact for
  a given number of threads, not across them: between one and two threads the tensors of the
  default sweep differed in the last digits (by a relative 2e-13 in the median and up to
  4e-6), and single points moved by up to 14% of the diagonal of the embedding (median 4%),
  while the clusters and their scores stayed (silhouette of all tensors 0.725 and 0.730).
  A call of the script by hand reproduces the files of the workflow only with
  `OMP_NUM_THREADS=1`.
- **Another machine.** Another machine, or another build of torch and its LAPACK, was not
  tried and has to be expected to change the last digits of the tensors in the same way as
  another number of threads does. The clusters do not depend on it; the coordinates of
  single points do.
- **What makes the tensors again.** Snakemake makes an identification job again when its
  file in `tensors/` is missing and needed or named, when an argument under `identification`
  changed, and when the script changed, which both rules take as an input (*executed* as a
  dry run and as a run after a change of the script: 4 and 1 jobs). The script is an input
  under its full path, so with the repository at another path the jobs are made again as
  well. A change of a rule does
  not make its jobs again (see [What makes a job again](#what-makes-a-job-again)). `-n`
  lists the jobs of a call with their reasons. With `--rerun-triggers mtime`, Snakemake goes
  by missing files and time stamps only and does not notice changed arguments. An embedding
  job that is given other arguments of the identification than its tensors were identified
  with stops with a message of the script that names the difference (*executed* with
  `sweep: orders` on the tensors of the default sweep).

---

## Adding a Workflow

A workflow `<name>` consists of

- `workflow/rules/<name>.smk`, included in `workflow/Snakefile`, with a target rule `<name>`
  that is added to the inputs of the rule `all`;
- `configurations/<name>.json` with one top-level key `<name>`, loaded in the rule file with
  `configfile: os.path.join(CONFIG_DIR, "<name>.json")`. Lists that a script of
  `koopmanrl_utils/` needs as well are kept in this file only, and the script reads it.
  Lists that the file of another workflow holds already are not repeated: the rule file
  loads that file with a `configfile:` line of its own and reads them from `config`, as
  `rules/ablations.smk` does for the algorithms and the benchmarks. Snakemake then names
  the file twice among its config files.

Conventions, with `rules/episodic_returns.smk` as the example:

- **Names.** Rules are called `<name>_<step>`; constants of the rule file carry a short
  prefix (`ER_`), since all rule files share one namespace.
- **Paths.** Everything is written below `RESULTS/<name>/`, and the logs of the jobs below
  `RESULTS/<name>/logs/<step>/`. Paths are built from `RESULTS`, never written out.
- **Wildcards.** Two wildcards are constrained for all workflows in `rules/common.smk`:
  `{seed}`, a whole number without leading zeros, and `{ext}`, the format of a table
  (`csv` or `dat`). Every other wildcard is constrained in the rules that use it
  (`wildcard_constraints:`), with `alternatives(<names>)`. A number in a path gets one
  spelling, which is also passed on the command line; see
  [Grid values in paths](#grid-values-in-paths).
- **Runs.** One job per run, with `directory(...)` of the run as its output, so that the
  algorithm, which writes `runs/` and `saved_models/` relative to its working directory,
  shares nothing with other runs and Snakemake can remove a failed run as a whole.
- **Project code.** Shell commands start with `{PYTHON} -m <module>` and send their output
  to `{log:q}`. Every path in a shell command is written with `:q` (`{output:q}`,
  `{input:q}`, `{params.output_dir:q}`), so that a results directory with a blank in its
  name works. `PYTHON` is deliberately not passed through `params:`, so that a different
  path of the repository does not make every run out of date.
- **Scripts as inputs.** A job that builds a data frame or a table takes the script it
  calls as an input (`FRAME_SCRIPT`, `script_file(<module>)` of `rules/common.smk`), so it
  is made again when the script changes. A run job takes no code as an input; see
  [What makes a job again](#what-makes-a-job-again).
- **Data frames.** `data_frame(run_dirs, pattern, fields, runs_dir, options, frame, log)`
  of `rules/common.smk` calls `koopmanrl_utils.dataframe_creator` on the run directories of
  such jobs and checks the result against them: `pattern` is the path pattern of a run
  directory, and `fields` names the wildcards that the fields of an entry are compared
  with, as `{"seed": "seed"}`.
- **Resources.** Every rule declares `threads` and `resources: mem_mb`.
- **Settings.** Values that may change between calls are read from `config["<name>"]` and
  passed to the jobs through `params:`. A value inside the braces of
  `--config '<name>={...}'` arrives as a string, also a number and the word `null`, and a
  value of a file given with `--configfile` with its type, so the rule file has to take
  both. The helpers of `workflow/helpers.py` do, and every setting is read through one of
  them when the jobs are worked out: `known_keys` refuses keys that the JSON file does not
  have (`defaults("<name>")` of `rules/common.smk` reads the file), `selection` reads a list
  of names or `null`, `whole_number` and `whole_numbers` read numbers and lists of seeds, `choice` reads one of several names, and
  `script_options(options, what, refused, known)` writes a mapping of arguments of a
  script as its command line; `rules/tsne.smk` uses the last. Their messages name the
  setting, what it takes and what was given.
- **Steps that are worth keeping apart.** A step whose result several later steps or
  settings build on gets a job and a folder of its own, and the steps after it read its
  stored result, so that they can be made again without it. In `rules/tsne.smk` this is the
  identification of the tensors.
- **Functions.** Functions that need nothing of Snakemake are kept in
  `workflow/helpers.py`, which `rules/common.smk` imports and
  `tests/test_workflow_helpers.py` tests in the project environment; functions that need
  `config`, `shell` or `workflow` are kept in `rules/common.smk`. `--lint` reports a rule
  file that defines functions next to its rules; `rules/ablations.smk` uses tables and
  `lambda` expressions instead.

Checks before a change is committed:

```bash
uvx --python 3.12 snakemake --lint
uvx --python 3.12 snakemake -n all
uvx snakefmt workflow
uv run pytest tests/test_workflow_helpers.py
```

`--lint` reports two things for every rule with a shell command, which are left as they
are: no conda environment or container is given, since the jobs use the project environment
of `uv`, and `PYTHON` is used from outside of the rule, for the reason given above. With
these reports it ends with exit status 1, so the status does not tell whether there is a
new report; read its output.
