# Trajectory Visualisation Pipeline

This document describes the end-to-end pipeline for generating publication-quality
static figures (and optionally animated GIFs) of controlled and uncontrolled
trajectories for KoopmanRL environments.

---

## Overview

The pipeline consists of two sequential steps:

```
Step 1 – generate_trajectories.py
    Runs a trained policy (and a zero/baseline policy) in the gym environment
    and saves trajectories, actions, and costs as .npy (and optionally .dat) files.

Step 2a – generate_trajectory_figure.py   (static PNG)
    Reads the saved .npy files and renders a 3D publication-quality figure.

Step 2b – generate_gifs.py                (animated GIF — optional)
    Reads the same .npy files and renders frame-by-frame GIFs.
```

Steps 2a and 2b are independent and can be run in either order or both.

Supported environments: `FluidFlow-v0`, `Lorenz-v0`, `DoubleWell-v0`.
`LinearSystem-v0` is excluded as a plotting target due to it resetting itself at the start of each interaction with it, and as such not being amenable to clean visualization.

Run all three scripts from the root of the repository. Step 1 resolves the
hyperparameter configuration as the relative path
`configurations/<algo>_<env_slug>_hparams.json` and loads checkpoints from
`./saved_models/...`; Steps 1 and 2 write into `video_frames/` relative to the
working directory.

---

## Checkpoints

The `sakc`, `skvi` and `sac` policies load the weights that a training run saved
under `saved_models/` of the directory it was started from:

| `--algo` | Checkpoint file | Saved by |
|---|---|---|
| `sakc` | `saved_models/SAKC/<env_id>/sakc_chkpts_<seed>_<unix time>/step_<N>.pt` | `koopmanrl.soft_actor_koopman_critic`, at step 1 and every 1,000 steps |
| `skvi` | `saved_models/SKVI/<env_id>/skvi_chkpts_<seed>_<unix time>/epoch_<N>.pt` | `koopmanrl.soft_koopman_value_iteration`, at epoch 1 and every 10 epochs |
| `sac` | `saved_models/<env_id>/sac_chkpts_<unix time>/step_<N>.pt` | `koopmanrl.sac_continuous_action`, at step 1 and every 1,000 steps |

`--chkpt_timestamp` is the part of the folder name after `sakc_chkpts_`,
`skvi_chkpts_` or `sac_chkpts_`, and `--chkpt_step` / `--chkpt_epoch` is the
`<N>` of the file.

A run started by hand from the repository root saves its checkpoints where
Step 1 looks for them. The reproduction launchers start every run in a working
directory of its own, so their checkpoints land elsewhere:

| Launcher | Checkpoints |
|---|---|
| `koopmanrl_utils.run_optimized_experiments` | `episodic_returns_results/saved_models/` (`<output_dir>/saved_models/`) |
| Snakemake workflow (`workflow/`) | `results/episodic_returns/runs/<algorithm>/<benchmark>/<seed>/saved_models/` |

To plot such a run, copy or link the checkpoint folder of that run into
`./saved_models/` at the repository root, keeping the
`SAKC/<env_id>/...`, `SKVI/<env_id>/...` or `<env_id>/...` layout, for example:

```bash
mkdir -p saved_models/SAKC/FluidFlow-v0
cp -r results/episodic_returns/runs/sakc/fluid_flow/6597/saved_models/SAKC/FluidFlow-v0/sakc_chkpts_6597_* \
    saved_models/SAKC/FluidFlow-v0/
```

Alternatively, run Step 1 from the run's working directory and pass the
configuration explicitly with
`--config_file <repository root>/configurations/<algo>_<env_slug>_hparams.json`;
without it the configuration is not found and the defaults are used.

---

## Step 1 — Generate trajectories

**Module:** `koopmanrl_utils.movies.generate_trajectories`

```bash
uv run -m koopmanrl_utils.movies.generate_trajectories [flags]
```

### What it does

1. Validates the environment (must be one of the three supported envs).
2. Auto-loads the best-found hyperparameter configuration from
   `configurations/<algo>_<env_slug>_hparams.json` (SAKC and SKVI only).
3. Builds a vectorised gym environment (`gym.vector.SyncVectorEnv`).
4. Instantiates three policies: the **main policy** (`--algo`), a
   **baseline policy** (`--baseline_algo`), and an explicit **zero policy**
   (always included to serve as the uncontrolled reference).
5. Rolls out each policy for `--num_trajectories` episodes of
   `--num_steps` steps, seeding the RNG identically before each rollout so
   that all three policies start from the same initial condition.
6. Asserts shared initial conditions across the three trajectory sets.
7. Saves the results under `<output_dir>/<env_id>_<unix_timestamp>/`
   (`<output_dir>/<env_id>_<run_label>/` with `--run_label`):
   - `zero_policy_trajectories.npy`, `_actions.npy`, `_costs.npy`
   - `main_policy_trajectories.npy`, `_actions.npy`, `_costs.npy`
   - `baseline_policy_trajectories.npy`, `_actions.npy`, `_costs.npy`
   - `metadata.npy` — pickled dict with `env_id` and policy name strings
   - `double_well_vector_field.npy` (`DoubleWell-v0` only) — the deterministic
     drift of the uncontrolled double well on a
     `--vector_field_resolution` × `--vector_field_resolution` grid over the
     state range, with columns `x`, `y`, `dx`, `dy`, `dxn`, `dyn` (raw and
     L2-normalised drift). It is meant for TikZ; Step 2a does not read it.
8. Optionally writes the same data as tab-separated `.dat` files suitable
   for direct ingestion by TikZ / PGFPlots (`--emit_dat`), including
   `double_well_vector_field.dat` for `DoubleWell-v0`.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--env_id` | `str` | `FluidFlow-v0` | Gym environment ID. One of `FluidFlow-v0`, `Lorenz-v0`, `DoubleWell-v0`. |
| `--seed` | `int \| None` | from config | RNG seed for environment resets and policy sampling; for SKVI also the seed of the rebuilt Koopman tensor. Falls back to `123` if not in config. |
| `--torch_deterministic` | `bool` | `True` | Set `torch.backends.cudnn.deterministic`. Passing the flag sets it to `False`. |
| `--cuda` | `bool` | `True` | Use CUDA if available. Passing the flag disables CUDA. |
| `--num_trajectories` | `int \| None` | from config | Number of independent episodes to roll out. For SAKC/SKVI defaults to the config's `num-paths`; falls back to `1`. See [Trajectory count and length](#trajectory-count-and-length). |
| `--num_steps` | `int \| None` | from config | Steps per episode. For SAKC/SKVI defaults to the config's `num-steps-per-path`; otherwise the environment's `max_episode_steps` (2,000 for all three environments). |
| `--algo` | `str` | `sakc` | Main policy algorithm. Choices: `zero`, `random`, `lqr`, `skvi`, `sakc`, `sac`. |
| `--baseline_algo` | `str` | `zero` | Baseline policy for comparison. Choices: `zero`, `random`, `lqr`. |
| `--gamma` | `float` | `0.99` | Discount factor (used by LQR / SKVI). |
| `--alpha` | `float` | `1.0` | Entropy regularisation coefficient (used by LQR / SKVI). |
| `--num_actions` | `int` | `101` | Discrete action grid size for SKVI. |
| `--state_order` | `int \| None` | from config | Monomial order for the state observable (SKVI only). Loaded from config; falls back to `2`. |
| `--action_order` | `int \| None` | from config | Monomial order for the action observable (SKVI only). Loaded from config; falls back to `2`. |
| `--skvi_lr` | `float \| None` | from config | Value function learning rate (SKVI only). Loaded from config; falls back to `1e-3`. |
| `--skvi_koopman_num_paths` | `int \| None` | from config | Number of paths of the data the Koopman tensor is rebuilt from (SKVI only). Loaded from config (`num-paths`); falls back to `100`. Independent of `--num_trajectories`. |
| `--skvi_koopman_num_steps` | `int \| None` | from config | Steps per path of that data (SKVI only). Loaded from config (`num-steps-per-path`); falls back to `300`. Independent of `--num_steps`. |
| `--regressor` | `str` | `ols` | Koopman tensor regression method (SKVI only): `ols`, `sindy`, `rrr`, `ridge`. |
| `--chkpt_timestamp` | `str \| None` | — | Folder suffix of the checkpoint directory: `<seed>_<unix time>` for SAKC and SKVI (e.g. `6597_1768954004`), `<unix time>` for SAC. See [Checkpoints](#checkpoints). **Required** for `sakc`, `skvi` and `sac`. |
| `--chkpt_step` | `int \| None` | — | Checkpoint step number. **Required** for `sakc` and `sac`. |
| `--chkpt_epoch` | `int \| None` | — | Checkpoint epoch number. **Required** for `skvi`. |
| `--config_file` | `str \| None` | auto-resolved | Explicit path to a hparams JSON. Auto-resolved to `configurations/<algo>_<env_slug>_hparams.json` when omitted (SAKC/SKVI only). |
| `--output_dir` | `str` | `video_frames` | Root directory for output; a sub-folder `<env_id>_<unix_timestamp>` is created automatically. |
| `--run_label` | `str \| None` | — | Name the sub-folder `<env_id>_<run_label>` instead of `<env_id>_<unix_timestamp>`, e.g. `--run_label sakc_fig` gives `video_frames/FluidFlow-v0_sakc_fig/`. |
| `--emit_dat` | `bool` | `False` | Also write `.dat` files for TikZ / PGFPlots ingestion alongside the `.npy` files. |
| `--vector_field_resolution` | `int` | `20` | Grid points per axis of `double_well_vector_field.npy` (`DoubleWell-v0` only). |

### Config precedence

For `sakc` and `skvi`, `seed`, `num_trajectories`, `num_steps`, `state_order`,
`action_order`, `skvi_lr`, `skvi_koopman_num_paths` and `skvi_koopman_num_steps`
are auto-populated from the corresponding JSON file.  Any value given on the
CLI takes precedence:
**CLI > config file > built-in default**.

### Trajectory count and length

The configuration files hold no trajectory count or length for plotting:
`num_trajectories` and `num_steps` default to the configuration's
**Koopman identification budget**, `num-paths` and `num-steps-per-path`.
For SAKC on the fluid flow (`configurations/sakc_fluid_flow_hparams.json`)
that is 50 trajectories of 175 steps each, while an episode lasts 2,000 steps.

For a figure, pass both:

```bash
--num_trajectories 1 --num_steps 2000
```

Leaving `--num_steps` out gives the full episode (`max_episode_steps`, 2,000)
only for `zero`, `random`, `lqr` and `sac`, which read no configuration; for
`sakc` and `skvi` the configuration fills it in, and it cannot be reset to
`None` from the CLI. Do not exceed 2,000: the vectorised environment then
starts a new episode within the trajectory.

> **Note (SKVI):** The Koopman tensor is always rebuilt from the environment
> with the hyperparameters of the config file and `--seed`; no pre-saved tensor
> file is required.  Checkpoint files (`.pt`) only store the value function
> weights.  The rebuilt tensor matches the one of the training run only if
> `--seed` is the seed of that run (the `<seed>` in its folder name),
> `--skvi_koopman_num_paths` / `--skvi_koopman_num_steps` are its
> identification budget, and `--regressor` its regressor.  A run started with
> the config file and no `--seed` used the config's seed, which is also the
> default here; runs of the reproduction launchers used the seed of the run.

### Minimal examples

```bash
# SAKC on FluidFlow — one full-length trajectory; the seed comes from config
uv run -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 6597_1768954004 \
    --chkpt_step 50000 \
    --num_trajectories 1 \
    --num_steps 2000

# SKVI on Lorenz — Koopman tensor rebuilt automatically from config
# (125 training epochs: the last checkpoint is epoch_120.pt)
uv run -m koopmanrl_utils.movies.generate_trajectories \
    --env_id Lorenz-v0 \
    --algo skvi \
    --chkpt_timestamp 8953_1768956979 \
    --chkpt_epoch 120 \
    --num_trajectories 1 \
    --num_steps 2000

# LQR on DoubleWell — no config, explicit overrides
uv run -m koopmanrl_utils.movies.generate_trajectories \
    --env_id DoubleWell-v0 \
    --algo lqr \
    --baseline_algo zero \
    --num_trajectories 1 \
    --seed 42

# Emit .dat files for TikZ into a named folder
uv run -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 6597_1768954004 \
    --chkpt_step 50000 \
    --num_trajectories 1 \
    --num_steps 2000 \
    --run_label sakc_fig \
    --emit_dat
```

---

## Step 2a — Generate static figure

**Module:** `koopmanrl_utils.movies.generate_trajectory_figure`

```bash
uv run -m koopmanrl_utils.movies.generate_trajectory_figure [flags]
```

### What it does

Reads the `.npy` files written by Step 1 and produces a single
high-resolution 3D PNG figure.  Optionally overlays the uncontrolled
(zero-policy) trajectory, adds a quiver vector field of the uncontrolled
dynamics, marks the reference point and initial condition, and writes the
plotted data points as `.dat` files.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_folder` | `str` | **required** | Path to a folder produced by `generate_trajectories.py` (contains the `.npy` files). |
| `--seed` | `int` | `123` | RNG seed used when constructing the environment for axis limits and the vector field. |
| `--trajectory_idx` | `int` | `0` | Which trajectory index (0-based) from the saved array to plot. |
| `--plot_uncontrolled` | `bool` | `False` | Overlay the zero-policy (uncontrolled) trajectory in blue. |
| `--plot_vector_field` | `bool` | `False` | Overlay a quiver plot of the uncontrolled vector field. Skipped automatically for `DoubleWell` (stochastic dynamics). |
| `--vector_field_resolution` | `int` | `8` | Grid points per axis for the quiver plot. Higher values give a denser field but are slower to compute. |
| `--step_limit` | `int \| None` | `None` | Plot only the first N steps. `None` plots the full trajectory. |
| `--show_coordinate_frame` | `bool` | `True` | Axis labels, tick marks, and pane edges are shown by default. Passing the flag sets it to `False` and hides all axes (notebook-style). |
| `--dpi` | `int` | `300` | Figure resolution in dots per inch. |
| `--output_file` | `str \| None` | `<data_folder>/trajectory_figure.png` | Output PNG path. |
| `--emit_dat` | `bool` | `False` | Write the plotted trajectory points as `.dat` files alongside the PNG. Also writes `vector_field.dat` when `--plot_vector_field` is active. |
| `--view_elev` | `float` | `20.0` | 3D view elevation angle in degrees. |
| `--view_azim` | `float` | `45.0` | 3D view azimuth angle in degrees. |

### Output files (with `--emit_dat`)

| File | Columns | Description |
|---|---|---|
| `main_trajectory_plot.dat` | `step`, `x0`, `x1`, `x2` (or `potential` for DoubleWell) | Main-policy trajectory. |
| `zero_trajectory_plot.dat` | same | Zero-policy trajectory (only written with `--plot_uncontrolled`). |
| `vector_field.dat` | `X`, `Y`, `Z`, `dX`, `dY`, `dZ` | L2-normalised quiver vectors (only written with `--plot_vector_field`). |

### Examples

```bash
# Minimal — just the controlled trajectory
uv run -m koopmanrl_utils.movies.generate_trajectory_figure \
    --data_folder video_frames/FluidFlow-v0_1744000000

# Full publication figure (the coordinate frame is shown by default)
uv run -m koopmanrl_utils.movies.generate_trajectory_figure \
    --data_folder video_frames/FluidFlow-v0_1744000000 \
    --plot_uncontrolled \
    --plot_vector_field \
    --vector_field_resolution 8 \
    --view_elev 25 \
    --view_azim 60 \
    --emit_dat \
    --output_file figures/fluid_flow_trajectory.png
```

---

## Step 2b — Generate animated GIFs (optional)

**Module:** `koopmanrl_utils.movies.generate_gifs`

```bash
uv run -m koopmanrl_utils.movies.generate_gifs [flags]
```

### What it does

Reads the same `.npy` output folder as Step 2a and produces two GIFs per
trajectory (so a folder with 50 trajectories gives 100 GIFs):

- A 3D trajectory animation that grows the path step-by-step.
- A cost-ratio animation showing the main-policy cost divided by the
  zero-policy cost, smoothed with a moving average.

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data_folder` | `str` | `""` | **Required.** Folder produced by `generate_trajectories.py`. |
| `--seed` | `int` | `123` | RNG seed used when constructing the environment. |
| `--save_every_n_steps` | `int` | `100` | Write one animation frame every N environment steps. Lower values yield smoother but larger GIFs. |
| `--plot_uncontrolled` | `bool` | `False` | Overlay the zero-policy trajectory in the animation. |
| `--ma_window_size` | `int \| None` | env-specific | Moving average window for the cost-ratio plot. Defaults to `200` for all supported environments. |
| `--emit_dat` | `bool` | `False` | Write per-step cost data as a `.dat` file alongside each GIF. |

### Examples

```bash
# Basic GIF
uv run -m koopmanrl_utils.movies.generate_gifs \
    --data_folder video_frames/Lorenz-v0_1744000000

# With uncontrolled overlay, finer frames, and .dat export
uv run -m koopmanrl_utils.movies.generate_gifs \
    --data_folder video_frames/FluidFlow-v0_1744000000 \
    --save_every_n_steps 10 \
    --plot_uncontrolled \
    --emit_dat
```

---

## Supporting modules

The following modules are used internally by the pipeline scripts; they are
not intended to be invoked directly.

| Module | Role |
|---|---|
| `abstract_policy.py` | Abstract base class `Policy` with a single `get_action` method. All policy wrappers extend this. |
| `algo_policies.py` | Concrete policy wrappers: `LQR`, `SKVI`, `SAKC`, `SAC`. Each wraps the corresponding trained model and exposes `get_action`. |
| `default_policies.py` | `ZeroPolicy` (always returns 0) and `RandomPolicy` (samples from the action space). Used as the uncontrolled and baseline references. |
| `env_enum.py` | `EnvEnum` string enum mapping human-readable names to gym IDs. |
| `generator.py` | `Generator` class that owns the environment, policy, and RNG state, and implements `generate_trajectories(num_trajectories, num_steps_per_trajectory)`. |

---

## Complete worked example (SAKC on FluidFlow)

```bash
# 1. Generate one full-length trajectory (the seed is auto-loaded from config)
uv run -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 6597_1768954004 \
    --chkpt_step 50000 \
    --num_trajectories 1 \
    --num_steps 2000 \
    --emit_dat

# Note the output folder name printed to stdout, e.g.:
#   Saved .npy files to 'video_frames/FluidFlow-v0_1744123456'

# 2a. Static figure with vector field and coordinate frame
uv run -m koopmanrl_utils.movies.generate_trajectory_figure \
    --data_folder video_frames/FluidFlow-v0_1744123456 \
    --plot_uncontrolled \
    --plot_vector_field \
    --emit_dat \
    --output_file figures/fluid_flow.png

# 2b. Animated GIF (optional)
uv run -m koopmanrl_utils.movies.generate_gifs \
    --data_folder video_frames/FluidFlow-v0_1744123456 \
    --plot_uncontrolled \
    --save_every_n_steps 10
```

---

## Limitations

- **One trajectory per figure.** `generate_trajectory_figure.py` plots the
  single trajectory chosen with `--trajectory_idx`; there is no option to
  overlay several trajectories of a folder.
- **Checkpoint identity is recorded nowhere.** The configuration files hold no
  checkpoint folder, step or epoch, so `--chkpt_timestamp` and
  `--chkpt_step` / `--chkpt_epoch` always have to be given on the CLI; and
  `metadata.npy` stores only the environment and the policy names, not the
  checkpoint, seed or step count a folder was generated with. Use
  `--run_label` to keep track of them.
