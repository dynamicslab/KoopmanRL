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

---

## Step 1 — Generate trajectories

**Module:** `koopmanrl_utils.movies.generate_trajectories`

```bash
python -m koopmanrl_utils.movies.generate_trajectories [flags]
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
7. Saves the results under `<output_dir>/<env_id>_<unix_timestamp>/`:
   - `zero_policy_trajectories.npy`, `_actions.npy`, `_costs.npy`
   - `main_policy_trajectories.npy`, `_actions.npy`, `_costs.npy`
   - `baseline_policy_trajectories.npy`, `_actions.npy`, `_costs.npy`
   - `metadata.npy` — pickled dict with `env_id` and policy name strings
8. Optionally writes the same data as tab-separated `.dat` files suitable
   for direct ingestion by TikZ / PGFPlots (`--emit_dat`).

### Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--env_id` | `str` | `FluidFlow-v0` | Gym environment ID. One of `FluidFlow-v0`, `Lorenz-v0`, `DoubleWell-v0`. |
| `--seed` | `int \| None` | from config | RNG seed for environment resets and policy sampling. Falls back to `123` if not in config. |
| `--torch_deterministic` | `bool` | `True` | Set `torch.backends.cudnn.deterministic`. |
| `--cuda` | `bool` | `True` | Enable CUDA if available. |
| `--num_trajectories` | `int \| None` | from config | Number of independent episodes to roll out. Falls back to `1`. |
| `--num_steps` | `int \| None` | from config | Steps per episode. Uses environment's `max_episode_steps` when `None`. |
| `--algo` | `str` | `sakc` | Main policy algorithm. Choices: `zero`, `random`, `lqr`, `skvi`, `sakc`. |
| `--baseline_algo` | `str` | `zero` | Baseline policy for comparison. Choices: `zero`, `random`, `lqr`. |
| `--gamma` | `float` | `0.99` | Discount factor (used by LQR). |
| `--alpha` | `float` | `1.0` | Entropy regularisation coefficient (used by LQR / SKVI). |
| `--num_actions` | `int` | `101` | Discrete action grid size for SKVI. |
| `--state_order` | `int \| None` | from config | Monomial order for the state observable (SKVI only). Loaded from config; falls back to `2`. |
| `--action_order` | `int \| None` | from config | Monomial order for the action observable (SKVI only). Loaded from config; falls back to `2`. |
| `--skvi_lr` | `float \| None` | from config | Value function learning rate (SKVI only). Loaded from config; falls back to `1e-3`. |
| `--regressor` | `str` | `ols` | Koopman tensor regression method (SKVI only): `ols`, `sindy`, `rrr`, `ridge`. |
| `--chkpt_timestamp` | `str \| None` | — | Folder suffix of the checkpoint directory. Both SAKC and SKVI use `{seed}_{unix_timestamp}` (e.g. `1_1768954004`). **Required** for `sakc` and `skvi`. |
| `--chkpt_step` | `int \| None` | — | Checkpoint step number. **Required** for `sakc`. |
| `--chkpt_epoch` | `int \| None` | — | Checkpoint epoch number. **Required** for `skvi`. |
| `--config_file` | `str \| None` | auto-resolved | Explicit path to a hparams JSON. Auto-resolved to `configurations/<algo>_<env_slug>_hparams.json` when omitted (SAKC/SKVI only). |
| `--output_dir` | `str` | `video_frames` | Root directory for output; a timestamped sub-folder is created automatically. |
| `--emit_dat` | `bool` | `False` | Also write `.dat` files for TikZ / PGFPlots ingestion alongside the `.npy` files. |

### Config precedence

For `sakc` and `skvi`, `seed`, `num_trajectories`, `num_steps`, `state_order`,
`action_order`, and `skvi_lr` are auto-populated from the corresponding JSON
file.  Any value given on the CLI takes precedence:
**CLI > config file > built-in default**.

> **Note (SKVI):** The Koopman tensor is always rebuilt from the environment
> using the same hyperparameters stored in the config file; no pre-saved tensor
> file is required.  Checkpoint files (`.pt`) only store the value function
> weights.

### Minimal examples

```bash
# SAKC on FluidFlow — seed and trajectory count come from config
python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 1732368170 \
    --chkpt_step 50000

# SKVI on Lorenz — Koopman tensor rebuilt automatically from config
python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id Lorenz-v0 \
    --algo skvi \
    --chkpt_timestamp 1_1768956979 \
    --chkpt_epoch 150

# LQR on DoubleWell — no config, explicit overrides
python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id DoubleWell-v0 \
    --algo lqr \
    --baseline_algo zero \
    --num_trajectories 1 \
    --seed 42

# Emit .dat files for TikZ
python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 1732368170 \
    --chkpt_step 50000 \
    --emit_dat
```

---

## Step 2a — Generate static figure

**Module:** `koopmanrl_utils.movies.generate_trajectory_figure`

```bash
python -m koopmanrl_utils.movies.generate_trajectory_figure [flags]
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
| `--show_coordinate_frame` | `bool` | `True` | Show axis labels, tick marks, and pane edges. `False` hides all axes (notebook-style). |
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
python -m koopmanrl_utils.movies.generate_trajectory_figure \
    --data_folder video_frames/FluidFlow-v0_1744000000

# Full publication figure
python -m koopmanrl_utils.movies.generate_trajectory_figure \
    --data_folder video_frames/FluidFlow-v0_1744000000 \
    --plot_uncontrolled \
    --plot_vector_field \
    --vector_field_resolution 8 \
    --show_coordinate_frame \
    --view_elev 25 \
    --view_azim 60 \
    --emit_dat \
    --output_file figures/fluid_flow_trajectory.png
```

---

## Step 2b — Generate animated GIFs (optional)

**Module:** `koopmanrl_utils.movies.generate_gifs`

```bash
python -m koopmanrl_utils.movies.generate_gifs [flags]
```

### What it does

Reads the same `.npy` output folder as Step 2a and produces two GIFs per
trajectory:

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
python -m koopmanrl_utils.movies.generate_gifs \
    --data_folder video_frames/Lorenz-v0_1744000000

# With uncontrolled overlay, finer frames, and .dat export
python -m koopmanrl_utils.movies.generate_gifs \
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
| `algo_policies.py` | Concrete policy wrappers: `LQR`, `SKVI`, `SAKC`. Each wraps the corresponding trained model and exposes `get_action`. |
| `default_policies.py` | `ZeroPolicy` (always returns 0) and `RandomPolicy` (samples from the action space). Used as the uncontrolled and baseline references. |
| `env_enum.py` | `EnvEnum` string enum mapping human-readable names to gym IDs. |
| `generator.py` | `Generator` class that owns the environment, policy, and RNG state, and implements `generate_trajectories(num_trajectories, num_steps_per_trajectory)`. |

---

## Complete worked example (SAKC on FluidFlow)

```bash
# 1. Generate trajectories (seed and num-paths auto-loaded from config)
python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 1732368170 \
    --chkpt_step 50000 \
    --emit_dat

# Note the output folder name printed to stdout, e.g.:
#   Saved .npy files to 'video_frames/FluidFlow-v0_1744123456'

# 2a. Static figure with vector field and coordinate frame
python -m koopmanrl_utils.movies.generate_trajectory_figure \
    --data_folder video_frames/FluidFlow-v0_1744123456 \
    --plot_uncontrolled \
    --plot_vector_field \
    --emit_dat \
    --output_file figures/fluid_flow.png

# 2b. Animated GIF (optional)
python -m koopmanrl_utils.movies.generate_gifs \
    --data_folder video_frames/FluidFlow-v0_1744123456 \
    --plot_uncontrolled \
    --save_every_n_steps 10
```
