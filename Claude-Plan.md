# Plan: Trajectory Visualization for KoopmanRL

## Goal

Build a production-quality, CLI-driven trajectory visualization pipeline that mirrors the
capabilities of the upstream `movies/plotting_trajectories.ipynb` notebook
(Pdbz199/koopman-rl), while adding:

1. Full CLI hyperparameter control, with automatic loading of best-found configurations
   from `configurations/<algo>_<env_slug>_hparams.json`
2. Optional `.dat` file export for TikZ ingestion
3. A visible coordinate frame on all Matplotlib 3D trajectory plots

**LinearSystem is excluded** as a plotting target throughout — it exists in the
configurations directory but is not a visualization goal for this pipeline.

---

## Current State Audit

### What exists and what is broken

| File | Status | Notes |
|---|---|---|
| `generator.py` | Working | `Generator` class correctly rolls out policies |
| `generate_trajectories.py` | Partially broken | Hardcodes `SAC(Q)` + specific checkpoint timestamps; uses `argparse` + `distutils.strtobool` instead of `tap` |
| `generate_gifs.py` | Mostly working | `data_folder: bool = True` type annotation is wrong (must be `str`); `ma_window_size` lacks `Optional` annotation |
| `generate_trajectory_figure.py` | Stub / broken | References wrong import paths; no real implementation |
| `algo_policies.py` | Import broken | `from movies import Policy` — must be `from koopmanrl_utils.movies.abstract_policy import Policy` |
| `default_policies.py` | Import broken | Same `from movies import Policy` issue |
| `env_enum.py` | Working | — |
| `abstract_policy.py` | Working | — |

### Data flow (current)

```
generate_trajectories.py
    → video_frames/<env_id>_<timestamp>/{main,baseline,zero}_policy_{trajectories,actions,costs}.npy
    → video_frames/<env_id>_<timestamp>/metadata.npy

generate_gifs.py --data-folder=<above>
    → video_frames/<env_id>_<timestamp>/trajectory_frame_N.png  (intermediate)
    → video_frames/<env_id>_<timestamp>/trajectory_N.gif
    → video_frames/<env_id>_<timestamp>/costs_N.gif
```

### What the upstream notebook does that is missing here

- Static single-figure plots of full trajectories (no GIF loop)
- Vector field quiver plot overlaid with trajectories
- Visible coordinate frame (notebook calls `set_axis_off()`; we want labels, ticks, pane edges)
- No `.dat` export

---

## Environments and Algorithms

### Plotting target environments

| Gym ID | Env slug | State dim | Action dim | Notes |
|---|---|---|---|---|
| `FluidFlow-v0` | `fluid_flow` | 3 | 1 | — |
| `Lorenz-v0` | `lorenz` | 3 | 1 | dt=0.01; max 2000 steps |
| `DoubleWell-v0` | `double_well` | 2 (+1 potential) | 1 | 4th trajectory dim is potential surface |

The env slug is the lowercase, underscore-separated form used in configuration filenames.
The mapping function used throughout the scripts:

```python
ENV_SLUG = {
    "FluidFlow-v0":  "fluid_flow",
    "Lorenz-v0":     "lorenz",
    "DoubleWell-v0": "double_well",
}
```

### Algorithms (selectable by `--algo`)

| Flag value | Class | Checkpoint needed | Config file key |
|---|---|---|---|
| `zero` | `ZeroPolicy` | No | — |
| `random` | `RandomPolicy` | No | — |
| `lqr` | `LQR` | No (computed from env) | — |
| `skvi` | `SKVI` | Yes: `--chkpt-timestamp`, `--chkpt-epoch` | `skvi_<env_slug>_hparams.json` |
| `sakc` | `SAKC(is_koopman=True, is_value_based=False)` | Yes: `--chkpt-timestamp`, `--chkpt-step` | `sakc_<env_slug>_hparams.json` |

---

## Configuration Files

### Location and naming

```
configurations/
├── sakc_double_well_hparams.json
├── sakc_fluid_flow_hparams.json
├── sakc_lorenz_hparams.json
├── skvi_double_well_hparams.json
├── skvi_fluid_flow_hparams.json
└── skvi_lorenz_hparams.json
```

### SAKC config structure (example: `sakc_fluid_flow_hparams.json`)

```json
{
    "env-id": "FluidFlow-v0",
    "seed": 6597,
    "v-lr": 0.009423359172870875,
    "q-lr": 0.0017865746944645956,
    "num-paths": 50,
    "num-steps-per-path": 175,
    "state-order": 3,
    "action-order": 3,
    "total-timesteps": 50000,
    "target-score": null,
    "num-envs": 1,
    "metric": "charts/episodic_return",
    "metric-last-n-average-window": 5
}
```

### SKVI config structure (example: `skvi_fluid_flow_hparams.json`)

```json
{
    "env-id": "FluidFlow-v0",
    "seed": 6517,
    "learning-rate": 0.00031904756404241047,
    "number-of-train-epochs": 125,
    "num-paths": 200,
    "num-steps-per-path": 225,
    "state-order": 4,
    "action-order": 2,
    "total-timesteps": 50000,
    "num-envs": 1,
    "metric": "charts/episodic_return",
    "metric-last-n-average-window": 5
}
```

### Fields consumed by the trajectory pipeline

The config files record training hyperparameters; only a subset is meaningful at
trajectory-generation time:

| Config key | Maps to CLI flag | Notes |
|---|---|---|
| `seed` | `--seed` | RNG seed for reproducibility |
| `num-paths` | `--num-trajectories` | Number of rollouts to generate |
| `num-steps-per-path` | `--num-steps` | Steps per rollout |

All other fields (`v-lr`, `q-lr`, `learning-rate`, `state-order`, `action-order`, etc.)
are training-time parameters; they are **ignored** by the trajectory scripts. Checkpoint
identity (`--chkpt-timestamp` and `--chkpt-step`/`--chkpt-epoch`) is still provided on
the CLI because it is not stored in the config files.

### Config auto-discovery

When `--algo` is `sakc` or `skvi` and `--env-id` is one of the three plotting targets,
the script resolves the config path automatically:

```python
def resolve_config_path(algo: str, env_id: str) -> str:
    slug = ENV_SLUG[env_id]
    return f"configurations/{algo}_{slug}_hparams.json"
```

A `--config-file` CLI flag allows overriding this path explicitly. Config values are
treated as **defaults** — any flag explicitly passed on the CLI takes precedence,
following standard layered-config precedence: CLI > config file > code defaults.

---

## Phases

---

### Phase 0 — Fix broken imports (prerequisite for everything)

**Files:** `algo_policies.py`, `default_policies.py`

Replace the two broken import lines:

```python
# Before
from movies import Policy

# After
from koopmanrl_utils.movies.abstract_policy import Policy
```

This is a blocking issue; nothing downstream can be imported without it.

---

### Phase 1 — Refactor `generate_trajectories.py`

**Goal:** full CLI via `tap`, config-file auto-discovery, policy factory, `.dat` export.

#### 1.1 Argument class

```python
from tap import Tap
from typing import Optional

class Args(Tap):
    env_id: str = "FluidFlow-v0"
    """Gym environment ID. One of: FluidFlow-v0, Lorenz-v0, DoubleWell-v0."""
    seed: Optional[int] = None
    """RNG seed. If omitted, loaded from config file; falls back to 123."""
    torch_deterministic: bool = True
    """Set torch.backends.cudnn.deterministic."""
    cuda: bool = True
    """Enable CUDA if available."""
    num_trajectories: Optional[int] = None
    """Number of trajectories. If omitted, loaded from config (num-paths)."""
    num_steps: Optional[int] = None
    """Steps per trajectory. If omitted, loaded from config (num-steps-per-path)."""

    # Algorithm selection
    algo: str = "sakc"
    """Main policy: zero | random | lqr | skvi | sakc."""
    baseline_algo: str = "zero"
    """Baseline policy: zero | random | lqr."""

    # RL evaluation hyperparameters (not in config files; always CLI)
    gamma: float = 0.99
    """Discount factor."""
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    num_actions: int = 101
    """Discrete action grid size for SKVI."""

    # Checkpoint arguments (required for skvi / sakc)
    chkpt_timestamp: Optional[int] = None
    """Unix timestamp of the training run checkpoint to load."""
    chkpt_step: Optional[int] = None
    """Step number for SAKC checkpoint."""
    chkpt_epoch: Optional[int] = None
    """Epoch number for SKVI checkpoint."""
    koopman_model_name: Optional[str] = None
    """Saved Koopman tensor model name (required for SKVI)."""

    # Config override
    config_file: Optional[str] = None
    """Explicit path to hparams JSON. Auto-resolved from algo + env-id if omitted."""

    # Output
    output_dir: str = "video_frames"
    """Root directory for output."""
    emit_dat: bool = False
    """Also write trajectory data as .dat files for TikZ."""
```

#### 1.2 Config loading

```python
import json

def load_config(args: Args) -> dict:
    path = args.config_file or resolve_config_path(args.algo, args.env_id)
    with open(path) as f:
        return json.load(f)

def apply_config(args: Args, cfg: dict) -> Args:
    """Fill in any Args field that was not set on the CLI."""
    if args.seed is None:
        args.seed = cfg.get("seed", 123)
    if args.num_trajectories is None:
        args.num_trajectories = cfg.get("num-paths", 1)
    if args.num_steps is None:
        args.num_steps = cfg.get("num-steps-per-path", None)
    return args
```

Config loading is skipped (with a warning) when `--algo` is `zero`, `random`, or `lqr`,
as no config file exists for those policies.

#### 1.3 Policy factory

```python
def build_policy(algo: str, args: Args, envs, device) -> Policy:
    if algo == "zero":
        return ZeroPolicy(is_2d=True, name="Zero Policy")
    elif algo == "random":
        return RandomPolicy(envs.envs[0], name="Random Policy")
    elif algo == "lqr":
        return LQR(args=args, envs=envs, name="LQR")
    elif algo == "skvi":
        assert args.chkpt_timestamp and args.chkpt_epoch and args.koopman_model_name, \
            "SKVI requires --chkpt-timestamp, --chkpt-epoch, --koopman-model-name"
        return SKVI(
            args=args, envs=envs,
            saved_koopman_model_name=args.koopman_model_name,
            trained_model_start_timestamp=args.chkpt_timestamp,
            chkpt_epoch_number=args.chkpt_epoch,
            device=device, name="SKVI",
        )
    elif algo == "sakc":
        assert args.chkpt_timestamp and args.chkpt_step, \
            "SAKC requires --chkpt-timestamp and --chkpt-step"
        return SAKC(
            args=args, envs=envs,
            is_value_based=False,
            is_koopman=True,
            chkpt_timestamp=args.chkpt_timestamp,
            chkpt_step_number=args.chkpt_step,
            device=device, name="SAKC",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
```

#### 1.4 `.dat` export

After saving `.npy` files, when `args.emit_dat` is set, write three `.dat` files per
policy alongside the `.npy` files. Format: tab-separated, `#`-prefixed header row
(PGFPlots ignores `#` lines with the default `comment chars={#}` setting).

Columns:
- `trajectories.dat`: `step  x0  x1  x2  [x3 ...]`
- `actions.dat`: `step  a0  [a1 ...]`
- `costs.dat`: `step  cost`

Values written with `%.6g` format. One row per time step.

---

### Phase 2 — Implement `generate_trajectory_figure.py`

**Goal:** produce a high-quality static PNG of the full trajectory (or a prefix of it),
with optional uncontrolled overlay, optional vector field quiver, and a visible coordinate
frame. The script equivalent of the interactive notebook.

#### 2.1 Argument class

```python
class Args(Tap):
    data_folder: str
    """Folder containing .npy trajectory data (output of generate_trajectories.py)."""
    seed: int = 123
    trajectory_idx: int = 0
    """Which trajectory to plot (0-indexed)."""
    plot_uncontrolled: bool = False
    """Overlay the zero-policy (uncontrolled) trajectory."""
    plot_vector_field: bool = False
    """Overlay the uncontrolled vector field as a quiver plot."""
    vector_field_resolution: int = 8
    """Grid points per axis for the quiver plot."""
    step_limit: Optional[int] = None
    """Plot only the first N steps (None = full trajectory)."""
    show_coordinate_frame: bool = True
    """Show axis labels, ticks, and pane edges (False reproduces the notebook's axis-off style)."""
    dpi: int = 300
    output_file: Optional[str] = None
    """Output PNG. Defaults to <data_folder>/trajectory_figure.png."""
    emit_dat: bool = False
    """Write the plotted trajectory points as .dat files alongside the PNG."""
    view_elev: float = 20.0
    """3D view elevation angle."""
    view_azim: float = 45.0
    """3D view azimuth angle."""
```

#### 2.2 Coordinate frame (Phase 5 detail)

When `show_coordinate_frame=True` (default), the 3D axis must:

```python
# Transparent pane fills, visible edges — clean coordinate-frame appearance
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("gray")
ax.yaxis.pane.set_edgecolor("gray")
ax.zaxis.pane.set_edgecolor("gray")

# LaTeX axis labels
ax.set_xlabel(r"$x_1$", fontsize=14, labelpad=10)
ax.set_ylabel(r"$x_2$", fontsize=14, labelpad=10)
ax.set_zlabel(r"$x_3$", fontsize=14, labelpad=10)

# Background grid on panes
ax.grid(True)

# View angle
ax.view_init(elev=args.view_elev, azim=args.view_azim)
```

For DoubleWell, z is the potential surface: label becomes `r"$V(x_1, x_2)$"`.

When `show_coordinate_frame=False`, call `ax.set_axis_off()` to match the notebook style.

#### 2.3 Trajectory plot logic

Draw the full trajectory (or up to `step_limit` steps), no animation:

- **Main policy** (controlled): `tab:orange`, linewidth 3
- **Zero policy** (uncontrolled, if `plot_uncontrolled`): `tab:blue`, linewidth 2
- **Reference point**: black scatter, 100 pt², labeled "Target" in the legend
- **Initial condition**: gray scatter, 80 pt², labeled `r"$x_0$"` in the legend

Axis limits: `[min − 5% pad, max + 5% pad]` across both trajectories (or just the main
trajectory when `plot_uncontrolled=False`).

#### 2.4 Vector field overlay

When `plot_vector_field=True`, evaluate the uncontrolled dynamics `env.f(x, zero_action)`
on a `resolution × resolution × resolution` grid spanning the trajectory bounding box
with a 20% margin. Normalize each delta vector. Quiver arguments:

```python
ax.quiver(X, Y, Z, dX, dY, dZ,
          length=0.5, normalize=True,
          color="gray", alpha=0.35)
```

This reproduces the `plot_vector_field()` function from the upstream notebook as a
first-class feature.

#### 2.5 `.dat` export

When `emit_dat=True`, write alongside the PNG:

- `main_trajectory_plot.dat` — trajectory points actually plotted (respects `step_limit`
  and `trajectory_idx`); columns: `step  x0  x1  x2`
- `zero_trajectory_plot.dat` — if `plot_uncontrolled`; same columns
- `vector_field.dat` — if `plot_vector_field`; columns: `X  Y  Z  dX  dY  dZ`
  (normalized, for `\addplot3` in TikZ)

---

### Phase 3 — Fix `generate_gifs.py`

Targeted bug fixes only; GIF generation logic is otherwise unchanged:

1. `data_folder: bool = True` → `data_folder: str`
2. `ma_window_size: int` → `ma_window_size: Optional[int] = None`, falling back to
   `default_ma_windows[env_id]` when `None`
3. Add `emit_dat: bool = False` — writes per-step cost data as `costs_plot.dat` alongside
   each GIF (same tab-separated format as Phase 1)
4. Guard the cost GIF section: skip it if baseline policy data is absent rather than
   crashing on a division-by-zero

---

### Phase 4 — `.dat` format specification (TikZ compatibility)

All `.dat` files produced by any script must be readable by PGFPlots `\addplot table`:

```
# step	x0	x1	x2
0	1.234567e+00	-4.500000e-01	2.110000e+00
1	...
```

Rules:
- Tab-separated
- First line is a `#` comment (header) — PGFPlots ignores it by default
- Values formatted as `%.6g`
- One row per time step

Example PGFPlots snippet:

```latex
\addplot3 table [x=x0, y=x1, z=x2] {main_trajectory_plot.dat};
\addplot3 table [x=x0, y=x1, z=x2] {zero_trajectory_plot.dat};
```

---

## File Inventory After Implementation

```
configurations/
├── sakc_fluid_flow_hparams.json    # read-only; consumed by generate_trajectories.py
├── sakc_lorenz_hparams.json
├── sakc_double_well_hparams.json
├── skvi_fluid_flow_hparams.json
├── skvi_lorenz_hparams.json
└── skvi_double_well_hparams.json

koopmanrl_utils/movies/
├── abstract_policy.py                # unchanged
├── algo_policies.py                  # Phase 0: fix imports
├── default_policies.py               # Phase 0: fix imports
├── env_enum.py                       # unchanged
├── generator.py                      # unchanged
├── generate_trajectories.py          # Phase 1: full rewrite
├── generate_trajectory_figure.py     # Phase 2: full implementation
└── generate_gifs.py                  # Phase 3: targeted fixes
```

---

## Example Invocations (post-implementation)

### Note on CLI flag style

`tap` exposes argument names using **underscores** (e.g. `--env_id`, not `--env-id`).
This matches Python attribute naming and is consistent throughout all three scripts.

### SAKC on FluidFlow — config auto-discovered, seed and num-trajectories from config

```bash
uv run python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id FluidFlow-v0 \
    --algo sakc \
    --chkpt_timestamp 1732368170 \
    --chkpt_step 50000 \
    --emit_dat
# Loads configurations/sakc_fluid_flow_hparams.json automatically.
# seed=6597, num_trajectories=50, num_steps=175 come from the config.
```

### SKVI on Lorenz — CLI seed overrides config

```bash
uv run python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id Lorenz-v0 \
    --algo skvi \
    --chkpt_timestamp 1738511670 \
    --chkpt_epoch 125 \
    --koopman_model_name <model_name> \
    --seed 42 \
    --emit_dat
# Loads configurations/skvi_lorenz_hparams.json.
# seed=42 (CLI) overrides seed=8953 (config).
# num_trajectories=150, num_steps=250 come from config.
```

### LQR baseline on DoubleWell — no config file, all defaults

```bash
uv run python -m koopmanrl_utils.movies.generate_trajectories \
    --env_id DoubleWell-v0 \
    --algo lqr \
    --baseline_algo zero \
    --num_trajectories 1
```

### Static figure with coordinate frame, uncontrolled overlay, and vector field

```bash
uv run python -m koopmanrl_utils.movies.generate_trajectory_figure \
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

### Animated GIF from existing trajectory data

```bash
uv run python -m koopmanrl_utils.movies.generate_gifs \
    --data_folder video_frames/Lorenz-v0_1744000000 \
    --save_every_n_steps 10 \
    --plot_uncontrolled \
    --emit_dat
```

---

## Open Questions

1. **DoubleWell state dimension**: `Generator` appends the potential as a 4th state
   component. `generate_trajectory_figure.py` must handle this by treating x1/x2 as the
   horizontal plane and the potential as the z-axis — verify this matches the existing
   `generate_gifs.py` behavior before implementing.

2. **Multiple trajectories in the figure script**: the current design plots a single
   `trajectory_idx`. A future `--overlay-all-trajectories` flag could draw all N with alpha
   blending — out of scope for now.

3. **Checkpoint identity not stored in config**: checkpoint timestamps and step/epoch numbers
   are not in the JSON files. They remain required CLI flags for `sakc` and `skvi`. If a
   registry of trained checkpoints is added later, the config loading logic in `apply_config`
   is the natural extension point.
