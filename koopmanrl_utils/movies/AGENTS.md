# Movies Guide

The movies subdirectory contains the requisite logic to generate trajectory figures, movies, or GIFs of the applied control for illustration purposes. `PIPELINE.md` is the guide to the pipeline: commands, flags, checkpoints, outputs and limitations.

## Running Scripts

All scripts need to be run from the root of the repository (configurations and checkpoints are resolved relative to it), as modules:

1. `uv run -m koopmanrl_utils.movies.generate_trajectories` rolls out a policy (`zero`, `random`, `lqr`, `sac`, `skvi`, `sakc`) and stores the trajectories in `video_frames/<env>_<timestamp or --run_label>/` (gitignored).
2. `uv run -m koopmanrl_utils.movies.generate_trajectory_figure --data_folder <that folder>` renders a static figure, optionally with `.dat` files for TikZ.
3. `uv run -m koopmanrl_utils.movies.generate_gifs --data_folder <that folder>` renders a GIF.

Only `FluidFlow-v0`, `Lorenz-v0` and `DoubleWell-v0` are plotting targets. The trained policies are loaded from `saved_models/`; `PIPELINE.md` describes the checkpoint flags of each algorithm. There are no tests for this pipeline.

## Directory Structure

```
movies/
├── __init__.py                    # Initialization file
├── abstract_policy.py             # Abstract base class of the policies
├── AGENTS.md                      # This file
├── algo_policies.py               # Policies of SKVI, SAKC, SAC and LQR, loaded from their checkpoints
├── default_policies.py            # Zero and random policies
├── env_enum.py                    # Enumeration of the reinforcement learning environments
├── generate_gifs.py               # Generates GIF illustrations of the applied control policy
├── generate_trajectories.py       # Rolls out a policy and stores the trajectories (.npy, optionally .dat)
├── generate_trajectory_figure.py  # Static 3D trajectory figure from the stored trajectories
├── generator.py                   # Generates controlled or uncontrolled trajectories
└── PIPELINE.md                    # Guide to the pipeline
```

See the root `AGENTS.md` for setup, testing and the working checklist.
