"""
Generate a static publication-quality 3D trajectory figure from saved rollout data.

Reads the .npy files produced by generate_trajectories.py and renders a single
high-resolution PNG.  Optionally overlays the uncontrolled (zero-policy) trajectory
and the uncontrolled vector field as a quiver plot.

Example usage:

    python -m koopmanrl_utils.movies.generate_trajectory_figure \\
        --data_folder video_frames/FluidFlow-v0_1744000000 \\
        --plot_uncontrolled \\
        --plot_vector_field \\
        --vector_field_resolution 8 \\
        --show_coordinate_frame \\
        --view_elev 25 \\
        --view_azim 60 \\
        --emit_dat \\
        --output_file figures/fluid_flow_trajectory.png
"""

import os
import warnings
from typing import Optional

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tap import Tap

from koopmanrl.environments import DoubleWell, FluidFlow, Lorenz  # noqa: F401 — registers gym envs
from koopmanrl_utils.movies.env_enum import EnvEnum

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Environment slug / label helpers
# ---------------------------------------------------------------------------

ENV_SLUG: dict[str, str] = {
    "FluidFlow-v0": "fluid_flow",
    "Lorenz-v0": "lorenz",
    "DoubleWell-v0": "double_well",
}

SUPPORTED_ENVS = set(ENV_SLUG.keys())


def _axis_labels(env_id: str) -> tuple[str, str, str]:
    """Return (xlabel, ylabel, zlabel) for the 3D plot."""
    if env_id == EnvEnum.DoubleWell:
        return r"$x_1$", r"$x_2$", r"$V(x_1, x_2)$"
    return r"$x_1$", r"$x_2$", r"$x_3$"


# ---------------------------------------------------------------------------
# Argument class
# ---------------------------------------------------------------------------


class Args(Tap):
    data_folder: str
    """Folder containing .npy trajectory data (output of generate_trajectories.py)."""

    seed: int = 123
    """RNG seed (used when constructing the environment for axis limits / vector field)."""

    trajectory_idx: int = 0
    """Which trajectory to plot (0-indexed)."""

    plot_uncontrolled: bool = False
    """Overlay the zero-policy (uncontrolled) trajectory."""

    plot_vector_field: bool = False
    """Overlay the uncontrolled vector field as a quiver plot (skipped for DoubleWell)."""

    vector_field_resolution: int = 8
    """Grid points per axis for the quiver plot."""

    step_limit: Optional[int] = None
    """Plot only the first N steps. None means the full trajectory."""

    show_coordinate_frame: bool = True
    """Show axis labels, ticks, and pane edges. False reproduces the notebook axis-off style."""

    dpi: int = 300
    """Figure resolution in dots per inch."""

    output_file: Optional[str] = None
    """Output PNG path. Defaults to <data_folder>/trajectory_figure.png."""

    emit_dat: bool = False
    """Write the plotted trajectory points as tab-separated .dat files alongside the PNG."""

    view_elev: float = 20.0
    """3D view elevation angle in degrees."""

    view_azim: float = 45.0
    """3D view azimuth angle in degrees."""


# ---------------------------------------------------------------------------
# .dat export helper
# ---------------------------------------------------------------------------


def _write_dat(path: str, data: np.ndarray, col_names: list[str]) -> None:
    """
    Write *data* (shape: steps × cols) to a tab-separated .dat file with a
    #-prefixed header.  PGFPlots ignores comment lines by default, so the header
    is available for human reference without affecting \\addplot table.
    """
    header = "# " + "\t".join(col_names)
    np.savetxt(path, data, fmt="%.6g", delimiter="\t", header=header, comments="")


def _trajectory_dat_cols(env_id: str, state_dim: int) -> list[str]:
    if env_id == EnvEnum.DoubleWell:
        return ["step", "x0", "x1", "potential"]
    return ["step"] + [f"x{i}" for i in range(state_dim)]


# ---------------------------------------------------------------------------
# Coordinate frame rendering
# ---------------------------------------------------------------------------


def _apply_coordinate_frame(ax, env_id: str, view_elev: float, view_azim: float) -> None:
    """Configure transparent pane faces with visible edges and LaTeX axis labels."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")

    xlabel, ylabel, zlabel = _axis_labels(env_id)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=14, labelpad=10)

    ax.grid(True)
    ax.view_init(elev=view_elev, azim=view_azim)


# ---------------------------------------------------------------------------
# Vector field computation (non-DoubleWell only)
# ---------------------------------------------------------------------------


def _compute_vector_field(env, trajectories_main: np.ndarray, trajectories_zero: np.ndarray, resolution: int):
    """
    Evaluate env.f(x, 0) on a resolution³ grid that covers the bounding box of
    both trajectory sets with a 20 % margin.  Returns (X, Y, Z, dX_n, dY_n, dZ_n)
    where the delta vectors are already L2-normalised.
    """
    # Bounding box with 20 % margin
    margin = 0.20
    all_data = np.concatenate([trajectories_main, trajectories_zero], axis=0)

    def padded(arr, axis):
        lo, hi = arr[:, :, axis].min(), arr[:, :, axis].max()
        span = max(hi - lo, 1e-6)
        return lo - margin * span, hi + margin * span

    x_lo, x_hi = padded(all_data, 0)
    y_lo, y_hi = padded(all_data, 1)
    z_lo, z_hi = padded(all_data, 2)

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    zs = np.linspace(z_lo, z_hi, resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs)

    grid_pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    zero_action = np.zeros((1, env.action_dim))

    deltas = []
    for pt in grid_pts:
        col = pt.reshape(-1, 1)
        nxt = env.f(col, zero_action)
        deltas.append((nxt - col).flatten())

    deltas = np.array(deltas)          # (N, 3)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True).clip(min=1e-10)
    deltas_norm = deltas / norms

    dX = deltas_norm[:, 0].reshape(X.shape)
    dY = deltas_norm[:, 1].reshape(Y.shape)
    dZ = deltas_norm[:, 2].reshape(Z.shape)

    return X, Y, Z, dX, dY, dZ


# ---------------------------------------------------------------------------
# Axis limit helpers
# ---------------------------------------------------------------------------


def _axis_limits(main: np.ndarray, zero: Optional[np.ndarray], pad: float = 0.05):
    """Return (lo, hi) tuples for each spatial axis."""
    datasets = [main] if zero is None else [main, zero]

    def span(axis):
        lo = min(d[:, axis].min() for d in datasets)
        hi = max(d[:, axis].max() for d in datasets)
        margin = max((hi - lo) * pad, 1e-6)
        return lo - margin, hi + margin

    return span(0), span(1), span(2)


# ---------------------------------------------------------------------------
# Environment factory (minimal — no video recording)
# ---------------------------------------------------------------------------


def _make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = Args().parse_args()

    # -----------------------------------------------------------------------
    # Load trajectory data
    # -----------------------------------------------------------------------
    def _load(name: str) -> np.ndarray:
        return np.load(os.path.join(args.data_folder, name))

    main_traj = _load("main_policy_trajectories.npy")
    zero_traj = _load("zero_policy_trajectories.npy")
    metadata = np.load(os.path.join(args.data_folder, "metadata.npy"), allow_pickle=True).item()
    env_id = metadata["env_id"]

    if env_id not in SUPPORTED_ENVS:
        raise ValueError(
            f"env_id '{env_id}' is not a supported plotting target. "
            f"Choose from: {sorted(SUPPORTED_ENVS)}"
        )

    # -----------------------------------------------------------------------
    # Construct environment (needed for reference_point and f())
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    envs = gym.vector.SyncVectorEnv([_make_env(env_id, args.seed)])
    env = envs.envs[0]

    # -----------------------------------------------------------------------
    # Select trajectory slice
    # -----------------------------------------------------------------------
    idx = args.trajectory_idx
    step_end = args.step_limit  # None → full trajectory

    main_xy = main_traj[idx, :step_end]  # (steps, state_dim)
    zero_xy = zero_traj[idx, :step_end]

    state_dim = main_xy.shape[1]

    # -----------------------------------------------------------------------
    # Build figure
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(21, 14), dpi=args.dpi, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # -----------------------------------------------------------------------
    # Axis limits
    # -----------------------------------------------------------------------
    zero_for_limits = zero_xy if args.plot_uncontrolled else None
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = _axis_limits(main_xy, zero_for_limits)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)

    # -----------------------------------------------------------------------
    # Vector field (non-DoubleWell only)
    # -----------------------------------------------------------------------
    if args.plot_vector_field:
        if env_id == EnvEnum.DoubleWell:
            warnings.warn(
                "Vector field is not plotted for DoubleWell because its dynamics are stochastic.",
                stacklevel=2,
            )
        else:
            X, Y, Z, dX, dY, dZ = _compute_vector_field(
                env, main_traj[[idx]], zero_traj[[idx]], args.vector_field_resolution
            )
            ax.quiver(X, Y, Z, dX, dY, dZ, length=0.5, normalize=True, color="gray", alpha=0.35)

    # -----------------------------------------------------------------------
    # Plot trajectories
    # -----------------------------------------------------------------------
    x_m, y_m, z_m = main_xy[:, 0], main_xy[:, 1], main_xy[:, 2]
    ax.plot3D(x_m, y_m, z_m, color="tab:orange", linewidth=3, label=metadata["main_policy_name"])

    if args.plot_uncontrolled:
        x_z, y_z, z_z = zero_xy[:, 0], zero_xy[:, 1], zero_xy[:, 2]
        ax.plot3D(x_z, y_z, z_z, color="tab:blue", linewidth=2, label=metadata["zero_policy_name"])

    # -----------------------------------------------------------------------
    # Reference point and initial condition markers
    # -----------------------------------------------------------------------
    ref = env.reference_point
    ref_z = ref[2] if env_id != EnvEnum.DoubleWell else 0.0
    ax.scatter3D(ref[0], ref[1], ref_z, color="black", s=100, zorder=5, label="Target")

    ax.scatter3D(
        main_xy[0, 0], main_xy[0, 1], main_xy[0, 2],
        color="gray", s=80, zorder=5, label=r"$x_0$"
    )

    # -----------------------------------------------------------------------
    # Coordinate frame or axis-off
    # -----------------------------------------------------------------------
    if args.show_coordinate_frame:
        _apply_coordinate_frame(ax, env_id, args.view_elev, args.view_azim)
    else:
        ax.set_axis_off()
        ax.view_init(elev=args.view_elev, azim=args.view_azim)

    ax.legend(fontsize=12, loc="upper left")

    # -----------------------------------------------------------------------
    # Save PNG
    # -----------------------------------------------------------------------
    output_file = args.output_file or os.path.join(args.data_folder, "trajectory_figure.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, dpi=args.dpi)
    print(f"Saved figure to '{output_file}'")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # .dat export
    # -----------------------------------------------------------------------
    if args.emit_dat:
        col_names = _trajectory_dat_cols(env_id, state_dim)
        steps = np.arange(len(main_xy)).reshape(-1, 1)

        _write_dat(
            os.path.join(args.data_folder, "main_trajectory_plot.dat"),
            np.hstack([steps, main_xy]),
            col_names,
        )
        print(f"Saved main_trajectory_plot.dat to '{args.data_folder}'")

        if args.plot_uncontrolled:
            _write_dat(
                os.path.join(args.data_folder, "zero_trajectory_plot.dat"),
                np.hstack([steps, zero_xy]),
                col_names,
            )
            print(f"Saved zero_trajectory_plot.dat to '{args.data_folder}'")

        if args.plot_vector_field and env_id != EnvEnum.DoubleWell:
            vf_data = np.stack(
                [X.flatten(), Y.flatten(), Z.flatten(), dX.flatten(), dY.flatten(), dZ.flatten()],
                axis=1,
            )
            _write_dat(
                os.path.join(args.data_folder, "vector_field.dat"),
                vf_data,
                ["X", "Y", "Z", "dX", "dY", "dZ"],
            )
            print(f"Saved vector_field.dat to '{args.data_folder}'")


if __name__ == "__main__":
    main()
