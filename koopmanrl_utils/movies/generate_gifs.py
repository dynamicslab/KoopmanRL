"""
Animate saved trajectory data as GIFs.

Reads the .npy files produced by generate_trajectories.py and writes one
trajectory GIF and one cost-ratio GIF per trajectory.

Example usage:

    python -m koopmanrl_utils.movies.generate_gifs \\
        --data_folder video_frames/Lorenz-v0_1744000000 \\
        --save_every_n_steps 10 \\
        --plot_uncontrolled \\
        --emit_dat
"""

import os
from typing import Optional

import gym
import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tap import Tap

from koopmanrl.environments import (  # noqa: F401 — registers gym envs
    DoubleWell,
    FluidFlow,
    Lorenz,
)
from koopmanrl_utils.movies.env_enum import EnvEnum

# ---------------------------------------------------------------------------
# Argument class
# ---------------------------------------------------------------------------


class Args(Tap):
    seed: int = 123
    """Seed of the experiment."""

    data_folder: str = ""
    """Folder containing trajectory data (output of generate_trajectories.py)."""

    save_every_n_steps: int = 100
    """Save a frame every n steps."""

    plot_uncontrolled: bool = False
    """Overlay the zero-policy (uncontrolled) trajectory."""

    ma_window_size: Optional[int] = None
    """Moving average window size for the cost plot. Defaults to env-specific value."""

    emit_dat: bool = False
    """Write per-step cost data as a .dat file alongside each GIF."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_MA_WINDOWS: dict[str, int] = {
    "FluidFlow-v0": 200,
    "Lorenz-v0": 200,
    "DoubleWell-v0": 200,
}


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def moving_average(a: np.ndarray, n: int, keep_first: bool) -> np.ndarray:
    """Compute a causal moving average of window size *n*."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    moving_avg = ret[n - 1 :] / n
    if keep_first:
        return np.concatenate((a[: n - 1], moving_avg))
    return moving_avg


def _write_dat(path: str, data: np.ndarray, col_names: list[str]) -> None:
    header = "# " + "\t".join(col_names)
    np.savetxt(path, data, fmt="%.6g", delimiter="\t", header=header, comments="")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = Args().parse_args()

    # -----------------------------------------------------------------------
    # Load trajectory and cost data
    # -----------------------------------------------------------------------
    def _load(name: str) -> np.ndarray:
        return np.load(os.path.join(args.data_folder, name))

    main_policy_trajectories = _load("main_policy_trajectories.npy")
    main_policy_costs = _load("main_policy_costs.npy")
    baseline_policy_costs = _load("baseline_policy_costs.npy")

    if args.plot_uncontrolled:
        zero_trajectories = _load("zero_policy_trajectories.npy")

    metadata = np.load(os.path.join(args.data_folder, "metadata.npy"), allow_pickle=True).item()
    env_id: str = metadata["env_id"]
    is_double_well = env_id == EnvEnum.DoubleWell

    # -----------------------------------------------------------------------
    # Seed and environment
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(env_id, args.seed)])

    # -----------------------------------------------------------------------
    # Moving average window
    # -----------------------------------------------------------------------
    ma_window = args.ma_window_size if args.ma_window_size is not None else DEFAULT_MA_WINDOWS.get(env_id, 200)

    # -----------------------------------------------------------------------
    # Trajectory GIF loop
    # -----------------------------------------------------------------------
    for trajectory_num in range(main_policy_trajectories.shape[0]):
        trajectory_fig = plt.figure(figsize=(21, 14), dpi=300, constrained_layout=True)
        trajectory_ax = trajectory_fig.add_subplot(111, projection="3d")
        trajectory_frames = []

        full_x = main_policy_trajectories[trajectory_num, :, 0]
        full_y = main_policy_trajectories[trajectory_num, :, 1]

        if args.plot_uncontrolled:
            full_x_zero = zero_trajectories[trajectory_num, :, 0]
            full_y_zero = zero_trajectories[trajectory_num, :, 1]

        if is_double_well:
            full_z = main_policy_trajectories[trajectory_num, :, 2]
            if args.plot_uncontrolled:
                full_z_zero = zero_trajectories[trajectory_num, :, 2]

            step_size = 0.1
            X_mesh, Y_mesh = np.meshgrid(
                np.arange(
                    start=envs.envs[0].state_minimums[0],
                    stop=envs.envs[0].state_maximums[0] + step_size,
                    step=step_size,
                ),
                np.arange(
                    start=envs.envs[0].state_minimums[1],
                    stop=envs.envs[0].state_maximums[1] + step_size,
                    step=step_size,
                ),
            )
        else:
            full_z = main_policy_trajectories[trajectory_num, :, 2]
            if args.plot_uncontrolled:
                full_z_zero = zero_trajectories[trajectory_num, :, 2]

        # Compute axis limits once across the full trajectories
        if args.plot_uncontrolled:
            max_x = np.max([np.max(full_x), np.max(full_x_zero)])
            max_y = np.max([np.max(full_y), np.max(full_y_zero)])
            max_z = np.max([np.max(full_z), np.max(full_z_zero)])
            min_x = np.min([np.min(full_x), np.min(full_x_zero)])
            min_y = np.min([np.min(full_y), np.min(full_y_zero)])
            min_z = np.min([np.min(full_z), np.min(full_z_zero)])
        else:
            max_x, max_y, max_z = np.max(full_x), np.max(full_y), np.max(full_z)
            min_x, min_y, min_z = np.min(full_x), np.min(full_y), np.min(full_z)

        for step_num in range(main_policy_trajectories.shape[1]):
            if step_num == 0 or (step_num + 1) % args.save_every_n_steps == 0:
                x = full_x[: (step_num + 1)]
                y = full_y[: (step_num + 1)]
                z = full_z[: (step_num + 1)]

                if args.plot_uncontrolled:
                    x_zero = full_x_zero[: (step_num + 1)]
                    y_zero = full_y_zero[: (step_num + 1)]
                    z_zero = full_z_zero[: (step_num + 1)]

                trajectory_ax.clear()

                # Reference point marker
                trajectory_ax.scatter(
                    envs.envs[0].reference_point[0],
                    envs.envs[0].reference_point[1],
                    envs.envs[0].reference_point[2] if not is_double_well else 0.0,
                    color="green",
                    s=100,
                    zorder=1,
                )

                trajectory_ax.set_xlim(min_x, max_x)
                trajectory_ax.set_ylim(min_y, max_y)

                if is_double_well:
                    # Use the potential stored as dim 2 for the z-axis
                    Z_surface = envs.envs[0].potential(X_mesh, Y_mesh, u=0)
                    trajectory_ax.plot3D(x, y, z, alpha=1.0, linewidth=2, color="tab:orange")
                    trajectory_ax.plot_surface(X_mesh, Y_mesh, Z_surface, alpha=0.7, cmap=cm.coolwarm)
                    trajectory_ax.set_zlim(0, 15)
                else:
                    trajectory_ax.set_zlim(min_z, max_z)
                    if args.plot_uncontrolled:
                        trajectory_ax.plot3D(x_zero, y_zero, z_zero, color="tab:blue", zorder=2)
                    trajectory_ax.plot3D(x, y, z, linewidth=3, color="tab:orange", zorder=2)

                    trajectory_ax.grid(False)
                    trajectory_ax.set_axis_off()

                frame_path = os.path.join(args.data_folder, f"trajectory_frame_{step_num + 1}.png")
                plt.savefig(frame_path)
                plt.cla()
                trajectory_frames.append(imageio.imread(frame_path))

            if step_num == 0 or (step_num + 1) % 100 == 0:
                print(f"Created {step_num + 1} trajectory frame(s)")

        gif_path = os.path.join(args.data_folder, f"trajectory_{trajectory_num + 1}.gif")
        imageio.mimsave(gif_path, trajectory_frames, duration=0.1)
        print(f"Saved trajectory GIF: '{gif_path}'")

        plt.close(trajectory_fig)

    # -----------------------------------------------------------------------
    # Cost GIF loop (only when baseline data is available and non-zero)
    # -----------------------------------------------------------------------
    safe_to_render_costs = True
    if np.any(baseline_policy_costs == 0):
        print("Warning: baseline_policy_costs contains zeros; skipping cost GIF to avoid division by zero.")
        safe_to_render_costs = False

    if safe_to_render_costs:
        cost_fig = plt.figure(figsize=(21, 14), dpi=300)
        cost_ax = cost_fig.add_subplot(111)

        for cost_num in range(main_policy_costs.shape[0]):
            cost_frames = []

            all_main_costs = main_policy_costs[cost_num]
            all_baseline_costs = baseline_policy_costs[cost_num]
            log_all_cost_ratios = np.log(all_main_costs / all_baseline_costs)
            moving_avg = moving_average(log_all_cost_ratios, n=ma_window, keep_first=False)

            start_idx = ma_window - 1
            ma_x = np.arange(start_idx, moving_avg.shape[0] + start_idx)

            min_log = np.min(log_all_cost_ratios)
            max_log = np.max(log_all_cost_ratios)

            for step_num in range(main_policy_costs.shape[1]):
                if step_num == 0 or (step_num + 1) % args.save_every_n_steps == 0:
                    log_ratio_slice = log_all_cost_ratios[: (step_num + 1)]

                    if step_num >= start_idx:
                        ma_slice = moving_avg[: (step_num + 1) - start_idx]
                        ma_x_slice = ma_x[: (step_num + 1) - start_idx]
                    else:
                        ma_slice = np.array([])
                        ma_x_slice = np.array([])

                    cost_ax.clear()
                    cost_ax.set_xlim(0, main_policy_costs.shape[1])
                    cost_ax.set_ylim(min_log * 1.1, max_log * 1.1)
                    cost_ax.set_xlabel("Steps")
                    cost_ax.set_ylabel(
                        f"Cost Ratio ({metadata['main_policy_name']} / {metadata['baseline_policy_name']})"
                    )
                    cost_ax.set_title(
                        f"Cost Ratio: {metadata['main_policy_name']} / {metadata['baseline_policy_name']}"
                    )
                    cost_ax.title.set_fontsize(20)
                    cost_ax.axhline(y=0, color="r", linestyle="--")
                    cost_ax.grid(True)
                    cost_ax.plot(log_ratio_slice, alpha=0.5)
                    if len(ma_slice):
                        cost_ax.plot(ma_x_slice, ma_slice, linewidth=3)

                    frame_path = os.path.join(args.data_folder, f"cost_frame_{step_num + 1}.png")
                    plt.savefig(frame_path)
                    cost_frames.append(imageio.imread(frame_path))

                if step_num == 0 or (step_num + 1) % 100 == 0:
                    print(f"Created {step_num + 1} cost frame(s)")

            cost_gif_path = os.path.join(args.data_folder, f"costs_{cost_num + 1}.gif")
            imageio.mimsave(cost_gif_path, cost_frames, duration=0.1)
            print(f"Saved cost GIF: '{cost_gif_path}'")

            # .dat export — full cost ratio series for this trajectory
            if args.emit_dat:
                steps = np.arange(len(log_all_cost_ratios)).reshape(-1, 1)
                dat_data = np.hstack([steps, log_all_cost_ratios.reshape(-1, 1)])
                dat_path = os.path.join(args.data_folder, f"costs_{cost_num + 1}.dat")
                header = "# " + "\t".join(["step", "log_cost_ratio"])
                np.savetxt(dat_path, dat_data, fmt="%.6g", delimiter="\t", header=header, comments="")
                print(f"Saved cost .dat: '{dat_path}'")

        plt.close(cost_fig)

    print(f"All outputs saved to '{args.data_folder}'")


if __name__ == "__main__":
    main()
