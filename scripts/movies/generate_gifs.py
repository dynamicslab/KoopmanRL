"""
Example usage:
python -m movies.generate_gifs --save-every-n-steps=100 --plot-uncontrolled=True --data-folder=./video_frames/FluidFlow-v0_1723125311
"""

import argparse
import gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import torch

from custom_envs import *
from movies.env_enum import EnvEnum

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123,
    help="seed of the experiment (default: 123)")
parser.add_argument("--data-folder", type=str, required=True,
    help="Folder containing trajectory data")
parser.add_argument("--save-every-n-steps", type=int, default=None,
    help="Save a frame every n steps")
parser.add_argument("--plot-uncontrolled", type=bool, default=False,
    help="Plot the uncontrolled system dynamics")
parser.add_argument("--ma-window-size", type=int, default=None,
    help="Moving average window size")
args = parser.parse_args()

# Helper function to create environments
def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Load main policy data
main_policy_trajectories = np.load(f"{args.data_folder}/main_policy_trajectories.npy")
main_policy_costs = np.load(f"{args.data_folder}/main_policy_costs.npy")
# Load baseline policy data
baseline_trajectories = np.load(f"{args.data_folder}/baseline_policy_trajectories.npy")
baseline_policy_costs = np.load(f"{args.data_folder}/baseline_policy_costs.npy")
#  Load zero policy data
if args.plot_uncontrolled:
    zero_trajectories = np.load(f"{args.data_folder}/zero_policy_trajectories.npy")
    zero_costs = np.load(f"{args.data_folder}/zero_policy_costs.npy")
# Load metadata
metadata = np.load(f"{args.data_folder}/metadata.npy", allow_pickle=True).item()
# Extract env_id
env_id = metadata['env_id']

# Function to reset seeds
def reset_seed():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# Set seeds
reset_seed()

# Create gym env with ID
envs = gym.vector.SyncVectorEnv([make_env(metadata['env_id'], args.seed)])

# Function to compute moving average, preserving the first n values
def moving_average(a, n, keep_first):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    moving_avg = ret[n - 1:] / n

    # If `keep_first` is True, concatenate the first n-1 values of the original array
    if keep_first:
        result = np.concatenate((a[:n - 1], moving_avg))
        print(result.shape)
    else:
        result = moving_avg

    return result

# Dictionary of default moving average window sizes for each environment
default_ma_windows = {
    "LinearSystem-v0": 20,
    "FluidFlow-v0": 200,
    "Lorenz-v0": 200,
    "DoubleWell-v0": 200,
}
ma_window_size = (
    default_ma_windows[metadata['env_id']]
    if args.ma_window_size is None else args.ma_window_size
)

# Plot trajectories
# for trajectory_num in range(main_policy_trajectories.shape[0]):
for trajectory_num in range(main_policy_trajectories.shape[0]):
    # Create trajectory figure
    trajectory_fig = plt.figure(figsize=(21, 14), dpi=300, constrained_layout=True)
    trajectory_ax = trajectory_fig.add_subplot(111, projection='3d')

    # Create array of frames
    trajectory_frames = []

    # Extract main_ policy and zero_policy trajectory data
    full_x = main_policy_trajectories[trajectory_num, :, 0]
    full_y = main_policy_trajectories[trajectory_num, :, 1]
    if args.plot_uncontrolled:
        full_x_zero = zero_trajectories[trajectory_num, :, 0]
        full_y_zero = zero_trajectories[trajectory_num, :, 1]
    if env_id == EnvEnum.DoubleWell:
        full_u = main_policy_trajectories[trajectory_num, :, 2]
        full_z = main_policy_trajectories[trajectory_num, :, 3]
        if args.plot_uncontrolled:
            full_u_zero = zero_trajectories[trajectory_num, :, 2]
            full_z_zero = zero_trajectories[trajectory_num, :, 3]
    else:
        full_z = main_policy_trajectories[trajectory_num, :, 2]
        if args.plot_uncontrolled:
            full_z_zero = zero_trajectories[trajectory_num, :, 2]

    if env_id == EnvEnum.DoubleWell:
        step_size = 0.1
        X, Y = np.meshgrid(
            np.arange(
                start=envs.envs[0].state_minimums[0],
                stop=envs.envs[0].state_maximums[0]+step_size,
                step=step_size
            ),
            np.arange(
                start=envs.envs[0].state_minimums[1],
                stop=envs.envs[0].state_maximums[1]+step_size,
                step=step_size
            )
        )

    for step_num in range(main_policy_trajectories.shape[1]):
        if step_num == 0 or (step_num+1) % args.save_every_n_steps == 0:  # Only save every n steps
            # Extract main_policy and zero_policy trajectory data
            x = full_x[:(step_num+1)]
            y = full_y[:(step_num+1)]
            z = full_z[:(step_num+1)]
            if env_id == EnvEnum.DoubleWell:
                u = full_u[:(step_num+1)]
                Z = envs.envs[0].potential(X, Y, u[step_num])
                Z_path = envs.envs[0].potential(x, y, u[step_num])

            if args.plot_uncontrolled:
                # Extract zero_policy trajectory data
                x_zero = full_x_zero[:(step_num+1)]
                y_zero = full_y_zero[:(step_num+1)]
                z_zero = full_z_zero[:(step_num+1)]
                if env_id == EnvEnum.DoubleWell:
                    u_zero = full_u_zero[:(step_num+1)]
                    # Z_zero = envs.envs[0].potential(X_zero, Y_zero, u_zero[step_num])
                    # Z_path_zero = envs.envs[0].potential(x_zero, y_zero, u_zero[step_num])

            # Clear the previous plot
            trajectory_ax.clear()

            # Place Green dot at the reference point as determined from the environment with zorder smaller than all other objects
            trajectory_ax.scatter(
                envs.envs[0].reference_point[0],
                envs.envs[0].reference_point[1],
                envs.envs[0].reference_point[2],
                color='green',
                s=100,
                zorder=1
            )

            # Reapply the axis settings
            trajectory_ax.set_xticklabels([])
            trajectory_ax.set_yticklabels([])
            trajectory_ax.set_yticklabels([])

            # Set axis limits according to the trajectory limits, using zero_policy only if plot_uncontrolled is True
            if args.plot_uncontrolled:
                # Maximums
                max_x = np.max([np.max(full_x), np.max(full_x_zero)])
                max_y = np.max([np.max(full_y), np.max(full_y_zero)])
                max_z = np.max([np.max(full_z), np.max(full_z_zero)])
                # Minimums
                min_x = np.min([np.min(full_x), np.min(full_x_zero)])
                min_y = np.min([np.min(full_y), np.min(full_y_zero)])
                min_z = np.min([np.min(full_z), np.min(full_z_zero)])
            else:
                # Maximums
                max_x = np.max(full_x)
                max_y = np.max(full_y)
                max_z = np.max(full_z)
                # Minimums
                min_x = np.min(full_x)
                min_y = np.min(full_y)
                min_z = np.min(full_z)

            trajectory_ax.set_xlim(min_x, max_x)
            trajectory_ax.set_ylim(min_y, max_y)
            if not env_id == EnvEnum.DoubleWell:
                trajectory_ax.set_zlim(min_z, max_z)

            # Plot trajectory
            if env_id == EnvEnum.DoubleWell:
                trajectory_ax.plot3D(x, y, Z_path, alpha=1.0, linewidth=2, color='black', pad=0.1)
                trajectory_ax.plot_surface(X, Y, Z, alpha=0.7, cmap=cm.coolwarm)
                trajectory_ax.set_zlim(0,15)
            else:
                # Plot
                if args.plot_uncontrolled:
                    # Plot the zero trajectory on the same graph
                    trajectory_ax.plot3D(x_zero, y_zero, z_zero, color='tab:blue', zorder=2)
                trajectory_ax.plot3D(x, y, z, linewidth=3, color='tab:orange', zorder=2)

                # Adjust the view angle for better visibility
                # if env_enum == EnvEnum.DoubleWell:
                #     trajectory_ax.view_init(elev=20, azim=45)

                # Adjust layout to reduce white space
                plt.tight_layout(pad=0.01)

                # Turn off grid
                trajectory_ax.grid(False)

                # Turn off axis
                trajectory_ax.set_axis_off()

            # Save trajectory frame as image
            trajectory_frame_path = os.path.join(args.data_folder, f"trajectory_frame_{step_num+1}.png")
            plt.savefig(trajectory_frame_path)
            plt.cla()

            # Append frame to list for GIF creation
            trajectory_frames.append(imageio.imread(trajectory_frame_path))

        # Print out progress
        if step_num == 0 or (step_num+1) % 100 == 0:
            print(f"Created {step_num+1} trajectory video frame(s)")

    trajectory_gif_path = os.path.join(args.data_folder, f"trajectory_{trajectory_num+1}.gif")
    imageio.mimsave(trajectory_gif_path, trajectory_frames, duration=0.1)

    # Create cost figure
    cost_fig = plt.figure(figsize=(21, 14), dpi=300)
    cost_ax = cost_fig.add_subplot(111)
    cost_ax.set_ylabel('')  # Removes the y-axis label on a specific axis

    for cost_num in range(main_policy_costs.shape[0]):
        cost_frames = []

        # Calculate the cost ratios for this iteration
        all_main_costs = main_policy_costs[cost_num]
        all_baseline_costs = baseline_policy_costs[cost_num]
        all_cost_ratios = all_main_costs / all_baseline_costs
        log_all_cost_ratios = np.log(all_main_costs / all_baseline_costs)
        moving_average_log_all_cost_ratios = moving_average(
            log_all_cost_ratios,
            n=ma_window_size,
            keep_first=False
        )

        # Calculate the x values for the moving average
        cost_x_values_start_index = ma_window_size - 1
        moving_average_log_all_cost_ratio_x_values = np.arange(
            cost_x_values_start_index,
            moving_average_log_all_cost_ratios.shape[0] + cost_x_values_start_index
        )

        # Calculate the overall min and max for consistent scaling
        min_log_cost_ratio = np.min(log_all_cost_ratios)
        max_log_cost_ratio = np.max(log_all_cost_ratios)

        for step_num in range(main_policy_costs.shape[1]):
            # Only save every n steps
            if step_num == 0 or (step_num+1) % args.save_every_n_steps == 0:
                log_cost_ratio = log_all_cost_ratios[:(step_num+1)]
                # Plot moving average only if there are enough values
                if step_num >= cost_x_values_start_index:
                    moving_average_log_cost_ratio = moving_average_log_all_cost_ratios[:(step_num+1)-cost_x_values_start_index]
                    moving_average_log_cost_ratio_x_value = moving_average_log_all_cost_ratio_x_values[:(step_num+1)-cost_x_values_start_index]
                else:
                    moving_average_log_cost_ratio = np.array([])
                    moving_average_log_cost_ratio_x_value = np.array([])

                # Clear the previous plot
                cost_ax.clear()

                # Set axis limits
                cost_ax.set_xlim(0, main_policy_costs.shape[1])
                cost_ax.set_ylim(min_log_cost_ratio * 1.1, max_log_cost_ratio * 1.1)

                # Set axis labels
                cost_ax.set_xlabel("Steps")
                cost_ax.set_ylabel(f"Cost Ratio ({metadata['main_policy_name']} / {metadata['baseline_policy_name']})")

                # Set axis title
                cost_ax.set_title(f"Cost Ratio: {metadata['main_policy_name']} / {metadata['baseline_policy_name']}")

                # Plot a horizontal line at y=0
                cost_ax.axhline(y=0, color='r', linestyle='--')

                # Make Title larger
                cost_ax.title.set_fontsize(20)

                # Adjust layout to reduce white space
                plt.tight_layout(pad=0.1)

                # Turn on grid
                cost_ax.grid(True)

                # Plot values
                cost_ax.plot(log_cost_ratio, alpha=0.5)
                cost_ax.plot(moving_average_log_cost_ratio_x_value, moving_average_log_cost_ratio, linewidth=3)

                # Save trajectory frame as image
                cost_frame_path = os.path.join(args.data_folder, f"cost_frame_{step_num+1}.png")
                plt.savefig(cost_frame_path)

                # Append frame to list for GIF creation
                cost_frames.append(imageio.imread(cost_frame_path))

            # Print out progress
            if step_num == 0 or (step_num+1) % 100 == 0:
                print(f"Created {step_num+1} cost video frame(s)")

        cost_gif_path = os.path.join(args.data_folder, f"costs_{trajectory_num+1}.gif")
        imageio.mimsave(cost_gif_path, cost_frames, duration=0.1)

plt.close(trajectory_fig)
plt.close(cost_fig)

print(f"Plots and GIFs saved in {args.data_folder}")