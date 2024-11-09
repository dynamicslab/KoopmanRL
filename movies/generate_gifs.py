"""
Example usage:
python -m movies.generate_gifs --save-every-n-steps=100 --data-folder=./video_frames/FluidFlow-v0_1723125311
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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=123,
    help="seed of the experiment (default: 123)")
parser.add_argument("--data-folder", type=str, required=True,
    help="Folder containing trajectory data")
parser.add_argument("--save-every-n-steps", type=int, default=None,
    help="Save a frame every n steps.")
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

# Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
main_policy_trajectories = np.load(f"{args.data_folder}/main_policy_trajectories.npy")
main_policy_costs = np.load(f"{args.data_folder}/main_policy_costs.npy")
baseline_policy_costs = np.load(f"{args.data_folder}/baseline_policy_costs.npy")
metadata = np.load(f"{args.data_folder}/metadata.npy", allow_pickle=True).item()
is_double_well = metadata['env_id'] == 'DoubleWell-v0'

# Create gym env with ID
envs = gym.vector.SyncVectorEnv([make_env(metadata['env_id'], args.seed)])

# Plot trajectories
for trajectory_num in range(main_policy_trajectories.shape[0]):
    # Create trajectory figure
    trajectory_fig = plt.figure(figsize=(21, 14), dpi=300)
    trajectory_ax = trajectory_fig.add_subplot(111, projection='3d')

    # Create array of frames
    trajectory_frames = []

    # Extract trajectory data
    full_x = main_policy_trajectories[trajectory_num, :, 0]
    full_y = main_policy_trajectories[trajectory_num, :, 1]
    if is_double_well:
        full_u = main_policy_trajectories[trajectory_num, :, 2]
        full_z = main_policy_trajectories[trajectory_num, :, 3]
    else:
        full_z = main_policy_trajectories[trajectory_num, :, 2]

    if is_double_well:
        step_size = 0.1
        X, Y = np.meshgrid(
            np.arange(start=envs.envs[0].state_minimums[0], stop=envs.envs[0].state_maximums[0]+step_size, step=step_size),
            np.arange(start=envs.envs[0].state_minimums[1], stop=envs.envs[0].state_maximums[1]+step_size, step=step_size),
        )

    for step_num in range(main_policy_trajectories.shape[1]):
        if step_num == 0 or (step_num+1) % args.save_every_n_steps == 0:  # Only save every n steps
            x = full_x[:(step_num+1)]
            y = full_y[:(step_num+1)]
            z = full_z[:(step_num+1)]
            if is_double_well:
                u = full_u[:(step_num+1)]
                Z = envs.envs[0].potential(X, Y, u[step_num])
                Z_path = envs.envs[0].potential(x, y, u[step_num])

            # Clear the previous plot
            trajectory_ax.clear()

            # Reapply the axis settings
            trajectory_ax.set_xticklabels([])
            trajectory_ax.set_yticklabels([])
            trajectory_ax.set_yticklabels([])

            # Set axis limits
            trajectory_ax.set_xlim(envs.envs[0].state_minimums[0], envs.envs[0].state_maximums[0])
            trajectory_ax.set_ylim(envs.envs[0].state_minimums[1], envs.envs[0].state_maximums[1])
            if not is_double_well:
                trajectory_ax.set_zlim(envs.envs[0].state_minimums[2], envs.envs[0].state_maximums[2])

            # Plot trajectory
            if is_double_well:
                # trajectory_ax.contour(X, Y, Z)
                # trajectory_ax.plot(x, y)

                trajectory_ax.contour(X, Y, Z)
                trajectory_ax.plot3D(x, y, Z_path, alpha=1.0, linewidth=2, color='black')
                trajectory_ax.plot_surface(X, Y, Z, alpha=0.7, cmap=cm.coolwarm)
                trajectory_ax.set_zlim(0,15)
            else:
                trajectory_ax.plot3D(x, y, z)

                # Adjust the view angle for better visibility
                trajectory_ax.view_init(elev=20, azim=45)

                # Adjust layout to reduce white space
                plt.tight_layout(pad=0.1)

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
    cost_fig = plt.figure(figsize=(17, 11), dpi=300)
    cost_ax = cost_fig.add_subplot(111)

    for cost_num in range(main_policy_costs.shape[0]):
        cost_frames = []

        # Calculate the overall min and max for consistent scaling
        all_main_costs = main_policy_costs[cost_num]
        all_baseline_costs = baseline_policy_costs[cost_num]
        all_cost_ratios = all_main_costs / all_baseline_costs
        log_all_cost_ratios = np.log(all_main_costs / all_baseline_costs)
        # min_cost_ratio = np.min(all_cost_ratios)
        # max_cost_ratio = np.max(all_cost_ratios)
        min_log_cost_ratio = np.min(log_all_cost_ratios)
        max_log_cost_ratio = np.max(log_all_cost_ratios)

        for step_num in range(main_policy_costs.shape[1]):
            if step_num == 0 or (step_num+1) % args.save_every_n_steps == 0:  # Only save every n steps
                # cost_ratio = all_cost_ratios[:(step_num+1)]
                log_cost_ratio = log_all_cost_ratios[:(step_num+1)]

                # Clear the previous plot
                cost_ax.clear()

                # Set axis limits
                cost_ax.set_xlim(0, main_policy_costs.shape[1])
                # cost_ax.set_ylim(max(0, min_cost_ratio * 0.9), max_cost_ratio * 1.1)
                cost_ax.set_ylim(min_log_cost_ratio * 1.1, max_log_cost_ratio * 1.1)
                # cost_ax.set_ylim(0, 100)

                # Set axis labels
                cost_ax.set_xlabel("Steps")
                cost_ax.set_ylabel(f"Cost Ratio ({metadata['main_policy_name']} / {metadata['baseline_policy_name']})")

                # Set axis title
                cost_ax.set_title(f"Cost Ratio: {metadata['main_policy_name']} / {metadata['baseline_policy_name']}")

                # Plot a horizontal line at y=1
                # cost_ax.axhline(y=1, color='r', linestyle='--')  # Line at y=1 if using cost ratio
                cost_ax.axhline(y=0, color='r', linestyle='--')  # Line at y=0 using log cost ratio

                # Turn on grid lines
                cost_ax.grid()

                # Adjust layout to reduce white space
                plt.tight_layout()

                # Plot values
                # cost_ax.plot(cost_ratio)
                cost_ax.plot(log_cost_ratio)

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