# Imports
import argparse
import gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from analysis.utils import create_folder
from custom_envs import *
from movies.default_policies import ZeroPolicy
from movies.generator import Generator

# Allow environment ID to be passed as command line argument
parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="FluidFlow-v0",
                    help='Gym environment (default: FluidFlow-v0)')
args = parser.parse_args()

# Create gym env with ID
env = gym.make(args.env_id)

# Create instance of policy and generator
policy = ZeroPolicy()
generator = Generator(env, policy)

# Generate trajectories
trajectories = generator.generate_trajectories(num_trajectories=1) # (num_trajectories, steps_per_trajectory, state_dim)

# Make sure folders exist for storing video data
output_folder = f"video_frames/{args.env_id}_{int(time.time())}"
create_folder(output_folder)

# Store the trajectories on hard drive
np.save(f"{output_folder}/trajectories.npy", trajectories)

# Plot trajectories
fig = plt.figure()
is_3d_env = args.env_id != 'DoubleWell-v0'
if is_3d_env:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)

for trajectory_num in range(trajectories.shape[0]):
    frames = []

    # Calculate min and max values
    min_x = trajectories[trajectory_num, :, 0].min()
    max_x = trajectories[trajectory_num, :, 0].max()
    min_y = trajectories[trajectory_num, :, 1].min()
    max_y = trajectories[trajectory_num, :, 1].max()
    if is_3d_env:
        min_z = trajectories[trajectory_num, :, 2].min()
        max_z = trajectories[trajectory_num, :, 2].max()

    for step_num in range(trajectories.shape[1]):
        # Set axis limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        if is_3d_env:
            ax.set_zlim(min_z, max_z)

        if is_3d_env:
            ax.plot3D(
                trajectories[trajectory_num, :(step_num+1), 0],
                trajectories[trajectory_num, :(step_num+1), 1],
                trajectories[trajectory_num, :(step_num+1), 2]
            )
        else:
            ax.plot(
                trajectories[trajectory_num, :(step_num+1), 0],
                trajectories[trajectory_num, :(step_num+1), 1]
            )

        # Save frame as image
        frame_path = os.path.join(output_folder, f"frame_{step_num}.png")
        plt.savefig(frame_path)
        plt.cla()

        # Append frame to list for GIF creation
        frames.append(imageio.imread(frame_path))

    gif_path = os.path.join(output_folder, f"trajectory_{trajectory_num}.gif")
    imageio.mimsave(gif_path, frames, duration=0.1)