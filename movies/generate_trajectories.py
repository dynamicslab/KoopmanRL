# Imports
import argparse
import gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch

from analysis.utils import create_folder
from custom_envs import *
from distutils.util import strtobool
from movies.default_policies import ZeroPolicy
from movies.algo_policies import LQR, SAKC, SKVI
from movies.generator import Generator

# Allow environment ID to be passed as command line argument
parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="FluidFlow-v0",
                    help='Gym environment (default: FluidFlow-v0)')
parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment (default: 1)")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False` (default: True)")
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="if toggled, cuda will be enabled (default: True)")
parser.add_argument("--num-actions", type=int, default=101,
        help="number of actions that the policy can pick from (default: 101)")
parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
parser.add_argument("--alpha", type=float, default=1.0,
        help="entropy regularization coefficient (default: 1.0)")
args = parser.parse_args()

# Initialize device and run name
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# Help making vectorized envs
def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Create gym env with ID
env = gym.make(args.env_id)
envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])

""" CREATE YOUR POLICY INSTANCE HERE """

# policy = ZeroPolicy()

policy = LQR(
    args=args,
    envs=envs
)

# policy = SKVI(
#     args=args,
#     envs=envs,
#     saved_koopman_model_name="path_based_tensor",
#     path_to_checkpoint=f"./saved_models",
#     device=device
# )

# policy = SAKC(
#     envs=envs,
#     path_to_checkpoint=f"./saved_models/{args.env_id}/sac_chkpts_1889579283/step_999.pt",
#     device=device
# )

""" TRY NOT TO CHANGE ANYTHING BELOW """

# Create generator
generator = Generator(env, policy)

# Generate trajectories
trajectories = generator.generate_trajectories(num_trajectories=1) # (num_trajectories, steps_per_trajectory, state_dim)

# Make sure folders exist for storing video data
curr_time = int(time.time())
output_folder = f"video_frames/{args.env_id}_{curr_time}"
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
        ax.set_xlim(env.state_minimums[0], env.state_maximums[0])
        ax.set_ylim(env.state_minimums[1], env.state_maximums[1])
        if is_3d_env:
            ax.set_zlim(env.state_minimums[2], env.state_maximums[2])

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