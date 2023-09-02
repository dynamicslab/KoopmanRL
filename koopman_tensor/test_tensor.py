""" Imports """

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from custom_envs import *
from distutils.util import strtobool
from koopman_tensor.numpy_tensor import KoopmanTensor, Regressor
from koopman_tensor.observables import numpy_observables as observables
# from koopman_tensor.torch_tensor import KoopmanTensor as TorchKoopmanTensor, Regressor as TorchRegressor
# from koopman_tensor.observables import torch_observables
from matplotlib.animation import FuncAnimation

""" Allow environment specification """

parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="LinearSystem-v0",
                    help='Gym environment (default: LinearSystem-v0)')
parser.add_argument('--num-paths', type=int, default=100,
                    help='Number of paths for the dataset (default: 100)')
parser.add_argument('--num-steps-per-path', type=int, default=300,
                    help='Number of steps per path for the dataset (default: 300)')
parser.add_argument('--seed', type=int, default=123,
                    help='Seed for some level of reproducibility (default: 123)')
parser.add_argument('--save-model', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                    help='Whether to store the Koopman tensor model in a pickle file (default: False)')
args = parser.parse_args()

""" Create the environment """

if args.env_id == "DoubleWell-v0":
    is_3d_env = False
else:
    is_3d_env = True

env = gym.make(args.env_id)

""" Set seed """

# TODO: Reproducibility is broken in the custom envs

env.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

""" Collect data """

# Path-based data collection
X = np.zeros((args.num_paths, args.num_steps_per_path, env.state_dim))
Y = np.zeros_like(X)
U = np.zeros((args.num_paths, args.num_steps_per_path, env.action_dim))

for path_num in range(args.num_paths):
    state = env.reset()
    for step_num in range(args.num_steps_per_path):
        X[path_num, step_num] = state

        # action = np.array([0])
        action = env.action_space.sample()
        U[path_num, step_num] = action

        state, _, _, _ = env.step(action)
        Y[path_num, step_num] = state

""" Make sure trajectories look ok """

# Create a figure and 3D axis
fig = plt.figure()
if is_3d_env:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)

# Set limits for each axis
ax.set_xlim(X[0:, :, 0].min(), X[0:, :, 0].max())
ax.set_ylim(X[0:, :, 1].min(), X[0:, :, 1].max())
if is_3d_env:
    ax.set_zlim(X[0:, :, 2].min(), X[0:, :, 2].max())

# Initialize an empty line for the animation
if is_3d_env:
    line, = ax.plot([], [], [], lw=2)
else:
    line, = ax.plot([], [], lw=2)

# Function to initialize the plot
def init():
    line.set_data([], [])
    if is_3d_env:
        line.set_3d_properties([])
    return line,

# Set the number of frames
num_frames = X.shape[1]

# Function to update the plot for each frame of the animation
def animate(i):
    x = X[0, :i, 0]
    y = X[0, :i, 1]
    if is_3d_env:
        z = X[0, :i, 2]
    line.set_data(x, y)
    if is_3d_env:
        line.set_3d_properties(z)

    # Stop the animation when it's done
    if i == num_frames - 1:
        ani.event_source.stop()
        plt.close(fig)

    return line,

# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True, repeat=False)

plt.tight_layout()
plt.show()

""" Reshape data so that we have matrices of data instead of tensor """

total_num_datapoints = args.num_paths * args.num_steps_per_path

X = X.reshape(total_num_datapoints, env.state_dim).T
Y = Y.reshape(total_num_datapoints, env.state_dim).T
U = U.reshape(total_num_datapoints, env.action_dim).T

""" Construct Koopman tensor """

state_order = 2
action_order = 2

path_based_tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor=Regressor.OLS
)

""" Predict sample points """

sample_indices = (0, X.shape[1])
sample_x = X[:, sample_indices[0]:sample_indices[1]]
sample_u = U[:, sample_indices[0]:sample_indices[1]]

true_x_prime = Y[:, sample_indices[0]:sample_indices[1]]
estimated_x_prime = path_based_tensor.f(sample_x, sample_u)

single_step_estimation_error_norms = np.linalg.norm(true_x_prime - estimated_x_prime, axis=0)
avg_single_step_estimation_error_norm = single_step_estimation_error_norms.mean()
avg_single_step_estimation_error_norm_per_avg_state_norm = avg_single_step_estimation_error_norm / np.linalg.norm(X.mean(axis=1))
print(f"Average single step estimation error norm per average state norm: {avg_single_step_estimation_error_norm_per_avg_state_norm}")

""" Save Koopman tensor """

if args.save_model:
    with open(f'./koopman_tensor/saved_models/{args.env_id}/path_based_tensor.pickle', 'wb') as handle:
        pickle.dump(path_based_tensor, handle)