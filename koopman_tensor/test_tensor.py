""" Imports """

import argparse
import gym
import numpy as np
import pickle
import torch

from custom_envs import *
from koopman_tensor.torch_tensor import KoopmanTensor
from koopman_tensor.observables import torch_observables as observables

""" Allow environment specification """

parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="LinearSystem-v0",
                    help='Gym environment (default: LinearSystem-v0)')
parser.add_argument('--num-paths', default=100,
                    help='Number of paths for the dataset (default: 100)')
parser.add_argument('--num-steps-per-path', default=300,
                    help='Number of steps per path for the dataset (default: 300)')
args = parser.parse_args()

""" Collect data """

# Create the environment
env = gym.make(args.env_id)

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
        state, reward, _, _ = env.step(action)
        Y[path_num, step_num] = state

total_num_datapoints = args.num_paths * args.num_steps_per_path

X = X.reshape(total_num_datapoints, env.state_dim).T
Y = Y.reshape(total_num_datapoints, env.state_dim).T
U = U.reshape(total_num_datapoints, env.action_dim).T

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
U = torch.tensor(U, dtype=torch.float32)

""" Construct Koopman tensor """

state_order = 2
action_order = 2

path_based_tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
    # regressor='sindy'
)

""" Predict sample point """

sample_index = 0
sample_x = X[:, sample_index:1]
sample_u = U[:, sample_index:1]
estimated_phi_x_prime = path_based_tensor.phi_f(sample_x, sample_u)
print(estimated_phi_x_prime)
sample_x = X[:, sample_index:2]
sample_u = U[:, sample_index:2]
estimated_phi_x_prime = path_based_tensor.phi_f(sample_x, sample_u)
print(estimated_phi_x_prime)

true_x_prime = Y[:, sample_index:2]
estimated_x_prime = path_based_tensor.f(sample_x, sample_u)
print(true_x_prime)
print(estimated_x_prime)

""" Save Koopman tensor """

# with open('./analysis/tmp/path_based_tensor.pickle', 'wb') as handle:
#     pickle.dump(path_based_tensor, handle)