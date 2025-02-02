"""
Example usage:
python -m movies.generate_trajectories --env-id=FluidFlow-v0
"""

# Imports
import argparse
import gym
import numpy as np
import time
import torch

from analysis.utils import create_folder
from custom_envs import *
from distutils.util import strtobool
from movies.algo_policies import *
from movies.default_policies import ZeroPolicy
from movies.env_enum import EnvEnum
from movies.generator import Generator

# Command-line arguments
parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="FluidFlow-v0",
    help='Gym environment (default: FluidFlow-v0)')
parser.add_argument("--seed", type=int, default=123,
    help="seed of the experiment (default: 123)")
parser.add_argument('--num-trajectories', type=int, default=1,
    help="number of trajectories to generate (default: 1)")
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

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Function to reset seeds
def reset_seed():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

# Set seeds
reset_seed()

# Create gym env with ID
envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])

""" CREATE YOUR POLICY INSTANCES HERE """

timestamp_map = {
    EnvEnum.LinearSystem: {
        "SAKC": 1732367820,
        "SAC (V)": 1732364757,
        "SAC (Q)": 1738509308,
        "SKVI": 1738511301,
    },
    EnvEnum.FluidFlow: {
        "SAKC": 1732368170,
        "SAC (V)": 1732365537,
        "SAC (Q)": 1738509644,
        "SKVI": 1738511504,
    },
    EnvEnum.Lorenz: {
        "SAKC": 1732369019,
        "SAC (V)": 1732366427,
        "SAC (Q)": 1738509985,
        "SKVI": 1738511670,
    },
    EnvEnum.DoubleWell: {
        "SAKC": 1732368563,
        "SAC (V)": 1732367266,
        "SAC (Q)": 1738510316,
        "SKVI": 1738511824,
    },
}

# Zero Policy
zero_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

# Main policy
# main_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

# main_policy = LQR(
#     args=args,
#     envs=envs
# )

# main_policy = SKVI(
#     args=args,
#     envs=envs,
#     saved_koopman_model_name="path_based_tensor",
#     trained_model_start_timestamp=timestamp_map[args.env_id]["SKVI"],
#     chkpt_epoch_number=150,
#     device=device,
#     name="SKVI"
# )

# main_policy = SAKC(
#     args=args,
#     envs=envs,
#     is_value_based=True,
#     is_koopman=True,
#     chkpt_timestamp=timestamp_map[args.env_id]["SAKC"],
#     chkpt_step_number=50_000,
#     device=device,
#     name = "SAKC"
# )

# main_policy = SAKC(
#     args=args,
#     envs=envs,
#     is_value_based=True,
#     is_koopman=False,
#     chkpt_timestamp=timestamp_map[args.env_id]["SAC (V)"],
#     chkpt_step_number=50_000,
#     device=device,
#     name = "SAC (V)"
# )

main_policy = SAKC(
    args=args,
    envs=envs,
    is_value_based=False,
    is_koopman=False,
    chkpt_timestamp=timestamp_map[args.env_id]["SAC (Q)"],
    chkpt_step_number=50_000,
    device=device,
    name = "SAC (Q)"
)

# Baseline policy
baseline_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

# LQR Baseline
# baseline_policy = LQR(
#     args=args,
#     envs=envs
# )

# SAC (V) Baseline
# baseline_policy = SAKC(
#     args=args,
#     envs=envs,
#     is_value_based=True,
#     is_koopman=False,
#     chkpt_timestamp=timestamp_map[args.env_id]["SAC (V)"],
#     chkpt_step_number=50_000,
#     device=device,
#     name="SAC (V)"
# )

# Create generator
zero_policy_generator = Generator(args, envs, zero_policy)
main_policy_generator = Generator(args, envs, main_policy)
baseline_policy_generator = Generator(args, envs, baseline_policy)

def reset_seed():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# Generate trajectories
reset_seed()
(
    zero_policy_trajectories,
    zero_policy_actions,
    zero_policy_costs
) = zero_policy_generator.generate_trajectories(args.num_trajectories)  # (num_trajectories, steps_per_trajectory, state_dim)
reset_seed()
(
    main_policy_trajectories,
    main_policy_actions,
    main_policy_costs
) = main_policy_generator.generate_trajectories(args.num_trajectories)  # (num_trajectories, steps_per_trajectory, state_dim)
reset_seed()
(
    baseline_policy_trajectories,
    baseline_policy_actions,
    baseline_policy_costs
) = baseline_policy_generator.generate_trajectories(args.num_trajectories)  # (num_trajectories, steps_per_trajectory, state_dim)
print("Completed generating trajectories")

# We will not store the generated trajectories if they do not
# have the same initial conditions, so we check that with an assertion
assert (
    np.array_equal(zero_policy_trajectories[0, 0], main_policy_trajectories[0, 0]) and
    np.array_equal(main_policy_trajectories[0, 0], baseline_policy_trajectories[0, 0])
), "Your trajectories have different initial conditions"

# Save additional metadata
metadata = {
    "env_id": args.env_id,
    "main_policy_name": main_policy.name,
    "baseline_policy_name": baseline_policy.name,
    "zero_policy_name": zero_policy.name,
}
print(f"Metadata: {metadata}")

# Make sure folders exist for storing video data
curr_time = int(time.time())
output_folder = f"video_frames/{args.env_id}_{curr_time}"
create_folder(output_folder)

# Save zero policy trajectories and costs
np.save(f"{output_folder}/zero_policy_trajectories.npy", zero_policy_trajectories)
np.save(f"{output_folder}/zero_policy_actions.npy", zero_policy_actions)
np.save(f"{output_folder}/zero_policy_costs.npy", zero_policy_costs)
# Store the trajectories on hard drive
np.save(f"{output_folder}/main_policy_trajectories.npy", main_policy_trajectories)
np.save(f"{output_folder}/main_policy_actions.npy", main_policy_actions)
np.save(f"{output_folder}/main_policy_costs.npy", main_policy_costs)
# Save baseline policy trajectories and costs
np.save(f"{output_folder}/baseline_policy_trajectories.npy", baseline_policy_trajectories)
np.save(f"{output_folder}/baseline_policy_actions.npy", baseline_policy_actions)
np.save(f"{output_folder}/baseline_policy_costs.npy", baseline_policy_costs)
# Save metadata
np.save(f"{output_folder}/metadata.npy", metadata, allow_pickle=True)

# Print out success message and data path
print(f"Saved trajectories to {output_folder}")