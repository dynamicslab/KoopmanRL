# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from custom_envs import *
# from koopman_tensor.utils import load_tensor

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment (default: 1)")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False` (default: True)")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default (default: True)")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases (default: False)")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name (default: \"cleanRL\")")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project (default: None)")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder; default: False)")

    # Algorithm specific arguments
    # parser.add_argument("--env-id", type=str, default="Hopper-v4",
    #     help="the id of the environment")
    # parser.add_argument("--env-id", type=str, default="Hopper-v3",
    #     help="the id of the environment")
    parser.add_argument("--env-id", type=str, default="LinearSystem-v0",
        help="the id of the environment (default: LinearSystem-v0)")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments (default: 1000000)")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size (default: 1000000)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory (default: 256)")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning (default: 5000)")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer (default: 0.0003)")
    parser.add_argument("--v-lr", type=float, default=1e-3,
        help="the learning rate of the V network optimizer (default: 0.001)")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network optimizer (default: 0.001)")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed; default: 2)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks (default: 1)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient (default: 0.2)")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient (default: True)")
    parser.add_argument("--alpha-lr", type=float, default=1e-3,
        help="the learning rate of the alpha network optimizer (default: 0.001)")
    parser.add_argument("--koopman", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="use Koopman V function (default: False)")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class SoftVNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class SoftKoopmanVNetwork(nn.Module):
    def __init__(self, koopman_tensor):
        super().__init__()

        self.koopman_tensor = koopman_tensor
        self.phi_state_dim = self.koopman_tensor.Phi_X.shape[0]

        self.linear = nn.Linear(self.phi_state_dim, 1, bias=False)

    def forward(self, state):
        """ Linear in the phi(x)s """

        phi_xs = self.koopman_tensor.phi(state.T).T

        output = self.linear(phi_xs)

        return output


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        # action rescaling
        high_action = env.action_space.high
        low_action = env.action_space.low
        # high_action = np.clip(env.action_space.high, a_min=-1000, a_max=1000)
        # low_action = np.clip(env.action_space.low, a_min=-1000, a_max=1000)
        # dtype = torch.float32
        dtype = torch.float64
        action_scale = torch.tensor((high_action - low_action) / 2.0, dtype=dtype)
        action_bias = torch.tensor((high_action + low_action) / 2.0, dtype=dtype)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    actor.load_state_dict(torch.load(f'./saved_models/{args.env_id}/value_based_sa{"k" if args.koopman else ""}c_actor.pt'))

    envs.single_observation_space.dtype = np.float64
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Write data
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            print("Steps per second (SPS):", sps)
            writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()