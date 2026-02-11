import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cleanrl.sac_continuous_action import SoftQNetwork
from stable_baselines3.common.buffers import ReplayBuffer
from tap import Tap
from torch.utils.tensorboard import SummaryWriter

from koopmanrl.environments import DoubleWell, FluidFlow, LinearSystem, Lorenz
from koopmanrl.utils import create_folder, make_env

torch.set_default_dtype(torch.float64)
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ArgumentParser(Tap):
    exp_name: str = os.path.basename(__file__).rstrip(".py")  # the name of this experiment
    seed: int = 1  # seed of the experiment (default: 1)
    torch_deterministic: bool = True  # if toggled, `torch.backends.cudnn.deterministic=False` (default: True)
    cuda: bool = False  # if toggled, cuda will be enabled by default (default: True)
    capture_video: bool = (
        False  # whether to capture videos of the agent performances (check out `videos` folder; default: False)
    )
    env_id: str = "LinearSystem-v0"  # the id of the environment (default: LinearSystem-v0)
    total_timesteps: int = 50000  # total timesteps of the experiments (default: 50000)
    buffer_size: int = int(1e6)  # the replay memory buffer size (default: 1000000)
    gamma: float = 0.99  # the discount factor gamma (default: 0.99)
    tau: float = 0.005  # target smoothing coefficient (default: 0.005)
    batch_size: int = 256  # the batch size of sample from the reply memory (default: 256)
    learning_starts: int = int(5e3)  # timestep to start learning (default: 5000)
    policy_lr: float = 3e-4  # the learning rate of the policy network optimizer (default: 0.0003)
    v_lr: float = 1e-3  # the learning rate of the V network optimizer (default: 0.001)
    q_lr: float = 1e-3  # the learning rate of the Q network optimizer (default: 0.001)
    policy_frequency: int = 2  # the frequency of training policy (delayed; default: 2)
    target_network_frequency: int = 1  # the frequency of updates for the target nerworks (default: 1)
    noise_clip: float = 0.5  # noise clip parameter of the Target Policy Smoothing Regularization (default: 0.5)
    alpha: float = 0.2  # Entropy regularization coefficient (default: 0.2)
    autotune: bool = True  # automatic tuning of the entropy coefficient (default: True)
    alpha_lr: float = 1e-3  # the learning rate of the alpha network optimizer (default: 0.001)


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


def main():
    args = ArgumentParser().parse_args()
    curr_time = int(time.time())

    # Generate a random seed
    sampled_seed = np.random.randint(1000)

    run_name = f"{args.env_id}__{args.exp_name}__{sampled_seed}__{curr_time}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create folder for model checkpoints
    model_chkpt_path = f"./saved_models/{args.env_id}/value_based_sac_chkpts_{curr_time}"
    create_folder(model_chkpt_path)

    # TRY NOT TO MODIFY: seeding
    random.seed(sampled_seed)
    np.random.seed(sampled_seed)
    torch.manual_seed(sampled_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Running everything on CPU
    device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, sampled_seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)

    vf = SoftVNetwork(envs).to(device)
    vf_target = SoftVNetwork(envs).to(device)
    vf_target.load_state_dict(vf.state_dict())
    v_optimizer = optim.Adam(list(vf.parameters()), lr=args.v_lr)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

    # envs.single_observation_space.dtype = np.float32
    envs.single_observation_space.dtype = np.float64
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, log_probs, _ = actor.get_action(torch.Tensor(obs).to(device))
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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # Sample from replay buffer
            data = rb.sample(args.batch_size)

            # E_s_t~D [ 1/2 ( V_psi( s_t ) - E_a_t~pi_phi [ Q_theta( s_t, a_t ) - log pi_phi( a_t | s_t ) ] )^2 ]
            vf_values = vf(data.observations).view(-1)
            with torch.no_grad():
                state_actions, state_log_pis, _ = actor.get_action(data.observations)
                q_values = torch.min(qf1(data.observations, state_actions), qf2(data.observations, state_actions)).view(
                    -1
                )
            vf_loss = F.mse_loss(vf_values, q_values - alpha * state_log_pis.view(-1))

            v_optimizer.zero_grad()
            vf_loss.backward()
            v_optimizer.step()

            # E_( s_t, a_t )~D [ 1/2 ( Q_theta( s_t, a_t ) - Q_target( s_t, a_t ) )^2 ]
            with torch.no_grad():
                vf_next_target = (1 - data.dones.flatten()) * args.gamma * vf_target(data.next_observations).view(-1)
                q_target_values = data.rewards.flatten() + vf_next_target

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, q_target_values)
            qf2_loss = F.mse_loss(qf2_a_values, q_target_values)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # E_s_t~D,e_t~N [ log pi_phi( f_phi( e_t; s_t ) | s_t ) - Q_theta( s_t, f_phi( e_t; s_t ) ) ]
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(vf.parameters(), vf_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/vf_values", vf_values.mean().item(), global_step)
                writer.add_scalar("losses/vf_loss", vf_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                print("Steps per second (SPS):", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

            # Checkpoint policy network every so often
            if global_step == 0 or (global_step + 1) % 1000 == 0:
                torch.save(actor.state_dict(), f"{model_chkpt_path}/step_{global_step + 1}.pt")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
