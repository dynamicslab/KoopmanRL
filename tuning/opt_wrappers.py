# TODO:
#   Wrapper function for:
#   - SKVI
#   - SAKC
# --> Tie the PPO example together post-dinner

import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from cleanrl.ppo import make_env, Agent


def ppo_tuning_wrapper(
    seed=1,
    is_torch_deterministic=True,
    use_cuda=True,
    to_capture_video=False,
    env_id="CartPole-v1",
    number_of_environments=4,
    learning_rate=2.5e-4,
    total_timesteps=500000,
    number_of_steps=128,
    annealing_learning_rate=True,
    update_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    clip_coefficient=0.2,
    normalization_advantages=True,
    clipped_loss_vfunc=True,
    entropy_coefficient=0.01,
    vfunc_coefficient=0.5,
    max_gradient_norm=0.5,
    target_kl_divergence=None,
):
    """
    Special version of CleanRL's PPO implementation just for the use with Ray Tune. The major differences are twofold:
    - No argparsing, instead inputs are direct to the function
    - No storage to Tensorboard files, instead the files live in a buffer.

    Outside of that there has only been some dead code removal
    """

    batch_size = number_of_environments * number_of_steps
    minibatch_size = int(batch_size // num_minibatches)
    run_name = f"ppo__{env_id}__{seed}__{learning_rate}__{vfunc_coefficient}__{max_gradient_norm}"

    # Initialize the lists to store results into
    episodic_returns = []
    episodic_lengths = []
    learning_rates = []
    value_losses = []
    policy_gradient_losses = []
    entropy_losses = []
    old_approx_kullbackleibler = []
    approx_kullbackleibler = []
    mean_clipfracs = []
    explained_variances = []
    steps_per_seconds = []

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = is_torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(env_id, seed + i, i, to_capture_video, run_name)
            for i in range(number_of_environments)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (number_of_steps, number_of_environments) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (number_of_steps, number_of_environments) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((number_of_steps, number_of_environments)).to(device)
    rewards = torch.zeros((number_of_steps, number_of_environments)).to(device)
    dones = torch.zeros((number_of_steps, number_of_environments)).to(device)
    values = torch.zeros((number_of_steps, number_of_environments)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(number_of_environments).to(device)
    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if annealing_learning_rate:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, number_of_steps):
            global_step += 1 * number_of_environments
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            # Writing action
            for item in info:
                if "episode" in item.keys():
                    episodic_returns.append([item["episode"]["r"], global_step])
                    episodic_lengths.append([item["episode"]["l"], global_step])
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(number_of_steps)):
                if t == number_of_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coefficient).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if normalization_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coefficient, 1 + clip_coefficient
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clipped_loss_vfunc:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coefficient,
                        clip_coefficient,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - entropy_coefficient * entropy_loss
                    + v_loss * vfunc_coefficient
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_gradient_norm)
                optimizer.step()

            if target_kl_divergence is not None:
                if approx_kl > target_kl_divergence:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record the results
        learning_rates.append([optimizer.param_groups[0]["lr"], global_step])
        value_losses.append([v_loss.item(), global_step])
        policy_gradient_losses.append([pg_loss.item(), global_step])
        entropy_losses.append([entropy_loss.item(), global_step])
        old_approx_kullbackleibler.append([old_approx_kl.item(), global_step])
        approx_kullbackleibler.append([approx_kl.item(), global_step])
        mean_clipfracs.append([np.mean(clipfracs), global_step])
        explained_variances.append([explained_var, global_step])
        steps_per_seconds.append(
            [int(global_step / (time.time() - start_time)), global_step]
        )

    # Close the gym environment
    envs.close()

    # Return outputs relevant to hyperparameter optimization
    return {
        "charts/episodic_return": episodic_returns,
        "charts/episodic_length": episodic_lengths,
        "charts/learning_rate": learning_rates,
        "losses/value_loss": value_losses,
        "losses/policy_loss": policy_gradient_losses,
        "losses/entropy": entropy_losses,
        "losses/old_approx_kl": old_approx_kullbackleibler,
        "losses/approx_kl": approx_kullbackleibler,
        "losses/clipfrac": mean_clipfracs,
        "losses/explained_variance": explained_variances,
        "charts/SPS": steps_per_seconds,
    }


def skvi_tune():
    raise NotImplementedError


def sakc_tune():
    raise NotImplementedError
