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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from cleanrl.ppo import make_env, Agent
from cleanrl.soft_koopman_value_iteration import (
    generate_koopman_tensor,
    DiscreteKoopmanValueIterationPolicy,
)
from cleanrl.soft_actor_koopman_critic import (
    Actor,
    SoftKoopmanVNetwork,
    SoftQNetwork,
    ReplayBuffer,
)


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
    Special version of CleanRL's PPO implementation just for the use with Ray Tune.

    The major differences are twofold:
    - No argparsing, instead inputs are direct to the function
    - No storage to Tensorboard files, instead the files live in a buffer.
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


def skvi_tuning_wrapper(
    seed=1,
    is_torch_deterministic=True,
    env_id="LinearSystem-v0",
    learning_rate=1e-3,
    gamma=0.99,
    alpha=1.0,
    number_of_actions=101,
    number_of_training_epochs=150,
    batch_size=2**14,
    batch_scale=1,
    total_timesteps=50000,
    number_of_paths=100,
    number_of_steps_per_path=300,
    state_order=2,
    action_order=2,
    regressor_type="ols",
):
    """
    Function-based version of the soft Koopman value iteration algorithm for hyperparameter tuning.

    Major differences to the single-file version include, but are not limited to:
    - No argparsing, instead inputs are direct to the function
    - No storage to Tensorboard files, instead the files live in a dictionary buffer.
    """

    run_name = f"skvi__{env_id}__{seed}__{learning_rate}__{number_of_paths}__{number_of_steps_per_path}__{state_order}__{action_order}"

    # Initialize the lists to store results into
    episodic_returns_list = []
    episodic_lengths_list = []
    steps_per_seconds_list = []

    # CPU-only execution
    device = torch.device("cpu")

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = is_torch_deterministic

    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, False, run_name)])

    # Koopman tensor generation logic
    koopman_tensor = generate_koopman_tensor(
        env_id=env_id,
        seed=seed,
        num_paths=number_of_paths,
        num_steps_per_path=number_of_steps_per_path,
        state_order=state_order,
        action_order=action_order,
        regressor=regressor_type,
    )

    try:
        dt = envs.envs[0].dt
    except:
        dt = None

    # Construct set of all possible actions
    all_actions = torch.from_numpy(
        np.linspace(
            start=envs.single_action_space.low,
            stop=envs.single_action_space.high,
            num=number_of_actions,
        )
    ).T

    # Construct value iteration policy
    value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
        env_id=env_id,
        gamma=gamma,
        alpha=alpha,
        dynamics_model=koopman_tensor,
        all_actions=all_actions,
        cost=envs.envs[0].vectorized_cost_fn,
        use_ols=True,
        learning_rate=learning_rate,
        dt=dt,
    )

    # Use Koopman tensor training data to train policy
    value_iteration_policy.train(
        number_of_training_epochs,
        batch_size,
        batch_scale,
        how_often_to_chkpt=10,
    )

    envs.single_observation_space.dtype = np.float64
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = value_iteration_policy.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, info = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for item in info:
            if "episode" in info.keys():
                episodic_returns_list.append([item["episode"]["r"], global_step])
                episodic_lengths_list.append([item["episode"]["l"], global_step])
                break

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Write data
        if global_step % 100 == 0:
            steps_per_seconds_list.append(
                [int(global_step / (time.time() - start_time)), global_step]
            )

    # Close the gym environment
    envs.close()

    # Return outputs for the hyperparameter optimization
    return {
        "charts/episodic_return": episodic_returns_list,
        "charts/episodic_length": episodic_lengths_list,
        "charts/SPS": steps_per_seconds_list,
    }


def sakc_tuning_wrapper(
    seed=1,
    is_torch_deterministic=True,
    to_capture_video=False,
    env_id="LinearSystem-v0",
    total_timesteps=50000,
    buffer_size=int(1e6),
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    learning_starts=5000,
    policy_lr=3e-4,
    v_lr=1e-3,
    q_lr=1e-3,
    policy_frequency=2,
    target_network_frequency=1,
    noise_clip=0.5,
    alpha=0.2,
    autotune=True,
    alpha_lr=1e-3,
    number_of_paths=100,
    number_of_steps_per_path=300,
    state_order=2,
    action_order=2,
    regressor_type="ols",
):
    """
    Function-based version of the soft Koopman actor-critic algorithm, built for hyperparameter tuning.

    Major differences to the single-file version include, but are not limited to:
    - No argparsing, instead inputs are direct to the function
    - No storage to Tensorboard files, instead the files live in a dictionary buffer.
    """

    run_name = f"sakc__{env_id}__{policy_lr}__{seed}__{v_lr}__{q_lr}__{number_of_paths}__{number_of_steps_per_path}__{state_order}__{action_order}"

    # Initialization of the lists to store results into
    episodic_returns_list = []
    episodic_lengths_list = []
    vf_values_list = []
    vf_loss_list = []
    qf1_a_values_list = []
    qf2_a_values_list = []
    qf1_loss_list = []
    qf2_loss_list = []
    qf_loss_list = []
    actor_loss_list = []
    alpha_list = []
    sps_list = []
    if autotune:
        alpha_loss_list = []

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = is_torch_deterministic

    # Running everything on CPU
    device = torch.device("cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, to_capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)

    # Koopman Logic
    koopman_tensor = generate_koopman_tensor(
        env_id=env_id,
        seed=seed,
        num_paths=number_of_paths,
        num_steps_per_path=number_of_steps_per_path,
        state_order=state_order,
        action_order=action_order,
        regressor=regressor_type,
    )
    vf = SoftKoopmanVNetwork(koopman_tensor).to(device)
    vf_target = SoftKoopmanVNetwork(koopman_tensor).to(device)

    # Usual Value function logic
    vf_target.load_state_dict(vf.state_dict())
    v_optimizer = optim.Adam(list(vf.parameters()), lr=v_lr)

    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

    # Automatic entropy tuning
    if autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=alpha_lr)
    else:
        alpha = alpha

    # envs.single_observation_space.dtype = np.float32
    envs.single_observation_space.dtype = np.float64
    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, log_probs, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for item in infos:
            if "episode" in infos.keys():
                episodic_returns_list.append([item["episode"]["r"], global_step])
                episodic_lengths_list.append([item["episode"]["l"], global_step])
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
        if global_step > learning_starts:
            # Sample from replay buffer
            data = rb.sample(batch_size)

            # E_s_t~D [ 1/2 ( V_psi( s_t ) - E_a_t~pi_phi [ Q_theta( s_t, a_t ) - log pi_phi( a_t | s_t ) ] )^2 ]
            vf_values = vf(data.observations).view(-1)
            with torch.no_grad():
                state_actions, state_log_pis, _ = actor.get_action(data.observations)
                q_values = torch.min(
                    qf1(data.observations, state_actions),
                    qf2(data.observations, state_actions),
                ).view(-1)
            vf_loss = F.mse_loss(vf_values, q_values - alpha * state_log_pis.view(-1))
            # vf_loss = F.l1_loss(vf_values, q_values - alpha * state_log_pis.view(-1))
            # Calculate L1 regularization term
            # with torch.no_grad():
            #     l1_regularization = torch.tensor(0., requires_grad=True)
            #     for param in vf.parameters():
            #         l1_regularization += torch.norm(param, p=1)
            # total_vf_loss = vf_loss + l1_regularization

            v_optimizer.zero_grad()
            vf_loss.backward()
            v_optimizer.step()

            # E_( s_t, a_t )~D [ 1/2 ( Q_theta( s_t, a_t ) - Q_target( s_t, a_t ) )^2 ]
            with torch.no_grad():
                expected_phi_x_primes = koopman_tensor.phi_f(
                    data.observations.T, data.actions.T
                ).T
                vf_next_target = (
                    (1 - data.dones.flatten())
                    * gamma
                    * vf_target.linear(expected_phi_x_primes).view(-1)
                )
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
            if global_step % policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % target_network_frequency == 0:
                for param, target_param in zip(vf.parameters(), vf_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            if global_step % 100 == 0:
                vf_values_list.append([vf_values.mean().item(), global_step])
                vf_loss_list.append([vf_loss.item(), global_step])
                qf1_a_values_list.append([qf1_a_values.mean().item(), global_step])
                qf2_a_values_list.append([qf2_a_values.mean().item(), global_step])
                qf1_loss_list.append([qf1_loss.item(), global_step])
                qf2_loss_list.append([qf2_loss.item(), global_step])
                qf_loss_list.append([qf_loss.item() / 2.0, global_step])
                actor_loss_list.append([actor_loss.item(), global_step])
                alpha_list.append([alpha, global_step])
                if autotune:
                    alpha_loss_list.append([alpha_loss.item(), global_step])
                sps_list.append(
                    [int(global_step / (time.time() - start_time)), global_step]
                )

    envs.close()

    # Get optimal value function weights from Koopman model
    value_function_weights = list(vf.parameters())
    target_value_function_weights = list(vf_target.parameters())

    # Assemble the to be returned dictionary
    if autotune:
        _dictionary = {
            "charts/episodic_return": episodic_returns_list,
            "charts/episodic_length": episodic_lengths_list,
            "losses/vf_values": vf_values_list,
            "losses/vf_loss": vf_loss_list,
            "losses/qf1_values": qf1_a_values_list,
            "losses/qf2_values": qf2_a_values_list,
            "losses/qf1_loss": qf1_loss_list,
            "losses/qf2_loss": qf2_loss_list,
            "losses/qf_loss": qf_loss_list,
            "losses/actor_loss": actor_loss_list,
            "losses/alpha": alpha_list,
            "losses/alpha_loss": alpha_loss_list,
            "charts/SPS": sps_list,
            "losses/value_function_weights": value_function_weights,
            "losses/target_value_function_weights": target_value_function_weights,
        }
    else:
        _dictionary = {
            "charts/episodic_return": episodic_returns_list,
            "charts/episodic_length": episodic_lengths_list,
            "losses/vf_values": vf_values_list,
            "losses/vf_loss": vf_loss_list,
            "losses/qf1_values": qf1_a_values_list,
            "losses/qf2_values": qf2_a_values_list,
            "losses/qf1_loss": qf1_loss_list,
            "losses/qf2_loss": qf2_loss_list,
            "losses/qf_loss": qf_loss_list,
            "losses/actor_loss": actor_loss_list,
            "losses/alpha": alpha_list,
            "charts/SPS": sps_list,
            "losses/value_function_weights": value_function_weights,
            "losses/target_value_function_weights": target_value_function_weights,
        }
    return _dictionary
