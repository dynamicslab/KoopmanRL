import numpy as np
import torch

from cleanrl.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy
from cleanrl.linear_quadratic_regulator import LQRPolicy
from cleanrl.sac_continuous_action import Actor
from koopman_tensor.utils import load_tensor
from movies import Policy


class LQR(Policy):
    def __init__(self, args, envs):
        # Recover dt from environment
        try:
            dt = envs.envs[0].dt
        except:
            dt = None

        # Construct LQR policy
        discrete_systems = ('LinearSystem-v0')
        is_continuous = False if args.env_id in discrete_systems else True
        try:
            self.policy = LQRPolicy(
                A=envs.envs[0].continuous_A,
                B=envs.envs[0].continuous_B,
                Q=envs.envs[0].Q,
                R=envs.envs[0].R,
                reference_point=envs.envs[0].reference_point,
                gamma=args.gamma,
                alpha=args.alpha,
                dt=dt,
                is_continuous=is_continuous,
                seed=args.seed
            )
        except:
            self.policy = LQRPolicy(
                A=envs.envs[0].A,
                B=envs.envs[0].B,
                Q=envs.envs[0].Q,
                R=envs.envs[0].R,
                reference_point=envs.envs[0].reference_point,
                gamma=args.gamma,
                alpha=args.alpha,
                dt=dt,
                is_continuous=is_continuous,
                seed=args.seed
            )

    def get_action(self, state, *args, **kwargs):
        state_col_vec = state.reshape(-1, 1)
        return self.policy.get_action(state_col_vec, **kwargs)


class SKVI(Policy):
    def __init__(self, args, envs, saved_koopman_model_name, path_to_checkpoint, device):
        # Define device
        self.device = device

        # Load saved Koopman tensor
        self.koopman_tensor = load_tensor(env_id=args.env_id, saved_model_name=saved_koopman_model_name)

        # Construct set of all possible actions
        all_actions = torch.from_numpy(np.linspace(
            start=envs.single_action_space.low,
            stop=envs.single_action_space.high,
            num=args.num_actions
        )).T

        # Get dt from the environment
        try:
            dt = envs.envs[0].dt
        except:
            dt = None

        # Load SKVI policy
        self.policy = DiscreteKoopmanValueIterationPolicy(
            env_id=args.env_id,
            gamma=args.gamma,
            alpha=args.alpha,
            dynamics_model=self.koopman_tensor,
            all_actions=all_actions,
            cost=envs.envs[0].vectorized_cost_fn,
            save_data_path=path_to_checkpoint,
            seed=args.seed,
            load_model=True,
            args=args,
            dt=dt
        )

    def get_action(self, state, *args, **kwargs):
        state_row_vec = state.reshape(1, -1)
        actions = self.policy.get_action(torch.tensor(state_row_vec).to(self.device))
        actions = actions.detach().cpu().numpy()
        return actions


class SAKC(Policy):
    def __init__(self, envs, path_to_checkpoint, device):
        self.device = device
        self.policy = Actor(envs).to(device)
        self.policy.load_state_dict(path_to_checkpoint)

    def get_action(self, state, *args, **kwargs):
        actions, _, _ = self.policy.get_action(torch.tensor(state).to(self.device))
        actions = actions.detach().cpu().numpy()
        return actions