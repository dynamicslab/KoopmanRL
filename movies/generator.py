import numpy as np
import random
import torch

from movies.env_enum import EnvEnum

class Generator:
    def __init__(self, args, envs, policy):
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.envs = envs
        self.env_id = args.env_id
        self.policy = policy

    def generate_trajectories(self, num_trajectories, num_steps_per_trajectory=None):
        print(f"Generating {num_trajectories} {'trajectory' if num_trajectories == 1 else 'trajectories'}...")

        # Store trajectories, actions, and costs in lists
        trajectories = []
        actions = []
        costs = []

        # Loop through number of trajectories
        for _ in range(num_trajectories):
            # Create new trajectory lists
            trajectory = []
            actions_per_trajectory = []
            costs_per_trajectory = []

            # Reset environment and get initial state
            state = self.envs.reset()

            # Get initial action and cost for initial state
            action = np.array([[0]])  # Initial action is no action
            cost = self.envs.envs[0].cost_fn(state, action[0])[0, 0]  # Cost of initial state with no action

            # Add initial state to trajectory
            if self.env_id == EnvEnum.DoubleWell:
                # Get potential of initial state if DoubleWell
                potential = self.envs.envs[0].potential()

                # Add initial state to trajectory
                print(np.concatenate((state[0], [potential]), axis=0).shape)
                trajectory.append(
                    np.concatenate((state[0], [potential]), axis=0)
                )
            else:
                # Add initial state to trajectory
                trajectory.append(state[0])

            # Add initial action to trajectory
            actions_per_trajectory.append(action[0])
            # Add initial cost to trajectory
            costs_per_trajectory.append(cost)
            dones = [False]  # Keep track of done boolean values

            # Set up our loop condition
            # Using a function so the boolean value is not hardcoded and can be recomputed
            step_num = 1  # Start at 1 because we already added the initial state
            def check_loop_condition():
                if num_steps_per_trajectory is not None:
                    return step_num < num_steps_per_trajectory
                else:
                    # If the episode is over, return True
                    if step_num >= self.envs.envs[0].max_episode_steps:
                        return True

                    for done in dones:
                        if done is True:
                            return True

                    # Otherwise, return False
                    return False

            # Loop through trajectory until condition is met
            while check_loop_condition() is False:
                # Get action from policy and get new state
                action = self.policy.get_action(state)  # Expected to be shape of (1, 1)
                new_state, reward, dones, _ = self.envs.step(action)

                # Print progress
                if (step_num+1) % 100 == 0:
                    print(f"Finished generating step {step_num+1}")

                # Update state
                state = new_state
                cost = -reward[0]
                if self.env_id == EnvEnum.DoubleWell:
                    potential = self.envs.envs[0].potential(U=action[0, 0])
                step_num += 1

                # Append new state, action, and cost to respective local lists
                if self.env_id == EnvEnum.DoubleWell:
                    print(np.concatenate((state[0], [potential]), axis=0).shape)
                    trajectory.append(
                        np.concatenate((state[0], [potential]), axis=0)
                    )
                else:
                    trajectory.append(state[0])
                actions_per_trajectory.append(action[0])
                costs_per_trajectory.append(cost)

            # Append trajectory, actions, and costs to respective global lists
            trajectories.append(trajectory)
            actions.append(actions_per_trajectory)
            costs.append(costs_per_trajectory)

        # Cast lists into numpy arrays
        trajectories = np.array(trajectories)
        actions = np.array(actions)
        costs = np.array(costs)

        # Print success message
        print(f"Finished generating {num_trajectories} {'trajectory' if num_trajectories == 1 else 'trajectories'}!")

        return trajectories, actions, costs