import numpy as np
import random
import torch

class Generator:
    def __init__(self, env, policy, seed=1234):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = env
        self.policy = policy

    def generate_trajectories(self, num_trajectories, num_steps_per_trajectory=None):
        # Store trajectories in an array
        trajectories = []

        # Loop through number of trajectories
        for trajectory_num in range(num_trajectories):
            # Create new trajectory and reset environment
            trajectory = []
            state, _ = self.env.reset()
            done = False

            # Set up our loop condition
            # Using lambda functions so the boolean value is not hardcoded and can be recomputed
            if num_steps_per_trajectory is not None:
                step_num = 0
                loop_condition = lambda: step_num < num_steps_per_trajectory
            else:
                loop_condition = lambda: done is True

            # Loop through trajectory until condition is met
            while loop_condition() is False:
                # Append new state to trajectory
                trajectory.append(state)

                # Get action from generic policy and get new state
                action = self.policy.get_action(state)
                new_state, _, done, _ = self.env.step(action)

                # Update state
                state = new_state

            # Append trajectory to list of trajectories
            trajectories.append(trajectory)

        # Cast trajectories into numpy array
        trajectories = np.array(trajectories)

        return trajectories