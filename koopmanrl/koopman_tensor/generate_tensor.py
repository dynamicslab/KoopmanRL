import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from tap import Tap

from koopmanrl.koopman_tensor.observables import torch_observables as observables
from koopmanrl.koopman_tensor.torch_tensor import KoopmanTensor, Regressor
from koopmanrl.koopman_tensor.utils import save_tensor

torch.set_default_dtype(torch.float64)


class ArgumentParser(Tap):
    env_id: str = "LinearSystem-v0"  # Gym environment (default: LinearSystem-v0)
    num_paths: int = 100  # Number of paths for the dataset (default: 100)
    num_steps_per_path: int = 300  # Number of steps per path for the dataset (default: 300)
    state_order: int = 2  # Order of monomials used for the state dictionary (default: 2)
    action_order: int = 2  # Order of monomials used for the action dictionary (default: 2)
    seed: int = 123  # Seed for reproducibility (default: 123)
    save_model: bool = False  # Whether to store the Koopman tensor model in a pickle file (default: False)
    animate: bool = False  # Whether to show the animated dynamics over time (default: False)
    regressor: str = "ols"  # Which regressor to use to build the Koopman tensor (default: 'ols')


def main():
    args = ArgumentParser().parse_args()

    """ Set seeds and create environment """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    is_3d_env = False if args.env_id == "DoubleWell-v0" else True
    env = gym.make(args.env_id)
    env.observation_space.seed(args.seed)
    env.action_space.seed(args.seed)

    """ Collect data """
    state_dim = env.observation_space.shape
    state_dim = 1 if len(state_dim) == 0 else state_dim[0]
    action_dim = env.action_space.shape
    action_dim = 1 if len(action_dim) == 0 else action_dim[0]
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}\n")

    # Path-based data collection
    # i.e. generate a bunch of independent paths
    # and train the Koopman tensor on those transitions
    X = torch.zeros((args.num_paths, args.num_steps_per_path, state_dim))
    Y = torch.zeros_like(X)
    U = torch.zeros((args.num_paths, args.num_steps_per_path, action_dim))

    for path_num in range(args.num_paths):
        state = env.reset()
        # state = env.reset(seed=args.seed)
        for step_num in range(args.num_steps_per_path):
            X[path_num, step_num] = torch.tensor(state)
            action = env.action_space.sample()
            U[path_num, step_num] = torch.tensor(action)
            state, _, _, _ = env.step(action)
            Y[path_num, step_num] = torch.tensor(state)

    """ Make sure trajectories look ok """
    if args.animate:
        # Create a figure and 3D axis
        fig = plt.figure()
        if is_3d_env:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

        # Set limits for each axis
        ax.set_xlim(X[0:, :, 0].min(), X[0:, :, 0].max())
        ax.set_ylim(X[0:, :, 1].min(), X[0:, :, 1].max())
        if is_3d_env:
            ax.set_zlim(X[0:, :, 2].min(), X[0:, :, 2].max())

        # Initialize an empty line for the animation
        if is_3d_env:
            (line,) = ax.plot([], [], [], lw=2)
        else:
            (line,) = ax.plot([], [], lw=2)

        # Function to initialize the plot
        def init():
            line.set_data([], [])
            if is_3d_env:
                line.set_3d_properties([])
                return (line,)

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
            return (line,)

        # Create the animation
        ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()

    """ Reshape data so that we have matrices of data instead of tensor """
    total_num_datapoints = args.num_paths * args.num_steps_per_path
    X = X.reshape(total_num_datapoints, state_dim).T
    Y = Y.reshape(total_num_datapoints, state_dim).T
    U = U.reshape(total_num_datapoints, action_dim).T

    """ Construct Koopman tensor """
    try:
        path_based_tensor = KoopmanTensor(
            X,
            Y,
            U,
            phi=observables.monomials(args.state_order),
            psi=observables.monomials(args.action_order),
            regressor=Regressor(args.regressor),
            dt=env.dt,
        )
    except Exception:
        # Assume the error was because there is no dt for LinearSystem
        path_based_tensor = KoopmanTensor(
            X,
            Y,
            U,
            phi=observables.monomials(args.state_order),
            psi=observables.monomials(args.action_order),
            regressor=Regressor(args.regressor),
        )

    """ Predict sample points """
    sample_indices = (0, X.shape[1])
    sample_x = X[:, sample_indices[0] : sample_indices[1]]
    sample_u = U[:, sample_indices[0] : sample_indices[1]]
    true_x_prime = Y[:, sample_indices[0] : sample_indices[1]]
    estimated_x_prime = path_based_tensor.f(sample_x, sample_u)

    single_step_state_estimation_error_norms = np.linalg.norm(true_x_prime - estimated_x_prime, axis=0)
    avg_single_step_state_estimation_error_norm = single_step_state_estimation_error_norms.mean()
    max_single_step_state_estimation_error_norm = single_step_state_estimation_error_norms.max()
    avg_state_norm = np.linalg.norm(X.mean(axis=1))
    avg_single_step_state_estimation_error_norm_per_avg_state_norm = (
        avg_single_step_state_estimation_error_norm / avg_state_norm
    )
    max_single_step_state_estimation_error_norm_per_avg_state_norm = (
        max_single_step_state_estimation_error_norm / avg_state_norm
    )
    print(
        f"Average single step state estimation error norm per average state norm: {avg_single_step_state_estimation_error_norm_per_avg_state_norm}"  # noqa: E501
    )
    print(
        f"Max single step state estimation error norm per average state norm: {max_single_step_state_estimation_error_norm_per_avg_state_norm}"  # noqa: E501
    )

    true_phi_x_prime = path_based_tensor.Phi_Y[:, sample_indices[0] : sample_indices[1]]
    estimated_phi_x_prime = path_based_tensor.phi_f(sample_x, sample_u)

    single_step_phi_estimation_error_norms = np.linalg.norm(true_phi_x_prime - estimated_phi_x_prime, axis=0)
    avg_single_step_phi_estimation_error_norm = single_step_phi_estimation_error_norms.mean()
    max_single_step_phi_estimation_error_norm = single_step_phi_estimation_error_norms.max()
    avg_phi_norm = np.linalg.norm(path_based_tensor.Phi_X.mean(axis=1))
    avg_single_step_phi_estimation_error_norm_per_avg_phi_norm = (
        avg_single_step_phi_estimation_error_norm / avg_phi_norm
    )
    max_single_step_phi_estimation_error_norm_per_avg_phi_norm = (
        max_single_step_phi_estimation_error_norm / avg_phi_norm
    )
    print(
        f"Average single step phi estimation error norm per average phi norm: {avg_single_step_phi_estimation_error_norm_per_avg_phi_norm}"  # noqa: E501
    )
    print(
        f"Max single step phi estimation error norm per average phi norm: {max_single_step_phi_estimation_error_norm_per_avg_phi_norm}"  # noqa: E501
    )

    """ Save Koopman tensor """
    if args.save_model:
        save_tensor(path_based_tensor, args.env_id, "path_based_tensor")


if __name__ == "__main__":
    main()
