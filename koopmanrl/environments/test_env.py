import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tap import Tap


class ArgumentParser(Tap):
    env_id: str = "LinearSystem-v0"  # Gym environment (default: LinearSystem-v0)
    seed: int = 123  # Seed for reproducibility (defaut: 123)


def main():
    args = ArgumentParser().parse_args()

    np.random.seed(args.seed)
    is_3d_env = False if args.env_id == "DoubleWell-v0" else True

    # Create the environment
    env = gym.make(args.env_id)
    env.observation_space.seed(args.seed)
    env.action_space.seed(args.seed)

    # Set up the figure and axis
    fig = plt.figure()
    if is_3d_env:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    # Initialize the plot elements
    (line,) = ax.plot([], [], lw=2)
    ax.set_xlim(env.state_minimums[0], env.state_maximums[0])
    ax.set_ylim(env.state_minimums[1], env.state_maximums[1])
    if is_3d_env:
        ax.set_zlim(env.state_minimums[2], env.state_maximums[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{args.env_id} System Trajectory")

    def init():
        line.set_data([], [])
        if is_3d_env:
            line.set_3d_properties([])
            return (line,)

    state = env.reset()
    states = [state]

    def animate(i):
        next_state, _, _, _ = env.step(np.array([0]))
        states.append(next_state)
        line.set_data(np.array(states)[:, 0], np.array(states)[:, 1])
        if is_3d_env:
            line.set_3d_properties(np.array(states)[:, 2])

    FuncAnimation(fig, animate, init_func=init, frames=2000, interval=1, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
