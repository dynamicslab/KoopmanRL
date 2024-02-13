import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import os
import pandas as pd

from analysis.utils import create_folder
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def collect_episodic_returns(
    tensorboard_file_directory: str,
    tensorboard_file_name: str
):
    """
    Collects episodic returns and corresponding environment steps from a TensorBoard file.

    Parameters:
    - tensorboard_file_directory (str): The directory containing the TensorBoard file.
    - tensorboard_file_name (str): The name of the TensorBoard file.

    Returns:
    Tuple[List[float], List[int]]: A tuple containing two lists -
    1. episodic_returns (List[float]): List of episodic return values.
    2. steps (List[int]): List of corresponding step values.

    Example:
    >>> directory = '/path/to/tensorboard/files'
    >>> file_name = 'experiment_log'
    >>> returns, steps = collect_episodic_returns(directory, file_name)
    """

    summary_iterator = EventAccumulator(os.path.join(tensorboard_file_directory, tensorboard_file_name)).Reload()

    scalar_name = 'charts/episodic_return'

    steps = [e.step for e in summary_iterator.Scalars(scalar_name)]
    episodic_returns = [e.value for e in summary_iterator.Scalars(scalar_name)]

    return episodic_returns, steps

if __name__ == '__main__':
    # Going to collect the episodic return data for the following files
    path = "./runs"
    file_names = [
        "FluidFlow-v0__interpretability_discrete_value_iteration__1__1707412846",
        "FluidFlow-v0__interpretability_discrete_value_iteration__1__1707413302",
        "FluidFlow-v0__interpretability_discrete_value_iteration__1__1707413512",
        "FluidFlow-v0__interpretability_discrete_value_iteration__1__1707413673",
    ]

    # Initialize Pandas dataframe
    df = pd.DataFrame()

    # For capturing file seeds
    file_seeds = []

    # For each tensorboard file in the pre-defined list above,
    for i, file_name in enumerate(file_names):
        # Collect seed number and environment id
        summary_iterator = EventAccumulator(os.path.join(path, file_name)).Reload()
        hyperparams = str(summary_iterator.Tensors('hyperparameters/text_summary')[0]).split('|')
        # file_seed = i+1 #! Hardcoded seed value for testing
        file_seed = hyperparams[hyperparams.index('seed')+1]
        file_seeds.append(f"Seed {file_seed}")
        env_id = hyperparams[hyperparams.index('env_id')+1]

        # Collect the episodic returns and environment steps
        # and save the data into numpy files for use in other places
        episodic_returns, steps = collect_episodic_returns(path, file_name)
        create_folder(f'./analysis/{file_name}')
        np.save(f"./analysis/{file_name}/episodic_returns.npy", episodic_returns)
        np.save(f"./analysis/{file_name}/steps.npy", steps)

        # Print information about the data
        print(f"{file_name}'s episodic returns length: {len(episodic_returns)}")
        print(f"{file_name}'s steps length: {len(steps)}")
        print(f"{file_name}'s steps per episode: {steps[0]+1}")

        # Compute mean episodic return
        print(f"{file_name}'s mean episodic return: {np.mean(episodic_returns)}\n")

        # Store information in dataframe
        try:
            # Try to write to all entries at once
            df.loc[steps, f"Seed {file_seed}"] = episodic_returns
        except:
            # We end up here if there is an error accessing the index above
            # Loop through the steps and returns and add them to the dataframe
            for step, episodic_return in zip(steps, episodic_returns):
                df.loc[step, f"Seed {file_seed}"] = episodic_return

    # Save dataframe to local CSV
    df.to_csv(f'./analysis/{env_id}_performance.csv')

    # Plot the path data from the dataframe
    plt.plot(df)
    plt.title("Episodic Returns For Various Runs")
    plt.xlabel("Step in Environment")
    plt.ylabel("Episodic Return")
    plt.legend(file_seeds)
    plt.show()

    # Plot the average and standard deviation with two sigma
    mean_episodic_returns = df.mean(axis=1)
    std_episodic_returns = df.std(axis=1)
    two_sigma_upper = mean_episodic_returns + 2 * std_episodic_returns
    two_sigma_lower = mean_episodic_returns - 2 * std_episodic_returns

    plt.plot(mean_episodic_returns, label='Mean Episodic Return')
    plt.fill_between(
        mean_episodic_returns.index,
        two_sigma_lower,
        two_sigma_upper,
        color='gray',
        alpha=0.3,
        label='Two Sigma'
    )
    plt.title("Average Episodic Returns with Two Sigma Bounds")
    plt.xlabel("Step in Environment")
    plt.ylabel("Episodic Return")
    plt.legend()
    plt.show()