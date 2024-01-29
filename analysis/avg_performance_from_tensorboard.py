import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def create_folder(folder_path: str):
    """
    Create a folder at the specified path if it does not already exist.

    Parameters
    ----------
    folder_path : str
        The path at which the folder is to be created.

    Returns
    -------
    None

    Notes
    -----
    If the folder already exists, a message will be printed indicating that the folder is already present.

    Examples
    --------
    >>> create_folder('/path/to/new_folder')
    Folder '/path/to/new_folder' created.

    >>> create_folder('/path/to/existing_folder')
    Folder '/path/to/existing_folder' already exists.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

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
        "FluidFlow-v0__sac_continuous_action__1__1706390314",
        "FluidFlow-v0__sac_continuous_action_eval__1__1706392297"
    ]

    # For each tensorboard file in the pre-defined list above,
    for i, file_name in enumerate(file_names):
        # Collect the episodic returns and environment steps
        # and save the data into numpy files for use in other places
        episodic_returns, steps = collect_episodic_returns(path, file_name)
        create_folder(f'./analysis/{file_name}')
        np.save(f"./analysis/{file_name}/episodic_returns.npy", episodic_returns)
        np.save(f"./analysis/{file_name}/steps.npy", steps)

        # Example usage
        episodic_returns = np.load(f"./analysis/{file_name}/episodic_returns.npy")
        steps = np.load(f"./analysis/{file_name}/steps.npy")
        plt.plot(steps, episodic_returns)
        plt.show()