import os

import gym


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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
