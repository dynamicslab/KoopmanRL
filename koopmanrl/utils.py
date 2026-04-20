import json
import os
from typing import Any

import gym


def load_and_apply_config(
    args,
    key_map: dict[str, str],
    fallbacks: dict[str, Any],
):
    """
    Load a JSON config file and fill in any ArgumentParser field that was not
    explicitly set on the CLI (i.e. still None).  Fields set on the CLI always
    take priority (CLI > config file > fallback default).

    Parameters
    ----------
    args:
        Parsed tap ``ArgumentParser`` instance.
    key_map:
        Maps JSON hyphenated keys to ``args`` attribute names.
    fallbacks:
        Default values applied to any attribute that is still ``None`` after
        the config file has been applied.
    """
    if args.config_file is not None:
        with open(args.config_file) as fh:
            cfg = json.load(fh)

        for json_key, attr in key_map.items():
            if getattr(args, attr) is None and json_key in cfg and cfg[json_key] is not None:
                setattr(args, attr, cfg[json_key])

        print(f"Loaded configuration from '{args.config_file}'")

    # Apply fallback defaults for any field still None (no config file or key absent).
    for attr, default in fallbacks.items():
        if getattr(args, attr) is None:
            setattr(args, attr, default)

    return args


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
