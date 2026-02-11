# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "tensorboard",
#     "typed-argument-parser",
# ]
# ///

import json
import os
from typing import List, Tuple

from tap import Tap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class ArgumentParser(Tap):
    storage_dir: str = "/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/NewDataLogs/data"  # Directory to store the resulting data frame into # noqa: E501
    target_dir: str = "/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/NewDataLogs/data/Ablations/SAKC/runs"  # Directory which is to be walked, and data extracted from # noqa: E501
    system: str = "LinearSystem-v0"  # System for which the data frame is to be extracted. Options are: LinearSystem-v0, DoubleWell-v0, Lorenz-v0, and FluidFlow-v0. # noqa: E501
    mode: str = "SAKC_Ablations"  # Options include: SKVI_Ablations, SAKC_Ablations, and Episodic_Returns
    rl_algo: str = "value_based_sac_continuous_action"  # RL algorithm for which the data frame is to be extracted. Options are: value_based_sac_continuous_action, sac_continuous_action, and linear_quadratic_regulator # noqa: E501
    output_file: str = "test_frame.json"  # File name of the returned data frame


def tensorboard_extractor(tensorboard_file: str) -> Tuple[List[float], List[int]]:
    """
    Provided the path to a file, extracts the returns from said file and returns them as a <to be determined>
    """
    summary_iterator = EventAccumulator(tensorboard_file).Reload()

    scalar_name = "charts/episodic_return"

    print(tensorboard_file)

    # Extract the episodic returns and associated steps from the tensorboard file
    steps = [e.step for e in summary_iterator.Scalars(scalar_name)]
    episodic_returns = [e.value for e in summary_iterator.Scalars(scalar_name)]

    return episodic_returns, steps


def main() -> None:
    # Parse the input arguments
    args = ArgumentParser().parse_args()

    # TODO Add check with commensurate errors for misaligned settings

    # Dictionary to hold the data pre-JSON
    temp_dict = {}

    # Get the subfolders, and their respective names
    paths_of_subfolders = [f.path for f in os.scandir(args.target_dir) if f.is_dir()]

    for folder_name in paths_of_subfolders:
        for _root, _, _files in os.walk(folder_name):
            # Get the name of the subfolder
            _subfolder_name = folder_name.split("/")[-1]

            # Only collect the experiment for the chosen system
            if args.system == _subfolder_name.split("__")[0]:
                # Read in the steps, and episodic returns
                _episodic_returns, _steps = tensorboard_extractor(os.path.join(_root, _files[0]))

                # Get key experiment information from the naming scheme
                if args.mode == "SKVI_Ablations":
                    # Add Dict object to temporary dict
                    temp_dict[_subfolder_name] = {
                        "environment": _subfolder_name.split("__")[0],
                        "rl_algorithm": _subfolder_name.split("__")[1],
                        "seed": int(_subfolder_name.split("__")[4]),
                        "num_actions": int(_subfolder_name.split("__")[2]),
                        "num_training_epochs": int(_subfolder_name.split("__")[3]),
                        "episodic_returns": _episodic_returns,
                        "steps": _steps,
                        "time": int(_subfolder_name.split("__")[5]),
                    }
                elif args.mode == "SAKC_Ablations":
                    # Add dict object into the temporary dict
                    temp_dict[_subfolder_name] = {
                        "environment": _subfolder_name.split("__")[0],
                        "rl_algorithm": _subfolder_name.split("__")[1],
                        "seed": int(_subfolder_name.split("__")[2]),
                        "v_lr": float(_subfolder_name.split("__")[3]),
                        "q_lr": float(_subfolder_name.split("__")[4]),
                        "episodic_returns": _episodic_returns,
                        "steps": _steps,
                        "time": int(_subfolder_name.split("__")[5]),
                    }
                elif args.mode == "Episodic_Returns":
                    if args.rl_algo == _subfolder_name.split("__")[1]:
                        # Add dict object into the temporary dict
                        temp_dict[_subfolder_name] = {
                            "environment": _subfolder_name.split("__")[0],
                            "rl_algorithm": _subfolder_name.split("__")[1],
                            "seed": int(_subfolder_name.split("__")[2]),
                            "episodic_returns": _episodic_returns,
                            "steps": _steps,
                            "time": int(_subfolder_name.split("__")[3]),
                        }
                else:
                    raise ValueError("Mode not recognized")

    # Store the JSON object to the local file system
    with open(os.path.join(args.storage_dir, args.output_file), "w") as file:
        json.dump(temp_dict, file, indent=4)


if __name__ == "__main__":
    main()
