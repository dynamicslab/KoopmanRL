import json
import logging
import os
import warnings

import numpy as np
from rliable import metrics
from tap import Tap

warnings.filterwarnings("default")

# The answer to life, universe and everything
RAND_STATE = np.random.RandomState(42)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgumentParser(Tap):
    root_dir: str = "/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/NewDataLogs/SKVI_Ablations"  # The root directory from where the dataset is to be processed # noqa: E501
    data_frame: str = "skvi_ablation_double_well"  # The name of the JSON data frame
    output_dir: str = "/Users/lpaehler/Work/ReinforcementLearning/KoopmanRL/NewDataLogs"  # The directory where the output .dat file will be saved # noqa: E501
    output_name: str = "output.dat"  # The name of the .dat output file for TikZ plotting
    smoothing_window: int = 1  # The number of episodes to smooth over with IQM


def load_json_frame(frame_name):
    _path = os.path.join(os.getcwd(), f"{frame_name}.json")
    with open(_path, "r") as f:
        _dictionary = json.load(f)
    return _dictionary


def IQM(x):
    return metrics.aggregate_iqm(x)


# Reduction function to reduce the last value of the RL traces down to the IQM
def IQM_reduction(x):
    # Presumes an input of the shape [[], ...]

    # Grab relevant subslices of dictionaries and reduce it down to the singular IQM
    _final_val = IQM([i[-args.smoothing_window :] for i in x])
    return IQM(_final_val)


if __name__ == "__main__":
    # Parse the command line arguments
    args = ArgumentParser().parse_args()

    # Change the directory to the root directory of the dataset to be processed
    os.chdir(args.root_dir)

    # Define the JSOn data frame to be procesed
    to_be_parsed_dict = load_json_frame(args.data_frame)

    # List of episodic returns to collect the individual episodic returns into
    X = []  # Directly write into the three lists X, Y, and Z
    Y = []
    Z = []

    # Iterate over the unique keys of `num_actions`
    for i in [71, 81, 91, 101, 111, 121]:
        # Iterate over the unique keys of `num_training_epochs`
        for j in [75, 100, 125, 150, 175, 200]:
            # Accumulate all RL trajectories with the same hparams
            temp_list = []

            # Actually begin iteration over the nested dictionary
            for k in to_be_parsed_dict.keys():
                _temp_dict = to_be_parsed_dict[k]

                if _temp_dict["num_actions"] == i and _temp_dict["num_training_epochs"] == j:
                    temp_list.append(_temp_dict["episodic_returns"])

            # Construct X, Y, and Z vectors for the surface plot
            X.append(i)
            Y.append(j)

            # IQM reduction op, and then append a vector with `[i, j, iqm]` to the above list
            Z.append(IQM_reduction(temp_list))

    # Convert the list into NumPy arrays
    X_output = np.array(X, dtype=np.float64).reshape(36, 1)
    Y_output = np.array(Y, dtype=np.float64).reshape(36, 1)
    Z_output = np.array(Z, dtype=np.float64).reshape(36, 1)

    # Concat the arrays the goal is to end up with a shape of (36, 3)
    dat_output = np.concatenate((X_output, Y_output, Z_output), axis=1)

    np.savetxt(
        os.path.join(args.output_dir, args.output_name),
        dat_output,
        fmt="%.4f",
        header="x y z",
        delimiter=" ",
    )
