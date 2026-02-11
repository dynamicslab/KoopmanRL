import json
import logging
import os
import warnings

import numpy as np
from numpy.random import RandomState
from rliable import metrics
from rliable.library import StratifiedBootstrap
from tap import Tap

warnings.filterwarnings("default")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgumentParser(Tap):
    root_dir: str = "/home/lpaehler/Work/ReinforcementLearning/KoopmanRLLaptop/KoopmanRL/NewDataLogs/EpisodicReturns/LQR"  # The root directory of the dataframe holding the algorithm's performance # noqa: E501
    data_frame: str = "episodic_returns_lqr_double_well"  # The name of the JSON data frame
    output_dir: str = "/home/lpaehler/Work/ReinforcementLearning/KoopmanRLLaptop/KoopmanRL/NewDataLogs"  # The directory where the output .dat file will be saved # noqa: E501
    output_name: str = "output.dat"  # The name of the .dat output file for TikZ plotting
    smoothing_window: int = 1  # The number of episodes to smooth over with IQM
    confidence_band: float = 0.95  # Percent confidence band around the episodic return
    deterministic_bootstrap: bool = False  # Deterministic bootstrapping for reproducibility


def load_json_frame(frame_name):
    _path = os.path.join(os.getcwd(), f"{frame_name}.json")
    with open(_path, "r") as f:
        _dictionary = json.load(f)
    return _dictionary


def IQM(x):
    return metrics.aggregate_iqm(x)


def OG(x):
    return metrics.aggregate_optimality_gap(x)


def MEAN(x):
    return metrics.aggregate_mean(x)


def MEDIAN(x):
    return metrics.aggregate_median(x)


# Reduction function to reduce the last value of the RL traces down to the IQM
def IQM_reduction(x):
    # Presumes an input of the shape [[], ...]

    # Grab relevant subslices of dictionaries and reduce it down to the singular IQM
    _final_val = IQM([i[-args.smoothing_window :] for i in x])
    return IQM(_final_val)


# Helper function to compute the 95% confidence bands with percentile bootstrap & stratified sampling
def ComputeConfidenceBands(x, conf):
    raise NotImplementedError


if __name__ == "__main__":
    # Parse the command line arguments
    args = ArgumentParser().parse_args()

    # The answer to life, universe and everything
    rs = np.random.RandomState(42)

    # Change the directory to the root directory of the dataset to be processed
    os.chdir(args.root_dir)

    # Define the JSOn data frame to be procesed
    to_be_parsed_dict = load_json_frame(args.data_frame)

    # Accumulate all RL trajectories with the same hparams
    collected_episodic_returns = []
    plotting_timesteps = []

    # Actually begin iteration over the nested dictionary
    for k in to_be_parsed_dict.keys():
        # Load in the nested dictionary with its key
        _temp_dict = to_be_parsed_dict[k]

        # Append the episodic returns to the list
        collected_episodic_returns.append(_temp_dict["episodic_returns"])

        # Append the timesteps to the list
        plotting_timesteps.append(_temp_dict["steps"])

    # Convert both lists of lists into array to be able to process them with array ops
    episodic_returns_array = np.array(collected_episodic_returns, dtype=np.float64)
    timesteps_array = np.array(plotting_timesteps, dtype=np.float64)

    # Make sure that all the episodic returns expect the same plotting index
    unique_timesteps = np.unique(timesteps_array)

    # Calculate the IQM
    iqm_array = np.array(
        [IQM(episodic_returns_array[:, i]) for i in range(episodic_returns_array.shape[1])],
        dtype=np.float64,
    )

    # Calculate the 95% confidence band
    conf_list = []
    for i in range(episodic_returns_array.shape[1]):
        # Grab respective slice at that timestep
        _temp_array = episodic_returns_array[:, i]
        _temp_array = _temp_array.reshape(_temp_array.shape[0], 1)

        # Initialize the bootstrapping (this step doesn't work yet)
        _bs = StratifiedBootstrap(_temp_array)

        # Calculate the 95% confidence interval and append it to the prepared list
        conf_list.append(_bs.conf_int(IQM, method="percentile", reps=50000, size=args.confidence_band))

    # Convert confidence bounds into an array and reshape it into shape
    conf_array = np.array(conf_list, dtype=np.float64)
    conf_array = conf_array.reshape(conf_array.shape[0], -1)

    # Whip other arrays into shape
    unique_timesteps = unique_timesteps.reshape(unique_timesteps.shape[0], 1)
    iqm_array = iqm_array.reshape(iqm_array.shape[0], 1)

    # Assemble output dataframe
    dat_output = np.concatenate((unique_timesteps, iqm_array, conf_array), axis=1)

    # Store everything into a single output dataframe
    np.savetxt(
        os.path.join(args.output_dir, args.output_name),
        dat_output,
        fmt="%.4f",
        header="timesteps episodic_returns lower_confidence_bound upper_confidence_bound",
        delimiter=" ",
    )

    """
    Old code below this point.
    """

    """
    # TODO: Doesn't accept the smoothing window as an argument right now, to be added.
    for i in range(len(collected_episodic_returns[1])):
        # Temporary list to hold the entry for that timestep
        _iqm_timestep = []

        for j in range(len(collected_episodic_returns)):
            _iqm_timestep.append(collected_episodic_returns[j][i])

        # TODO: This is the spot where I could calculate the standard deviation, or whatever else I am going for.
        iqm_list.append(IQM(_iqm_timestep))

        # Target here: Pointwise 95% confidence bands based on percentile bootstrap with stratified sampling.
        # -> Do I have to import from rliable or is it best to just monkey-patch the functionality
        # -> Need to mark the areas in the draft where we need these intervals + add the above to the eval writeup.
        conf_list.append(
            ComputeConfidenceBands(_iqm_timestep, args.confidence_band)
        )  # TODO: this input here is most likely not correct.

    # Safety check that the length of the timesteps, and the IQM'd episodic returns are the same
    if len(iqm_list) != len(unique_timesteps[0]):
        raise ValueError(
            "The length of the IQM'd episodic returns and the timesteps are not the same!"
        )

    # Safety check that the length of the list of confidence bounds is correct
    if len(conf_list) != len(iqm_list):
        raise ValueError(
            "The length of the list of confidence bounds is not in accordance with the time series!"
        )
    """

    # Concat the arrays the goal is to end up with a shape of (len(iqm_list), 2)
    # dat_output = np.concatenate((X_output, Y_output), axis=1)
    # np.savetxt(
    #    os.path.join(args.output_dir, args.output_name),
    #    dat_output,
    #    fmt="%.4f",
    #    header="timesteps episodic_returns lower_confidence_bound upper_confidence_bound",
    #    delimiter=" ",
    # )
    # np.savetxt(
    #    os.path.join(args.output_dir, args.conf_output_name),
    #    Z_output,
    #    fmt="%.4f",
    #    header="lower_confidence_bound upper_confidence_bound",
    #    delimiter=" ",
    # )
