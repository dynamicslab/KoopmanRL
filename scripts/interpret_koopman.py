"""
Goal here is to point this at the trained files, either at the saved model or the Tensorboard files,
and return a printout of the polynomial representaiton of the Koopman operator.

Usage:
    python interpret_koopman.py --tensorboard_input <path/to/file>

--> Currently version isn't really functional. Need to revisit to at least have it functional for SKVI
"""

import json
import os

import torch
from tap import Tap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Read in the hyperparameter optimization instructions
class ArgumentParser(Tap):
    algorithm_type: str = "SKVI"  # Options are: SKVI, or SAKC
    environment: str = (
        "LinearSystem"  # Options are: LinearSystem, FluidFlow, Lorenz, DoubleWell
    )
    paths_to_files: str = "interpret_koopman.json"  # Single JSON file holding the paths to all the stored files
    # No storage routine to speak of at this time.
    storage_dir: str = "/home/lpaehler/Work/ReinforcementLearning/KoopmanRLLaptop/KoopmanRL/BayesianOptimization/koopman-rl"  # Directory into which to store the result
    output_file: str = (
        "koopman_polynomial"  # Name of the json with the best configuration
    )


def main():
    # Parse the input arguments
    args = ArgumentParser().parse_args()

    # Safety to protect against unimplemented algorithms
    match args.algorithm_type:
        case "SKVI" | "SAKC":
            pass
        case _:
            raise ValueError(
                "The given algorithm type is not implemented. Options are SAKC, and SKVI."
            )

    if args.algorithm_type == "SKVI":
        # TODO: Just load the `.pt` file, and that should in theory be the value tensor from which I can just read out the values
        value_weights = torch.load(args.skvi_stored_weights)

        # NOTE: Have the value weights, but it is not exactly clear of which order they are, need to check the order of the monomials
    elif args.algorithm_type == "SAKC":
        raise NotImplementedError(
            "The polynomial interpretation for Soft Actor Koopman Critic isn't implemented yet. To come soon!"
        )

    print(value_weights)


if __name__ == "__main__":
    main()

"""
value_function_weights = torch.tensor([
        [-333.7974], # 1
        [  22.5883], # x
        [  -8.0066], # y
        [-157.5718], # z
        [ 267.9301], # x*x
        [ -80.5217], # x*y
        [ -27.1598], # x*z
        [ 173.2158], # y*y
        [   6.2852], # y*z
        [ 149.4211]  # z*z
    ])
    value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
        args=args,
        gamma=args.gamma,
        alpha=args.alpha,
        dynamics_model=koopman_tensor,
        all_actions=all_actions,
        cost=envs.envs[0].vectorized_cost_fn,
        use_ols=True,
        learning_rate=args.lr,
        dt=dt,
    )
    value_iteration_policy.load_model(
        value_function_weights=value_function_weights,
        trained_model_start_timestamp=None,
        chkpt_epoch_number=None
    )
"""
