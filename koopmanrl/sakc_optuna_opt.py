import json
import os
import warnings

import numpy as np
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from tap import Tap

from koopmanrl.opt_wrappers import sakc_tuning_wrapper


# Read in the hyperparameter optimization instructions
class ArgumentParser(Tap):
    env_id: str = "LinearSystem-v0"  # Environment which is to be optimized
    num_samples: int = 50  # Number of samples permitted to the optimization
    max_concurrent: int = 4  # Maximum of concurrent sample evaluations
    num_envs: int = 1  # Number of environments to optimize over
    average_window: int = 5  # Number of time-steps the eval metrics is averaged over
    total_timesteps: int = 50000  # Number of total timesteps
    cpu_cores_per_trial: int = 28  # Number of CPU cores per trial
    storage_dir: str = "/home/lpaehler/Work/ReinforcementLearning/KoopmanRLLaptop/KoopmanRL/BayesianOptimization/koopman-rl/tuning"  # Directory into which to store the result
    output_file: str = (
        "sakc_linear_system_params"  # Name of the json with the best configuration
    )


def evaluate(config):
    _experiment = sakc_tuning_wrapper(
        seed=config["seed"],
        env_id=config["env-id"],
        v_lr=config["v-lr"],
        q_lr=config["q-lr"],
        number_of_paths=config["num-paths"],
        number_of_steps_per_path=config["num-steps-per-path"],
        state_order=config["state-order"],
        action_order=config["action-order"],
        total_timesteps=config["total-timesteps"],
    )
    return _experiment


def objective(config):
    _experiment = evaluate(config)

    # Extract the metrics from the experiment
    metric_values = [
        scalar_event[0]
        for scalar_event in _experiment[config["metric"]][
            -config["metric-last-n-average-window"] :
        ]
    ]
    if config["target-score"] is not None:
        normalized_score = (np.average(metric_values) - config["target-score"][0]) / (
            config["target-score"][1] - config["target-score"][0]
        )
    else:
        normalized_score = np.average(metric_values)
    tune.report(
        {
            "iterations": config["seed"],
            "normalized_score": normalized_score,
        }
    )


def main():
    # Ingest the command-line arguments
    args = ArgumentParser().parse_args()

    # Reduce the number of displayed error messages
    warnings.filterwarnings("ignore")

    # initialize Ray
    ray.init(configure_logging=False)

    # Definition of the search space
    search_space = {
        "env-id": args.env_id,
        "seed": tune.randint(0, 10000),
        "v-lr": tune.loguniform(0.0001, 0.1),
        "q-lr": tune.loguniform(0.0001, 0.1),
        "num-paths": tune.choice([50, 75, 100, 125, 150, 175, 200]),
        "num-steps-per-path": tune.choice(
            [75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        ),
        "state-order": tune.choice([1, 2, 3, 4]),
        "action-order": tune.choice([1, 2, 3, 4]),
        "total-timesteps": args.total_timesteps,
        "target-score": None,
        "num-envs": args.num_envs,
        "metric": "charts/episodic_return",
        "metric-last-n-average-window": args.average_window,
    }

    # Initialize the search algorithm
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=args.max_concurrent)

    # Define the number of CPU cores we assign to each trial
    trainable_with_resources = tune.with_resources(
        objective, {"cpu": args.cpu_cores_per_trial}
    )

    # Define the Tune trial & run it
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="normalized_score",
            mode="max",
            search_alg=algo,
            num_samples=args.num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    # Store the best results to a json file
    output_file_name = args.output_file + ".json"
    with open(os.path.join(args.storage_dir, output_file_name), "w") as file:
        json.dump(results.get_best_result().config, file, indent=4)

    # Print the best found hyperparameters of this initial trial
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    main()
