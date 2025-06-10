import os
import runpy
import sys
import time

import numpy as np

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from tensorboard.backend.event_processing import event_accumulator


# initialize Ray
ray.init(configure_logging=False)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def evaluate(config):
    # Algorithm command
    algo_command = [f"--{key}={value}" for key, value in config.items()]

    # Connect the runtime commands
    sys.argv = algo_command + [f"--env-id={config.env_id}", f"--seed={config.seed}", "--track=False"]
    with HiddenPrints():
        _experiment = runpy.run_path(path_name=config["script"], run_name="__main__")
    return _experiment



def objective(config):
    run = None

    experiment = evaluate(config)
    normalized_scores = []

    # Extract the metrics from Tensorboard
    ea = event_accumulator.EventAccumulator(f"runs/{experiment['run_name']}")
    ea.Reload()
    metric_values = [
        scalar_event.value for scalar_event in ea.Scalars(self.metric)[-self.metric_last_n_average_window :]
    ]
    # TODO: For debugging
    print (
        f"The average episodic return on {config["env_id"]} is {np.average(metric_values)} average over the last {config["metric_last_n_average_window"]} episodes."
    )
    if self.target_scores[env_id] is not None:
        normalized_scores += [
            (np.average(metric_values) - self.target_scores[env_id][0])
            / (self.target_score[env_id][1] - self.target_scores[env_id][0])
        ]
    else:
        normalized_scores += [np.average(metric_values)]
    if run:
        run.log({f"{env_id}_return": np.average(metric_values)})
    aggregated_normalized_score = np.median(normalized_scores)
    print(f"The median normalized score is {aggregated_normalized_score} with num_seeds={seed}")
    tune.report({"iterations": seed, "aggregated_normalized_score": aggregated_normalized_score})  # TODO: This should be the step


# Definition of the search space
search_space = {
    "learning-rate": tune.loguniform(0.0003, 0.003),
    "num-minibatches": tune.choice([1, 2, 4]),
    "update-epochs": tune.choice([1, 2, 4, 8]),
    "num-steps": tune.choice([5, 16, 32, 64, 128]),
    "vf-coef": tune.uniform(0, 5),
    "max-grad-norm": tune.uniform(0, 5),
    "total-timesteps": 100000,
    "num-envs": 16
}

# Define the search algorithm
algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)

# Set the number of samples to 1000
num_samples = 50

# Define the Tune trial & run it
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",  # TODO: Needs to be revised
        mode="max",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space,
)
results = tuner.fit()

# Print the best found hyperparameters of this initial trial
print("Best hyperparameters found were: ", results.get_best_result().config)


"""
Providing an initial set of hyperparameters to seed the search
"""

# Set of initial parameters to seed the algorithm
initial_params = [
    {"width": 1, "height": 2, "activation": "relu"},
    {"width": 4, "height": 2, "activation": "relu"},
]

# Seed the search algorithm with the initial parameters
searcher = OptunaSearch(points_to_evaluate=initial_params)
algo = ConcurrencyLimiter(searcher, max_concurrent=28)

# Prime Tune and run the trial
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space,
)
results = tuner.fit()

# Print the best results
print("Best hyperparameters found were: ", results.get_best_result().config)