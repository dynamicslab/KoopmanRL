import os
import runpy
import sys

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
    # Assemble the algorithm command
    algo_command = [f"--{key}={value}" for key, value in config.items()]

    # Connect the runtime commands & run it
    sys.argv = algo_command + [f"--env-id={config["env-id"]}", f"--seed={config["seed"]}", "--track=False"]
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
        scalar_event.value for scalar_event in ea.Scalars(config["metric"])[-config["metric-last-n-average-window"] :]
    ]
    print (
        f"The average episodic return on {config["env-id"]} is {np.average(metric_values)} average over the last {config["metric-last-n-average-window"]} episodes."
    )
    if config["target-score"] is not None:
        normalized_scores += [
            (np.average(metric_values) - config["target_score"][0])
            / (config["target_score"][1] - config["target_scores"][0])
        ]
    else:
        normalized_scores += [np.average(metric_values)]
    if run:
        run.log({f"{config["env-id"]}_return": np.average(metric_values)})
    aggregated_normalized_score = np.median(normalized_scores)
    print(f"The median normalized score is {aggregated_normalized_score} with num_seeds={config["seed"]}")
    tune.report({"iterations": config["seed"], "aggregated_normalized_score": aggregated_normalized_score})


# Definition of the search space
search_space = {
    "env-id": "CartPole-v1",
    "seed": tune.randint(0, 10000),
    "script": "cleanrl/ppo.py",
    "learning-rate": tune.loguniform(0.0003, 0.003),
    "num-minibatches": tune.choice([1, 2, 4]),
    "update-epochs": tune.choice([1, 2, 4, 8]),
    "num-steps": tune.choice([5, 16, 32, 64, 128]),
    "vf-coef": tune.uniform(0, 5),
    "max-grad-norm": tune.uniform(0, 5),
    "target-score": [0, 500],
    "total-timesteps": 100000,
    "num-envs": 16,
    "metric": "charts/episodic_return",
    "metric-list-n-average-window": 50,
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

# TODO: Just needs to be debugged now.
