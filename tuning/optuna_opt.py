import os
import sys

import numpy as np

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from opt_wrappers import ppo_tuning_wrapper

# initialize Ray
ray.init(configure_logging=False)


def evaluate(config):
    _experiment = ppo_tuning_wrapper(
        seed=config["seed"],
        env_id=config["env-id"],
        learning_rate=config["learning-rate"],
        num_minibatches=config["num-minibatches"],
        update_epochs=config["update-epochs"],
        number_of_steps=config["num-steps"],
        vfunc_coefficient=config["vf-coef"],
        max_gradient_norm=config["max-grad-norm"],
        total_timesteps=config["total-timesteps"]
    )
    return _experiment


def objective(config):

    experiment = evaluate(config)
    normalized_scores = []

    # Extract the metrics from the experiment
    # TODO: The extraction logic needs to be redone
    metric_values = [
        scalar_event.value
        for scalar_event in ea.Scalars(config["metric"])[
            -config["metric-last-n-average-window"] :
        ]
    ]
    print(
        f"The average episodic return on {config['env-id']} is {np.average(metric_values)} average over the last {config['metric-last-n-average-window']} episodes."
    )
    if config["target-score"] is not None:
        normalized_scores += [
            (np.average(metric_values) - config["target_score"][0])
            / (config["target_score"][1] - config["target_scores"][0])
        ]
    else:
        normalized_scores += [np.average(metric_values)]
    aggregated_normalized_score = np.median(normalized_scores)
    print(
        f"The median normalized score is {aggregated_normalized_score} with num_seeds={config['seed']}"
    )
    tune.report(
        {
            "iterations": config["seed"],
            "aggregated_normalized_score": aggregated_normalized_score,
        }
    )


# Definition of the search space
search_space = {
    "env-id": "CartPole-v1",
    "seed": tune.randint(0, 10000),
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

# Initialize the search algorithm
algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
num_samples = 50

# Define the Tune trial & run it
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="aggregated_normalized_score",
        mode="max",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space,
)
results = tuner.fit()

# Print the best found hyperparameters of this initial trial
print("Best hyperparameters found were: ", results.get_best_result().config)
