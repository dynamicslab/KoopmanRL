import numpy as np
import warnings

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from opt_wrappers import sakc_tuning_wrapper


def evaluate(config):
    _experiment = sakc_tuning_wrapper(
        seed=config["seed"],
        env_id=config["env-id"],
        gamma=config["gamma"],
        tau=config["tau"],
        batch_size=config["batch-size"],
        policy_lr=config["policy-lr"],
        v_lr=config["v-lr"],
        q_lr=config["q-lr"],
        alpha=config["alpha"],
        alpha_lr=config["alpha-lr"],
        number_of_paths=config["num-paths"],
        number_of_steps_per_path=config["num-steps-per-path"],
        state_order=config["state-order"],
        action_order=config["action-order"],
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


if __name__ == "__main__":
    # Reduce the number of displayed error messages
    warnings.filterwarnings("ignore")

    # initialize Ray
    ray.init(configure_logging=False)

    # Definition of the search space
    search_space = {  # TODO: Values need to be dialed in
        "env-id": "CartPole-v1",
        "seed": tune.randint(0, 10000),
        "gamma": tune.loguniform(0.0003, 0.003),
        "tau": tune.choice([1, 2, 4]),
        "batch-size": tune.choice([1, 2, 4, 8]),
        "policy-lr": tune.loguniform(0.0003, 0.003),
        "v-lr": tune.loguniform(0.0003, 0.003),
        "q-lr": tune.loguniform(0.0003, 0.003),
        "alpha": tune.uniform(0, 1),
        "alpha-lr": tune.loguniform(0.0003, 0.003),
        "num-paths": tune.choice([1, 2, 4, 8]),
        "num-steps-per-path": tune.choice([1, 2, 4, 8]),
        "state-order": tune.choice([1, 2, 4, 8]),
        "action-order": tune.choice([1, 2, 4, 8]),
        "target-score": [0, 500],
        "total-timesteps": 100000,
        "num-envs": 16,
        "metric": "charts/episodic_return",
        "metric-last-n-average-window": 50,
    }

    # Initialize the search algorithm
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    num_samples = 50

    # Define the Tune trial & run it
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="normalized_score",
            mode="max",
            search_alg=algo,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    # Print the best found hyperparameters of this initial trial
    print("Best hyperparameters found were: ", results.get_best_result().config)
