import time

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch


# initialize Ray
ray.init(configure_logging=False)

# Define the evaluation function which we are targeting to optimize
def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 0
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost

# Define the objective function
def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        tune.report({"iterations": step, "mean_loss": score})

# Definition of the search space
search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"]),
}

# Define the search algorithm
algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=28)

# Set the number of samples to 1000
num_samples = 1000

# Define the Tune trial & run it
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