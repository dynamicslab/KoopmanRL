import time

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch


# initialize Ray
ray.init(configure_logging=False)

# Evaluation function
def evaluate(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1

# Objective function
def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"])
        tune.report({"iterations": step, "mean_loss": score})

# Construct the algorithm
algo = BayesOptSearch()  # TODO: What do these utility_kwargs do?
algo = ConcurrencyLimiter(algo, max_concurrent=28)

num_samples = 1000

# Define the search space
search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
}

# Run Ray Tune
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

# Print the best result
print("Best hyperparameters found were: ", results.get_best_result().config)