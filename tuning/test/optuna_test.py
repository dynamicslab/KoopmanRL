import time
from typing import Dict, Optional, Any

import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

ray.init(configure_logging=False)  # initialize Ray


#def evaluate(step, width, height, activation):
#    time.sleep(0.1)
#    activation_boost = 10 if activation=="relu" else 0
#    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


#def objective(config):
#    for step in range(config["steps"]):
#        score = evaluate(step, config["width"], config["height"], config["activation"])
#        tune.report({"iterations": step, "mean_loss": score})


#search_space = {
#    "steps": 100,
#    "width": tune.uniform(0, 20),
#    "height": tune.uniform(-100, 100),
#    "activation": tune.choice(["relu", "tanh"]),
#}

algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
num_samples = 1000

#tuner = tune.Tuner(
#    objective,
#    tune_config=tune.TuneConfig(
#        metric="mean_loss",
#        mode="min",
#        search_alg=algo,
#        num_samples=num_samples,
#    ),
#    param_space=search_space,
#)
#results = tuner.fit()