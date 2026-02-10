<h1 align='center'>Koopman-Assisted Reinforcement Learning</h1>

## About

KoopmanRL is a reinforcement learning (RL) package designed around the two Koopman-Assisted RL (KARL) algorithms, Soft Koopman Value Iteration, and Soft Koopman Actor-Critic. It provides the utilities to build upon parts of its algorithms by either using only the Koopman tensor itself, or only components of the two KARL algorithms. In addition it provides utilities for automatic hyperparameter tuning of KARL algorithms, as well as 4 environments rooted in the control literature.

## Getting Started

Having [uv](https://docs.astral.sh/uv/) installed, one can easily get the project's environment up-and-running by syncing their local project environment with the lock file. Beginning after the successful clone of the project:

```bash
cd koopman-rl && uv sync
```

At which point we have the local project environment we need to run the computations. Alternatively, one can create a dedicated [virtual environment](https://docs.astral.sh/uv/pip/environments/) with uv, i.e.:

```bash
uv venv --python 3.10 && uv venv
```

After which one can install the KoopmanRL package into the virtual environment

```bash
uv pip install .
```

At which point one can run the first experiment, taking the (fast) Linear Quadratic Regulator as an illustrative example here:

```bash
uv run -m koopmanrl.linear_quadratic_regulator
```

Which will default to the linear system with the default hyperparameters. For all algorithms, and the hyperparameter optimization routines query the available hyperparameters with the `--help` identifier after the routine at which point all options will be listed in the terminal.

```bash
uv run -m koopmanrl.linear_quadratic_regulator --help
```

## Running Individual Experiments

KoopmanRL provides a number of algorithm implementations, which can all be run as quasi-scripts with their own (typed) argument parsers. The main files here are:

#### Control Algorithms:
* `koopmanrl.linear_quadratic_regulator`
* `koopmanrl.sac_continuous_action`
* `koopmanrl.value_based_sac_continuous_action`
* `koopmanrl.soft_koopman_value_iteration`
* `koopmanrl.soft_actor_koopman_critic`

#### Hyperparameter Optimization:
* `koopmanrl.skvi_optuna_opt`
* `koopmanrl.sakc_optuna_opt`

Each of the algorithms, be it a control algorithm, or one of the two hyperparameter optimization routines can be applied to one of the four environments. The implemented environments are:
1. LinearSystem-v0
2. FluidFlow-v0
3. Lorenz-v0
4. DoubleWell-v0

The algorithms can be applied to any of the environments with the `--env_id` flag. Utilizing the example of the Linear Quadratic Regulator again:

```bash
uv run -m koopmanrl.linear_quadratic_regulator --env_id FluidFlow-v0
```

> [!NOTE]
>Note that the hyperparameter optimizations need to be started with the python executable of the environments. E.g. with a uv environment:
>
>```bash
>uv run python -m koopmanrl.sakc_optuna_opt --env_id FluidFlow-v0
>```
>
>or when working with an activated virtual environment
>
>```bash
>python -m koopmanrl.sakc_optuna_opt --env_id FluidFlow-v0
>```

### Running Scripts

We provide a number of helper scripts to either reproduce the results, or illustrations from the paper as well as running the hyperparameter optimizations. These can be run with:

```bash
uv run scripts/run_optimized_experiments.py
```
