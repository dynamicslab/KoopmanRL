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

### Using Optimized Hyperparameter Configurations

The `configurations/` directory contains the best-found hyperparameter configurations for SAKC and SKVI across each supported environment, produced by the Optuna optimization routines. The naming scheme is `<algo>_<env_slug>_hparams.json`, e.g.:

```
configurations/sakc_fluid_flow_hparams.json
configurations/skvi_lorenz_hparams.json
configurations/sakc_double_well_hparams.json
```

Both `koopmanrl.soft_actor_koopman_critic` and `koopmanrl.soft_koopman_value_iteration` accept a `--config_file` flag that loads hyperparameters directly from one of these JSON files:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_fluid_flow_hparams.json

uv run python -m koopmanrl.soft_koopman_value_iteration \
    --config_file configurations/skvi_lorenz_hparams.json
```

Any flag passed explicitly on the command line takes precedence over the value in the config file, so individual hyperparameters can be overridden without editing the JSON:

```bash
uv run python -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_fluid_flow_hparams.json \
    --seed 42 \
    --total_timesteps 100000
```
