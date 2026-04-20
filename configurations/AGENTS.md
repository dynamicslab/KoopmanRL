# Configurations Directory Guide

The `configurations/` directory houses the best found hyperparameter configurations for the newly introduced reinforcement learning configurations.

## Contents

```
configurations/
├── sakc_double_well_hparams.json    # Best configuration of the Soft Actor Koopman-Critic for the Stochastic Double Well
├── sakc_fluid_flow_hparams.json     # Best configuration of the Soft Actor Koopman-Critic for the Fluid Flow
├── sakc_linear_system_hparams.json  # Best configuration of the Soft Actor Koopman-Critic for the Linear System
├── sakc_lorenz_hparams.json         # Best configuration of the Soft Actor Koopman-Critic for Lorenz
├── skvi_double_well_hparams.json    # Best configuration of the Soft Koopman Value Iteration for the Stochastic Double Well
├── skvi_fluid_flow_hparams.json     # Best configuration of the Soft Koopman Value Iteration for the Fluid Flow
├── skvi_linear_system_hparams.json  # Best configuration of the Soft Koopman Value Iteration for the Linear System
└── skvi_lorenz_hparams.json         # Best configuration of the Soft Koopman Value Iteration for Lorenz
```

## JSON Schema of Configuration

The data schema for the two algorithms varies slightly as such their schema is presented separately

### Soft Actor Koopman-Critic

An example of the JSON schema below:

```json
{
    "env-id": "DoubleWell-v0",
    "seed": 469,
    "v-lr": 0.0003310304069101045,
    "q-lr": 0.00039795751924458065,
    "num-paths": 150,
    "num-steps-per-path": 300,
    "state-order": 4,
    "action-order": 4,
    "total-timesteps": 50000,
    "target-score": null,
    "num-envs": 1,
    "metric": "charts/episodic_return",
    "metric-last-n-average-window": 5
}
```

### Soft Koopman Value Iteration

An example of the JSON schema below:

```json
{
    "env-id": "FluidFlow-v0",
    "seed": 6517,
    "learning-rate": 0.00031904756404241047,
    "number-of-train-epochs": 125,
    "num-paths": 200,
    "num-steps-per-path": 225,
    "state-order": 4,
    "action-order": 2,
    "target-score": null,
    "total-timesteps": 50000,
    "num-envs": 1,
    "metric": "charts/episodic_return",
    "metric-last-n-average-window": 5
}
```
