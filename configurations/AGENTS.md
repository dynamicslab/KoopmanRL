# Configurations Directory Guide

The `configurations/` directory houses the best found hyperparameter configurations for the newly introduced reinforcement learning configurations.

## Contents

```
configurations/
├── ablations.json                   # Grids and seeds of the two ablation figures; read by koopmanrl_utils/run_ablations.py and workflow/rules/ablations.smk
├── episodic_returns.json            # Algorithms, benchmarks and seeds of the episodic-return figures; read by koopmanrl_utils/run_optimized_experiments.py and workflow/rules/episodic_returns.smk
├── sakc_double_well_hparams.json    # Best configuration of the Soft Actor Koopman-Critic for the Stochastic Double Well
├── sakc_fluid_flow_hparams.json     # Best configuration of the Soft Actor Koopman-Critic for the Fluid Flow
├── sakc_linear_system_hparams.json  # Best configuration of the Soft Actor Koopman-Critic for the Linear System
├── sakc_lorenz_hparams.json         # Best configuration of the Soft Actor Koopman-Critic for Lorenz
├── skvi_double_well_hparams.json    # Best configuration of the Soft Koopman Value Iteration for the Stochastic Double Well
├── skvi_fluid_flow_hparams.json     # Best configuration of the Soft Koopman Value Iteration for the Fluid Flow
├── skvi_linear_system_hparams.json  # Best configuration of the Soft Koopman Value Iteration for the Linear System
├── skvi_lorenz_hparams.json         # Best configuration of the Soft Koopman Value Iteration for Lorenz
└── tsne.json                        # Arguments of koopmanrl_utils/tsne_koopman_tensor.py that the t-SNE workflow passes on, by step; read by workflow/rules/tsne.smk
```

`ablations.json`, `episodic_returns.json` and `tsne.json` also list the settings of their Snakemake workflow: the workflow refuses a key that its file does not have, so a new setting is added to the file first. The algorithms, the benchmarks and the grids of these files are not settings: the launchers read them from the files, and the workflow refuses other values given on its command line. `workflow/README.md` describes the settings.

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
