# Scripts Guide

The script directory contains a collection of scripts to reproduce the results from the paper, postprocess results, and to generate TikZ plots or movies thereof.

## Running Scripts

All scripts need to be run from the root of the repository, and can be run as

- `uv run koopmanrl_utils/<name of scripts>`

## Directory Structure

```
koopmanrl_utils/
├── movies/
│   ├── fluid_flow/                      # Subdirectory for experiments on the fluid flow environment
│   ├── __init__.py                      # Initialization file
│   ├── abstract_policy.py               # Abstract method for the policy
│   ├── AGENTS.md                        # Agents.md file for the subdirectory
│   ├── algo_policies.py                 # Algorithmic inspection of the Koopman policies
│   ├── default_policies.py              # Generate uncontrolled policies on the environments
│   ├── env_enum.py                      # Enumeration of the reinforcement learning environments
│   ├── generate_csvs.ipynb              # Stores the policies into a CSV
│   ├── generate_gifs.py                 # Generates GIF illustrations of the applied control policy
│   ├── generator.py                     # Generates controlled or uncontrolled trajectories
│   ├── hundred_episode_cost_average.py  # Averages across 100 episodes
│   └── plotting_trajectories.ipynb      # Plot the control trajectories
├── AGENTS.md                            # This file
├── dataframe_creator.py                 # Converts Tensorboard results to JSON data frames
├── interpret_koopman.json               # Configuration of test file to test interpretability on
├── interpret_koopman.py                 # Ingests a Koopman configuration, and interprets its tensor
├── plot_csv_from_tensorboards.py        # Ingests Tensorboard results and generates csv files
├── process_episodic_returns.py          # Generates episodic return plots from JSON dataframe
├── process_sakc_ablations.py            # Generates the ablation plots for the Soft Actor Koopman-Critic from the JSON dataframes
├── process_skvi_ablations.py            # Generates the ablation plots for the Soft Koopman Value Iteration from the JSON dataframes
├── run_optimized_experiments.py         # Runs the optimized experimental configurations of the Koopman algorithms
├── run_sakc_optimization.py             # Runs the Soft Actor Koopman Critic hyperparameter optimization
├── run_skvi_optimization.py             # Runs the Soft Koopman Value Iteration hyperparameter optimization
└── tsne_koopman_tensor.py               # Loads the saved Koopman tensors and generates a t-SNE plot from it
```

## Critical Patterns

### Working with simulation results

All utility scripts follow a few critical patterns induced by the structure of the reinforcement learning algorithms:

* The outputs of simulations are stored in the `runs/` folder, where `runs/` is found at the root of the repository. Each reinforcement learning experiment creates its own folder in which the Tensorboard file holding the experimental measurements can be found.
* All utility scripts essentially presume JSON files as inputs. The utility scripts to go from a Tensorboard file to a JSON file are:
    * `dataframe_creator.py` takes the path to the root of a filetree with the folders of experiments with their tensorboard files and returns a JSON file
    * `process_episodic_returns.py`, `process_sakc_ablations.py`, and `process_skvi_ablations.py` take said JSON file and return `.dat` frames for TikZ to generate episodic return plots, or 3D-surface plots for the ablations.
* The episodic return plots utilize a stratified bootstrapping scheme to generate 95% confidence intervals, which are used in the episodic return plots of the paper.
* Every single script is able to be executed in isolation.

### JSON Data Schema

The expected JSON schema of the scripts is the following:

```json
"<generated run name>": {
    "environment": "<name of reinforcement learning environment>",
    "rl_algorithm": "<reinforcement learning algorithm name>",
    "seed": 5412,
    "v_lr": 0.009423359172870875,
    "q_lr": 0.0017865746944645956,
    "episodic_returns": [
        -79874.671875,
        -77027.21875,
        -35590.19921875,
        -3206.347412109375,
        -1069.2921142578125,
        -1635.8477783203125,
        -859.1900634765625,
        -302.5566101074219,
        -508.9707336425781,
        -856.4261474609375,
        -218.68453979492188,
        -274.85968017578125,
        -216.1581573486328,
        -265.24371337890625,
        -303.97869873046875,
        -277.66534423828125,
        -792.0526733398438,
        -306.58154296875,
        -162.3712921142578,
        -204.51625061035156,
        -35.95591354370117,
        -173.9765625,
        -152.93174743652344,
        -622.8267822265625,
        -385.904541015625
    ],
    "steps": [
        1999,
        3999,
        5999,
        7999,
        9999,
        11999,
        13999,
        15999,
        17999,
        19999,
        21999,
        23999,
        25999,
        27999,
        29999,
        31999,
        33999,
        35999,
        37999,
        39999,
        41999,
        43999,
        45999,
        47999,
        49999
    ],
    "time": 1765986950
}
```

## Working Checklist

1. Review the relevant AGENTS guide(s) and existing tests/examples for the script you touch.
2. Prototype changes in single files or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (tests/test_*.py) alongside code changes.
4. Run the scoped pytest command (uv run test -m ...) before submitting.
5. Keep documentation edits minimal and aligned.
