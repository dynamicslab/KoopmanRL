# KoopmanRL Utilities Guide

The `koopmanrl_utils/` directory contains a collection of scripts to reproduce the results from the paper, postprocess results, and to generate TikZ plots or movies thereof.

## Running Scripts

All scripts need to be run from the root of the repository, as modules:

- `uv run -m koopmanrl_utils.<name of script>` (the directory has no `__init__.py`; it is imported as a namespace package)

Most scripts have a `tap` parser and list their options with `--help`. `run_skvi_optimization.py` and `run_sakc_optimization.py` do not: they ignore their arguments, so even `--help` starts four full hyperparameter optimizations. Read them instead of running them. `interpret_koopman.py` is an unfinished stub (it reads `args.skvi_stored_weights`, which its parser does not define, and `interpret_koopman.json` holds paths on the authors' machine), and `plot_csv_from_tensorboards.py` defaults to run names of 2024.

## Directory Structure

```
koopmanrl_utils/
├── movies/                              # Trajectory figures and GIFs of the trained policies (see movies/AGENTS.md, movies/PIPELINE.md)
├── ABLATIONS.md                         # From the runs to the pgfplots-ready tables of the two ablation figures
├── AGENTS.md                            # This file
├── dataframe_creator.py                 # Converts Tensorboard results to JSON data frames
├── EPISODIC_RETURNS.md                  # From the runs to the pgfplots-ready tables of the episodic-return figures
├── interpret_koopman.json               # Configuration of test file to test interpretability on
├── interpret_koopman.py                 # Unfinished stub for interpreting a stored Koopman tensor; does not run
├── koopman_prediction_validation.py     # Held-out one-step and multi-step prediction error of the Koopman tensor
├── koopman_regressor_comparison.py      # Held-out error of the Koopman tensor under 13 regression algorithms; TikZ figures
├── plot_csv_from_tensorboards.py        # Ingests Tensorboard results and generates csv files
├── process_episodic_returns.py          # Writes the pgfplots tables (.dat, or .csv) of the episodic returns from the JSON data frames
├── process_sakc_ablations.py            # Writes the pgfplots tables of the SAKC ablation from the JSON data frames
├── process_skvi_ablations.py            # Writes the pgfplots tables of the SKVI ablation from the JSON data frames
├── run_ablations.py                     # Runs the experiments of the two ablation figures: the hyperparameter grids of SKVI and SAKC
├── run_optimized_experiments.py         # Runs the experiments of the episodic-return figures: tuned SKVI and SAKC, and the LQR and SAC baselines
├── run_sakc_optimization.py             # Runs sakc_optuna_opt on the four environments; takes no arguments
├── run_skvi_optimization.py             # Runs skvi_optuna_opt on the four environments; takes no arguments
├── skvi_policy_checks.py                # Reads the SKVI policy off the Koopman tensor and checks it (LQR, pruning)
├── skvi_sensitivity_checks.py           # Koopman-tensor accuracy along the SKVI policy; sensitivity of SKVI's control
├── TSNE.md                              # From the Koopman tensors to the pgfplots-ready tables of the t-SNE figure
└── tsne_koopman_tensor.py               # Joint t-SNE of the Koopman tensors of the four benchmarks in a common basis; pgfplots CSV files
```

## Critical Patterns

### Working with simulation results

All utility scripts follow a few critical patterns induced by the structure of the reinforcement learning algorithms:

* The outputs of simulations are stored in the `runs/` folder of the directory an algorithm is started from: at the root of the repository for an algorithm run by hand, in `--output_dir` for the runs of `run_optimized_experiments.py` and `run_ablations.py`, and in `results/<workflow>/runs/...` for the Snakemake workflows. Each reinforcement learning experiment creates its own folder in which the Tensorboard file holding the experimental measurements can be found.
* All utility scripts essentially presume JSON files as inputs. The utility scripts to go from a Tensorboard file to a JSON file are:
    * `dataframe_creator.py` takes the path to the root of a filetree with the folders of experiments with their tensorboard files and returns a JSON file
    * `process_episodic_returns.py`, `process_sakc_ablations.py`, and `process_skvi_ablations.py` take said JSON file and return `.dat` frames for TikZ to generate episodic return plots, or 3D-surface plots for the ablations. All three write a `.csv` table instead when the output name ends in `.csv`; `EPISODIC_RETURNS.md` and `ABLATIONS.md` walk through the whole path for the episodic returns and for the ablations.
* The episodic return plots utilize a stratified bootstrapping scheme to generate 95% confidence intervals, which are used in the episodic return plots of the paper.
* Every script except the two `run_*_optimization.py` drivers can be executed in isolation.
* The launchers `run_optimized_experiments.py` and `run_ablations.py` start hundreds of 50,000-step runs by default. `--dry_run` prints the commands; `--num_workers` runs them in parallel with Ray; the filters for running a part are described at the top of each script and in `EPISODIC_RETURNS.md` / `ABLATIONS.md`.
* `tsne_koopman_tensor.py` solves its regressions with `--lstsq_driver gelsd` by default, the driver of SKVI and SAKC; see `TSNE.md`.
* Each script writes into its own gitignored results directory in the working directory (`episodic_returns_results/`, `ablation_results/`, `tsne_koopman_tensor_results/`, `skvi_policy_checks_results/`, `skvi_sensitivity_checks_results/`, `koopman_prediction_validation_results/`, `koopman_regressor_comparison_results/`, `video_frames/`, `figures/`). `tests/AGENTS.md` lists which test file covers which script.
* The Snakemake workflows in `workflow/` chain these scripts, one job per run, data frame and table. Lists that a script and its workflow both need, such as the seeds of `run_optimized_experiments.py`, are kept in `configurations/<workflow>.json` and read by both. `workflow/README.md` has the commands and conventions.

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

See the root `AGENTS.md` for setup, testing and the working checklist.
