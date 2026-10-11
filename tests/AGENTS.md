# Tests Guide

The pytest suite of KoopmanRL. `pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["."]`, so `uv run pytest` works from the repository root without installing the package in editable mode.

## Running Tests

```bash
uv run pytest tests/test_<area>.py            # the tests of what you touch
uv run pytest tests/test_<area>.py -k <name>  # a single test
uv run pytest --deselect tests/test_hparam_opt.py  # everything else (several minutes)
```

There are no pytest markers, so there is no fast/slow split to select with `-m`.

## Directory Structure

```
tests/
├── AGENTS.md                              # This file
├── utils.py                               # run_module: `uv run -m <module>` from the repository root, 300 s timeout
├── test_rl.py                             # Short runs of LQR, both SACs, SKVI and SAKC on the four environments; same-seed runs repeat
├── test_hparam_opt.py                     # One-sample runs of skvi_optuna_opt and sakc_optuna_opt on the four environments (Ray Tune)
├── test_koopman_tensor_regressors.py      # koopman_tensor/regressors.py and the four copies of the tensor class
├── test_koopman_prediction_validation.py  # koopmanrl_utils/koopman_prediction_validation.py
├── test_koopman_regressor_comparison.py   # koopmanrl_utils/koopman_regressor_comparison.py
├── test_skvi_policy_checks.py             # koopmanrl_utils/skvi_policy_checks.py
├── test_skvi_sensitivity_checks.py        # koopmanrl_utils/skvi_sensitivity_checks.py
├── test_tsne_koopman_tensor.py            # koopmanrl_utils/tsne_koopman_tensor.py
├── test_run_optimized_experiments.py      # Launcher of the episodic-return runs (commands, filters, configurations/episodic_returns.json)
├── test_run_ablations.py                  # Launcher of the ablation runs (grids, configurations/ablations.json)
├── test_episodic_return_processing.py     # dataframe_creator.py and process_episodic_returns.py
├── test_ablation_processing.py            # process_skvi_ablations.py and process_sakc_ablations.py
└── test_workflow_helpers.py               # workflow/helpers.py, loaded by path without Snakemake
```

`koopmanrl_utils/movies/` has no tests.

## Critical Patterns

* Most tests run a module as a subprocess through `tests/utils.py::run_module`, with the repository root as working directory. `test_rl.py` therefore writes `runs/` and `saved_models/` into the repository root (both gitignored).
* The reproducibility tests in `test_rl.py` run each module twice in separate `tmp_path` folders with `sys.executable` and compare the logged returns; they rely on the `gelsd` solves of SKVI and SAKC (see `koopmanrl/AGENTS.md`).
* `test_hparam_opt.py` needs care:
    * It asks Ray for `--cpu_cores_per_trial=16`, so on a machine with fewer than 16 cores no trial is ever scheduled and the test runs into its 600 s timeout.
    * It sets `RAY_TMPDIR` to pytest's `tmp_path`. On macOS that path makes Ray's AF_UNIX socket path exceed the OS limit (about 104 bytes) and the test fails at startup. Deselect it there.
* `test_workflow_helpers.py` imports `workflow/helpers.py` by path. Snakemake is not in the project environment, so the rule files themselves are checked with a dry run (`uvx --python 3.12 snakemake -n <target>`), not by pytest.
* CI does not run pytest; run the tests of the code you change locally before opening a pull request.
