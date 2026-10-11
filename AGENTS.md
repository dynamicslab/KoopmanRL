# KoopmanRL Agent Guide

Use this note as the entry point before touching the repository. KoopmanRL is the code of the paper *Koopman-Assisted Reinforcement Learning* (arXiv:2403.02290): the two KARL algorithms, Soft Koopman Value Iteration (SKVI) and Soft Actor Koopman-Critic (SAKC), the LQR and SAC baselines, four `gym` control environments, the scripts that reproduce the results of the paper, and the Snakemake workflows that chain them.

## Read First
- `koopmanrl/AGENTS.md` - algorithms, baselines, hyperparameter optimization, and the copies of the Koopman tensor class
- `koopmanrl/environments/AGENTS.md` - the four environments and how they are registered
- `koopmanrl/koopman_tensor/AGENTS.md` - tensor classes, the shared regressors, `generate_tensor`
- `koopmanrl/koopman_tensor/observables/AGENTS.md` - dictionaries (monomials, indicators, ...)
- `koopmanrl_utils/AGENTS.md` - launchers, processing scripts, validation and interpretability checks
- `koopmanrl_utils/movies/AGENTS.md` and `koopmanrl_utils/movies/PIPELINE.md` - trajectory figures and GIFs
- `configurations/AGENTS.md` - tuned hyperparameters and the run lists of the reproduction pipelines
- `tests/AGENTS.md` - how the tests run, and which ones need care
- `workflow/README.md` - Snakemake workflows, their commands, and the conventions for adding one
- `koopmanrl_utils/EPISODIC_RETURNS.md`, `ABLATIONS.md`, `TSNE.md` - from the runs to the tables of each figure of the paper

## Setup
- `uv sync --group dev` installs the package and the dev tools (pre-commit, pytest, pytest-cov). Python is pinned to `==3.10.*`.
- The environments use the legacy `gym==0.23.1`; its warning that Gym is unmaintained is expected and harmless.
- Snakemake is not in the project environment (it needs Python ≥ 3.11). It is run as a tool, `uvx --python 3.12 snakemake ...`, and its jobs call the project environment; see `workflow/README.md`.

## Repository Layout
```
configurations/          tuned SKVI/SAKC hyperparameters (*_hparams.json) and the run lists of the three reproduction pipelines
docs/                    Docusaurus documentation site, deployed to https://dynamicslab.github.io/KoopmanRL/
koopmanrl/               the package: algorithms, baselines, HPO drivers, environments/, koopman_tensor/
koopmanrl_utils/         scripts that reproduce, process and check the results of the paper; movies/ for figures and GIFs
tests/                   pytest suite, mostly subprocess runs of the modules
workflow/                Snakemake workflows: Snakefile, rules/*.smk, helpers.py, profiles/default
.github/workflows/       CI: pre-commit lint, docs build on PRs, docs deploy on main
```

## Running Code
- Run everything from the repository root as a module: `uv run -m koopmanrl.<module>` or `uv run -m koopmanrl_utils.<module>`. Every algorithm and most scripts take `--help` (typed `tap` parsers).
- Runs write TensorBoard logs to `runs/` and checkpoints to `saved_models/` in the working directory. These and the result directories of the scripts (`*_results/`, `results/`, `figures/`, `video_frames/`, `.snakemake/`) are gitignored; do not commit outputs. Compiled Python files (`__pycache__/`, `*.py[cod]`) are gitignored in every directory and are not tracked.
- Do not execute `koopmanrl_utils/run_skvi_optimization.py` or `run_sakc_optimization.py` to inspect them: they take no arguments, so even `--help` starts four full Ray Tune studies. Read the source instead. Before executing any other script, check that it has an argument parser.
- `koopmanrl.{skvi,sakc}_optuna_opt` default to `--cpu_cores_per_trial 28`, which Ray cannot schedule on a smaller machine (the study then idles forever), and to a `--storage_dir` on the authors' machine. Pass both when running them.
- The reproduction launchers (`run_optimized_experiments`, `run_ablations`) start hundreds of 50,000-step runs by default. Use `--dry_run` to list the commands, and the filters described in their guides to run a part.

## Global Conventions
- Reinforcement learning core logic has to stay script-addressable through `uv run -m ...`.
- Prefer a CleanRL single-file style for the algorithms; outside of hyperparameter optimization, avoid splitting an algorithm over modules.
- Parallelism within one process or launcher uses [Ray](https://github.com/ray-project/ray) (`--num_workers` of the launchers, Ray Tune for HPO). Chaining experiments, data frames and tables is done by the Snakemake workflows in `workflow/`.
- A list that a launcher and its workflow both need (algorithms, benchmarks, seeds, grids) is kept once, in `configurations/<workflow>.json`.
- The baselines `linear_quadratic_regulator`, `sac_continuous_action` and `value_based_sac_continuous_action` stay untouched, except for their `--seed` handling.
- The Koopman tensor class exists in four copies (see `koopmanrl/AGENTS.md`). A change to the tensor must be made in all of them; they share `koopmanrl/koopman_tensor/regressors.py`.
- SKVI and SAKC solve their least squares with the LAPACK driver `gelsd`, so same-seed runs repeat bit for bit on one machine at a fixed number of threads (`OMP_NUM_THREADS`), not across thread counts. Keep it that way when touching these solves.
- When a command-line interface changes, update the guide that documents it (README.md, the guides in `koopmanrl_utils/`, `workflow/README.md`, `docs/docs/`).

## Testing, Linting and CI
- Run the tests of what you touch: `uv run pytest tests/test_<area>.py [-k <name>]`. There are no pytest markers; `uv run pytest` runs the whole suite (several minutes). Details and known failures are in `tests/AGENTS.md`.
- `uv run pre-commit run --files <changed files>` (or `--all-files`): ruff (`--fix`, format; line length 120), isort (`--profile black`), vulture on `koopmanrl/`, the basic pre-commit-hooks (whitespace, end of file, YAML, files > 1 MB, merge conflicts, debug statements), and the local `forbid-compiled-python` hook, which fails on any `.pyc`/`.pyo`/`.pyd` or `__pycache__/` file.
- CI runs only pre-commit (`lint.yml`) and the docs build (`test-deploy-docs.yml` on PRs, `deploy-docs.yml` on `main`, Node 20). It does not run pytest, so run the relevant tests locally.
- Docs site: `cd docs && npm ci && npm run build`; the build fails on broken links.

## Working Checklist
1. Read the guide(s) of the directories you touch, and the tests that cover them.
2. Prototype changes in modules or helper scripts; avoid interactive REPL work.
3. Add or update targeted tests (`tests/test_*.py`, using `tests/utils.py::run_module` for module runs).
4. Run `uv run pytest tests/test_<area>.py` and `uv run pre-commit run --files <changed files>` before submitting.
5. Keep documentation edits minimal, aligned with the per-directory format, and in the same PR as the change they document.
