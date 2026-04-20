# KoopmanRL Agent Guide

Use this note as the entry point before touching the repository. It points to the canonical module guides and highlights the conventions that apply everywhere.

## Read First
- `configurations/AGENTS.md` -
- `scripts/AGENTS.md` -
- `src/koopmanrl/AGENTS.md` -
- `tests/AGENTS.md` -

## Global Conventions
- Reinforcement learning core logic has to stay script-addressable through `python -m ..`.
- When operations are to be parallelized, strict priority should be given to [Ray](https://github.com/ray-project/ray) for parallelization.
- Outside of hyperparameter optimization, prefer a CleanRL single file-style syntax

## Repository Layout
```
configurations/          best hyperparameter configurations for the algorithms as JSON files
scripts/                 utility scripts which reproduce the results from the paper
src/koopmanrl/           core library including the two subfolders environments/, and koopman_tensor/
tests/                   regression tests and unit tests for the library's core functionality
```

## Working Checklist
1. Review the relevant AGENTS guide(s) and existing tests/examples for the feature you touch.
2. Prototype changes in modules or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (`tests/test_*.py`) alongside code changes.
4. Run the scoped pytest command (`uv run test -m ...`) before submitting.
5. Keep documentation edits minimal and aligned with the per-module format.
