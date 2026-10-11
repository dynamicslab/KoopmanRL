# Environments Guide

Library of reinforcement learning control environments heavily inspired by dynamical system control theory literature. Of the 4 control environments `Lorenz` is the one chaotic environment. `DoubleWell` is two-dimensional and stochastic; the other three are three-dimensional.

## Directory Guide

```
environments/
├── __init__.py       # Imports the 4 environment classes, which registers them with gym
├── AGENTS.md         # This file
├── double_well.py    # The Stochastic double well environment
├── fluid_flow.py     # Fluid Flow control environment
├── linear_system.py  # Linear System control environment
├── lorenz.py         # Lorenz 1963 chaotic system control environment
└── test_env.py       # Rolls out an environment: uv run -m koopmanrl.environments.test_env --env_id <id> --seed <n>
```

## Design Guide

All four environments follow two guiding principles:

* All environments follow the legacy `gym` standard
* Environments are allowed to run with FP64, and are run on CPU

## Registration

Each environment module calls `gym.envs.registration.register` at import, under the ids `LinearSystem-v0`, `FluidFlow-v0`, `Lorenz-v0` and `DoubleWell-v0`. `gym.make("<id>")` therefore only works after `import koopmanrl.environments` (or one of its modules); a script that skips the import fails with `gym.error.NameNotFound`.

* `gym==0.23.1` is pinned; its warning that Gym is unmaintained is expected.
* `koopmanrl_utils/movies/` plots only FluidFlow, Lorenz and DoubleWell.
* `__pycache__/` of this directory is regenerated on import and gitignored, like every compiled Python file of the repository; the `forbid-compiled-python` pre-commit hook rejects one that is committed anyway (e.g. force-added).

See the root `AGENTS.md` for setup, testing and the working checklist.
