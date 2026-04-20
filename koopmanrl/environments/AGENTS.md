# Environments Guide

Library of reinforcement learning control environments heavily inspired by dynamical system control theory literature. Of the 4 control environments `Lorenz` is the one chaotic environment.

## Directory Guide

```
environments/
├── __init__.py       # Subdirectory initialization export all 4 environments
├── AGENTS.md         # This file
├── double_well.py    # The Stochastic double well environment
├── fluid_flow.py     # Fluid Flow control environment
├── linear_system.py  # Linear System control environment
├── lorenz.py         # Lorenz 1963 chaotic system control environment
└── test_env.py       # Utility to test an environment implementation
```

## Design Guide

All four environments follow two guiding principles:

* All environments follow the legacy `gym` standard
* Environments are allowed to run with FP64, and are run on CPU

## Working Checklist

1. Review the relevant AGENTS guide(s) and existing tests/examples for the functionality you touch.
2. Prototype changes in single files or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (tests/test_*.py) alongside code changes.
4. Run the scoped pytest command (uv run test -m ...) before submitting.
5. Keep documentation edits minimal and aligned.
