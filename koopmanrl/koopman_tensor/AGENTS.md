# Koopman Tensor Guide

Subdirectory containing the logic for the Koopman tensor generation. The core here can be broken down to two core files `numpy_tensor.py`, and `torch_tensor.py`.

## Running Core Functionality

While the tensors themselves are not executable, the generation routine can be run on the individual environments with

```bash
uv run python -m koopmanrl.koopman_tensor.generate_tensor
```

This will per default generate a Koopman tensor for the linear system. The options here are

* `LinearSystem-v0`
* `FluidFlow-v0`
* `Lorenz-v0`
* `DoubleWell-v0`

which for the example of the `FluidFlow-v0` would be executed as

```bash
uv run python -m koopmanrl.koopman_tensor.generate_tensor --env_id FluidFlow-v0
```

## Directory Structure

```
koopman_tensor/
├── observables/        # Subdirectory containing the NumPy, and Torch observables
├── __init__.py         # Initialization of the subdirectory
├── AGENTS.md           # This file
├── generate_tensor.py  # Generating the Koopman tensor for a specific environment
├── numpy_tensor.py     # Koopman tensor implementation in pure NumPy
├── torch_tensor.py     # Koopman tensor implementation in PyTorch
└── utils.py            # Utilities for loading and storing Koopman tensors
```

## Working Checklist

1. Review the relevant AGENTS guide(s) and existing tests/examples for the functionality you touch.
2. Prototype changes in single files or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (tests/test_*.py) alongside code changes.
4. Run the scoped pytest command (uv run test -m ...) before submitting.
5. Keep documentation edits minimal and aligned.
