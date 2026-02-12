# Observables Guide

Implementation of the Koopman observables the Koopman tensor is eventually constructed out of. Implementations are provided in NumPy, and PyTorch with the core feature set being Monomials, and Gaussians.

## Directory Structure

```
observables/
├── AGENTS.md             # This file
├── numpy_observables.py  # Implementation of the Koopman tensor observables in pure NumPy
└── torch_observables.py  # Implementation of the Koopman tensor observables in PyTorch
```

## Working Checklist

1. Review the relevant AGENTS guide(s) and existing tests/examples for the functionality you touch.
2. Prototype changes in single files or helper scripts—avoid interactive REPL work.
3. Add or update targeted tests (tests/test_*.py) alongside code changes.
4. Run the scoped pytest command (uv run test -m ...) before submitting.
5. Keep documentation edits minimal and aligned.
