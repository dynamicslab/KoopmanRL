# Observables Guide

Implementation of the Koopman observables the Koopman tensor is eventually constructed out of. Implementations are provided in NumPy, and PyTorch with the dictionaries `monomials`, `indicators`, `gaussians` and `identity`, and the helpers that enumerate monomial powers (`allMonomialPowers`).

## Directory Structure

```
observables/
├── AGENTS.md             # This file
├── numpy_observables.py  # Implementation of the Koopman tensor observables in pure NumPy
└── torch_observables.py  # Implementation of the Koopman tensor observables in PyTorch
```

## Critical Patterns

* `koopmanrl/koopman_observables.py` is an identical copy of `torch_observables.py`, and it is the one SKVI, SAKC and `koopmanrl_utils/skvi_policy_checks.py` import. Keep the two files identical.
* The directory has no `__init__.py`; it is imported as a namespace package (`koopmanrl.koopman_tensor.observables.torch_observables`).

See the root `AGENTS.md` for setup, testing and the working checklist.
