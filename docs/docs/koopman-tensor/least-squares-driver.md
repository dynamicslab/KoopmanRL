---
id: least-squares-driver
sidebar_position: 3
title: Least-Squares Driver
---

# Least-Squares Driver

On the CPU, `torch.linalg.lstsq` uses the LAPACK driver `gelsy` unless told otherwise. In the torch releases checked (2.9.1 installed; the same code in the sources of 1.9.0 to 2.13.0) it passes `gelsy` an uninitialised pivot array, so with identical inputs the solution differs from process to process in its last digits. For SAKC, whose critic is trained through the Koopman tensor, this made same-seed runs log different returns once training had started.

## Where which driver is used

| Code | Driver | Effect |
|------|--------|--------|
| `ols` in `koopmanrl/soft_koopman_value_iteration.py` and `koopmanrl/soft_actor_koopman_critic.py`, and the solve for the SKVI value-function weights | `gelsd` | On one machine with a fixed number of numerical threads, the tensor and the SKVI value-function weights are the same bit for bit in every process, and same-seed SAKC runs log the same returns. |
| `ols` in `koopmanrl/koopman_tensor/torch_tensor.py` (used by `generate_tensor`, `koopman_prediction_validation` and `koopman_regressor_comparison`) | default (`gelsy`) | Results differ in the last digits from call to call. `gelsd` was not adopted here because it is less accurate on the linear system, where the tensor model is exact (one-step errors of 1e-12 instead of 3e-16). |
| `koopmanrl_utils.tsne_koopman_tensor` | `--lstsq_driver`, default `gelsd` | `gelsd` and `gelss` give the same tensor in every call; `gelsy` does not. See [t-SNE](../reproduction/tsne.md#least-squares-driver). |

## What it changes

- **Bit-reproducibility needs a fixed thread count.** Between one numerical thread and two, the SKVI/SAKC tensor still differs in its last digits. Set `OMP_NUM_THREADS` to the same value for runs that should agree.
- **The numbers barely move.** The regressions of the tuned configurations are full rank on all four benchmarks, so both drivers return the same solution up to rounding: the tensors differed by a relative 2e-13 to 3e-10. Runs made after the change are nevertheless other runs than those made before it, in the same way that two earlier same-seed SAKC runs differed from each other.

The change was made in commit [`d0d3016`](https://github.com/dynamicslab/KoopmanRL/commit/d0d30162bdef481190e4418144bd8c9d7e5086a8), whose message records the checks.
