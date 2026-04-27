---
id: linear-system
sidebar_position: 1
title: Linear System
---

# Linear System (`LinearSystem-v0`)

The Linear System is the simplest benchmark environment and the canonical test bed for **LQR** and **SKVI**. The dynamics are globally linear, so LQR delivers the exact optimal solution and provides an unambiguous performance ceiling.

## Dynamics

The discrete-time state transition is

$$
x_{t+1} = A \, x_t + B \, u_t
$$

where $A \in \mathbb{R}^{3 \times 3}$ is a random stable matrix (largest absolute real eigenvalue $< 1$, drawn once at environment construction) and $B = \mathbf{1}_{3 \times 1}$.

## Spaces and limits

| Property | Value |
|----------|-------|
| State dimension | 3 |
| Action dimension | 1 |
| State range | $[-25, 25]^3$ |
| Action range | $[-10, 10]$ |
| Max episode steps | 200 |
| Integration | Direct (discrete map) |

## Cost function

$$
c(x, u) = x^\top Q x + u^\top R u, \qquad Q = I_3, \quad R = I_1
$$

Reference point: $x^* = \mathbf{0}$.

## Usage

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id LinearSystem-v0
uv run -m koopmanrl.linear_quadratic_regulator --env_id LinearSystem-v0
```

## Notes

- **A is randomised at construction.** Each new `LinearSystem()` instance draws a fresh $A$ matrix, so seeds must be fixed at the process level for reproducibility.
- **LQR is exact here.** SKVI and SAKC results on this environment should be compared against the LQR solution as an upper bound.
- **Source:** `koopmanrl/environments/linear_system.py`
