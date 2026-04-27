---
id: fluid-flow
sidebar_position: 2
title: Fluid Flow
---

# Fluid Flow (`FluidFlow-v0`)

The Fluid Flow environment models a two-dimensional oscillatory shear flow. It is a canonical example of a system with nonlinear amplitude-frequency coupling that still admits a low-dimensional Koopman representation.

## Dynamics

The continuous-time dynamics are

$$
\begin{aligned}
\dot{x}_1 &= (\mu + A z)\, x_1 - \omega\, x_2 \\
\dot{x}_2 &= \omega\, x_1 + (\mu + A z)\, x_2 + u \\
\dot{z}   &= -\lambda\bigl(z - x_1^2 - x_2^2\bigr)
\end{aligned}
$$

with parameters $\mu = 0.1$, $\omega = 1.0$, $A = -0.1$, $\lambda = 1.0$.

The action $u$ enters through the $\dot{x}_2$ equation. Integration uses RK45 with $\Delta t = 0.01\,\text{s}$.

## Spaces and limits

| Property | Value |
|----------|-------|
| State dimension | 3 |
| Action dimension | 1 |
| $x_1, x_2$ range | $[-1, 1]$ |
| $z$ range | $[0, 1]$ |
| Action range | $[-10, 10]$ |
| $\Delta t$ | 0.01 s |
| Max episode steps | 2000 (20 s) |

## Cost function

$$
c(x, u) = x^\top Q x + u^\top R u, \qquad Q = I_3, \quad R = I_1
$$

Reference point: $x^* = \mathbf{0}$.

## LQR linearisation

The LQR baseline uses the Jacobian linearised at the origin $(x_1, x_2, z) = (0, 0, 0)$:

$$
A_{\text{lin}} = \begin{pmatrix} \mu & -\omega & 0 \\ \omega & \mu & 0 \\ 0 & 0 & -\lambda \end{pmatrix}, \qquad
B_{\text{lin}} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}
$$

## Usage

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id FluidFlow-v0
uv run -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_fluid_flow_hparams.json
```

## Notes

- The third state $z = x_1^2 + x_2^2$ is the amplitude squared; the system has a natural limit cycle at radius $\sqrt{-\mu / A} = 1$.
- **Source:** `koopmanrl/environments/fluid_flow.py`
