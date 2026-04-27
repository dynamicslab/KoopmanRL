---
id: double-well
sidebar_position: 4
title: Double Well
---

# Double Well (`DoubleWell-v0`)

The Double Well environment models a two-dimensional stochastic dynamical system on a double-well potential landscape. The control task is to drive the state to the origin despite additive state-dependent noise and the bi-stable potential.

## Dynamics

The continuous-time drift is

$$
\begin{aligned}
\dot{x}_1 &= 4 x_1 - 4 x_1^3 + u \\
\dot{x}_2 &= -2 x_2 + u
\end{aligned}
$$

with additive stochastic diffusion:

$$
dx = f(x, u)\,dt + \sigma(x)\,dW
$$

where $\sigma(x) = \begin{pmatrix} 0.7 & x_1 \\ 0 & 0.5 \end{pmatrix}$ and $W$ is a Wiener process. Integration uses the Euler–Maruyama scheme with $\Delta t = 0.01\,\text{s}$.

## Potential surface

The uncontrolled potential is

$$
V(x_1, x_2) = (x_1^2 - 1)^2 + x_2^2
$$

This has two stable wells at $(\pm 1, 0)$ and a saddle at the origin. The control objective is to hold the state at the origin, which is an unstable equilibrium of the uncontrolled system.

## Spaces and limits

| Property | Value |
|----------|-------|
| State dimension | 2 |
| Action dimension | 1 |
| State range | $[-2, 2]^2$ |
| Action range | $[-25, 25]$ |
| $\Delta t$ | 0.01 s |
| Max episode steps | 2000 (20 s) |

## Cost function

$$
c(x, u) = x^\top Q x + u^\top R u, \qquad Q = I_2, \quad R = I_1
$$

Reference point: $x^* = \mathbf{0}$.

## Observation note

During trajectory generation, `Generator` appends the potential $V(x)$ as a third component, producing a 3-dimensional observation $(x_1, x_2, V(x_1, x_2))$. The visualisation pipeline uses $(x_1, x_2)$ as the horizontal plane and $V$ as the vertical axis.

## LQR linearisation

$$
A_{\text{lin}} = \begin{pmatrix} -8 & 0 \\ 0 & -2 \end{pmatrix}, \qquad
B_{\text{lin}} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

## Usage

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id DoubleWell-v0
uv run -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_double_well_hparams.json
```

## Notes

- The stochastic diffusion $\sigma(x)$ is state-dependent: the off-diagonal term $\sigma_{12} = x_1$ couples noise into both state components.
- Trajectories draw from pre-generated random numbers for efficiency; a fixed seed must be set before environment construction for fully reproducible rollouts.
- **Source:** `koopmanrl/environments/double_well.py`
