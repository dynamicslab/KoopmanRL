---
id: lorenz
sidebar_position: 3
title: Lorenz Attractor
---

# Lorenz Attractor (`Lorenz-v0`)

The Lorenz environment implements the classic Lorenz system — a three-dimensional chaotic system with two unstable equilibria. The control task is to stabilise the state near one of the non-trivial equilibria $x^* = (\sqrt{\beta(\rho-1)},\, \sqrt{\beta(\rho-1)},\, \rho - 1)$.

## Dynamics

$$
\begin{aligned}
\dot{x}_1 &= \sigma (x_2 - x_1) + u \\
\dot{x}_2 &= x_1 (\rho - x_3) - x_2 \\
\dot{x}_3 &= x_1 x_2 - \beta x_3
\end{aligned}
$$

Standard Lorenz parameters: $\sigma = 10$, $\rho = 28$, $\beta = 8/3$.

The action $u$ enters additively through $\dot{x}_1$. Integration uses RK45 with $\Delta t = 0.01\,\text{s}$.

## Spaces and limits

| Property | Value |
|----------|-------|
| State dimension | 3 |
| Action dimension | 1 |
| $x_1$ range | $[-20, 20]$ |
| $x_2$ range | $[-50, 50]$ |
| $x_3$ range | $[0, 50]$ |
| Action range | $[-75, 75]$ |
| $\Delta t$ | 0.01 s |
| Max episode steps | 2000 (20 s) |

## Reference point

$$
x^* = \bigl(\sqrt{\beta(\rho-1)},\; \sqrt{\beta(\rho-1)},\; \rho - 1\bigr)
\approx (8.485,\; 8.485,\; 27.0)
$$

## Cost function

$$
c(x, u) = (x - x^*)^\top Q (x - x^*) + u^\top R u
$$

with $Q = I_3$ and $R = 0.001 \cdot I_1$. The small action cost ($R = 0.001$) reflects the large control authority needed to overcome chaotic divergence.

## LQR linearisation

The LQR baseline linearises around $x^*$:

$$
A_{\text{lin}} = \begin{pmatrix} -\sigma & \sigma & 0 \\ \rho - z^* & -1 & 0 \\ y^* & x^* & -\beta \end{pmatrix}, \qquad
B_{\text{lin}} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}
$$

## Usage

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id Lorenz-v0
uv run -m koopmanrl.soft_actor_koopman_critic \
    --config_file configurations/sakc_lorenz_hparams.json
```

## Notes

- The Lorenz system is chaotic for the standard parameters; without control, trajectories diverge from $x^*$ exponentially.
- The large action range ($\pm 75$) is needed for reliable stabilisation from arbitrary initial conditions.
- **Source:** `koopmanrl/environments/lorenz.py`
