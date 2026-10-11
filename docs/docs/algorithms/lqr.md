---
id: lqr
sidebar_position: 3
title: Linear Quadratic Regulator (LQR)
---

# Linear Quadratic Regulator (LQR)

The Linear Quadratic Regulator is a classical optimal control algorithm included in KoopmanRL as a baseline comparator. It provides an exact analytical solution for linear systems with quadratic cost, serving as an upper-bound reference on environments where linearity holds.

KoopmanRL uses the discounted, entropy-regularised (maximum-entropy) LQR. The gain $C$ and the Riccati matrix $P$ are computed for the discounted system ($A$ scaled by $\sqrt{\gamma^{\Delta t}}$, $R$ by $1/\gamma^{\Delta t}$), from the environment's linearisation `continuous_A`, `continuous_B` (continuous-time Riccati equation), or, for `LinearSystem-v0`, from its discrete `A`, `B` with $\Delta t = 1$. Actions are sampled from a Gaussian centred on the LQR action,

$$
u \sim \mathcal{N}\bigl(-C\,(x - x^*),\; \alpha\,(R + B^\top P B)^{-1}\bigr),
$$

where $x^*$ is the environment's reference point and $\alpha$ the entropy coefficient.

## Running LQR

```bash
uv run -m koopmanrl.linear_quadratic_regulator
```

With a specific environment:

```bash
uv run -m koopmanrl.linear_quadratic_regulator --env_id FluidFlow-v0
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to run on |
| `--seed` | random | Random seed |
| `--total_timesteps` | `50000` | Environment steps |
| `--gamma` | `0.99` | Discount factor |
| `--alpha` | `1.0` | Entropy regularisation coefficient |

Without `--seed`, a seed is drawn at random and recorded in the run name (`<env_id>__<exp_name>__<seed>__<timestamp>`), so a run can be repeated with `--seed`.

## When to use LQR

LQR is most informative on the `LinearSystem-v0` environment where its optimality assumptions are exactly satisfied. On nonlinear environments (Lorenz, Fluid Flow, Double Well) it provides a linearised-dynamics baseline that the KARL algorithms aim to outperform.

## Source

`koopmanrl/linear_quadratic_regulator.py`
