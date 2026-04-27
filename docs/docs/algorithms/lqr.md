---
id: lqr
sidebar_position: 3
title: Linear Quadratic Regulator (LQR)
---

# Linear Quadratic Regulator (LQR)

The Linear Quadratic Regulator is a classical optimal control algorithm included in KoopmanRL as a baseline comparator. It provides an exact analytical solution for linear systems with quadratic cost, serving as an upper-bound reference on environments where linearity holds.

## Running LQR

```bash
uv run -m koopmanrl.linear_quadratic_regulator
```

With a specific environment:

```bash
uv run -m koopmanrl.linear_quadratic_regulator --env_id FluidFlow-v0
```

## When to use LQR

LQR is most informative on the `LinearSystem-v0` environment where its optimality assumptions are exactly satisfied. On nonlinear environments (Lorenz, Fluid Flow, Double Well) it provides a linearised-dynamics baseline that the KARL algorithms aim to outperform.

## Source

`koopmanrl/linear_quadratic_regulator.py`
