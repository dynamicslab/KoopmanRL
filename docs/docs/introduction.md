---
id: introduction
sidebar_position: 1
title: Introduction
---

# Koopman-Assisted Reinforcement Learning

KoopmanRL is a reinforcement learning (RL) package built around two **Koopman-Assisted RL (KARL)** algorithms:

- **Soft Koopman Value Iteration (SKVI)** — discrete value iteration enhanced with a Koopman operator representation of the system dynamics.
- **Soft Koopman Actor-Critic (SAKC)** — an actor-critic algorithm that uses a Koopman tensor for improved policy learning in nonlinear dynamical systems.

The library also provides utilities to build upon individual components — use only the Koopman tensor, only specific algorithmic pieces, or the full KARL pipelines — and includes automatic hyperparameter tuning routines and four control-oriented benchmark environments.

## Background

The **Koopman operator** is an infinite-dimensional linear operator that describes the evolution of observable functions of a nonlinear dynamical system. By lifting the state space into a higher-dimensional observable space, the Koopman operator allows nonlinear dynamics to be treated as linear, enabling powerful linear analysis and control techniques to be applied.

KARL combines this operator-theoretic perspective with maximum-entropy reinforcement learning, yielding algorithms that:

- Learn compact, interpretable representations of environment dynamics.
- Achieve improved sample efficiency compared to model-free baselines.
- Produce policies whose value functions can be analysed through the Koopman lens.

## Citation

If you use KoopmanRL in your research, please cite:

```bibtex
@software{koopmanrl,
  author    = {Paehler, Ludger and Rozwood, Preston and Mehrez, Edward J.},
  title     = {KoopmanRL: Koopman-Assisted Reinforcement Learning},
  year      = {2026},
  url       = {https://github.com/dynamicslab/KoopmanRL},
}
```
