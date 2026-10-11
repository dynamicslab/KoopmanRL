---
id: skvi
sidebar_position: 1
title: Soft Koopman Value Iteration (SKVI)
---

# Soft Koopman Value Iteration (SKVI)

Soft Koopman Value Iteration (SKVI) is the first of the two KARL algorithms. It replaces the standard Bellman backup in soft value iteration with one that exploits a learned Koopman tensor representation of the environment's transition dynamics.

## Algorithm overview

SKVI operates in discrete value-iteration fashion:

1. **Koopman tensor construction** — collect trajectories from the environment with a random agent and fit a Koopman tensor $\mathcal{K}$ that maps observable functions of the current state-action pair to observables of the next state, $\phi(x') \approx K(u)\,\phi(x)$.
2. **Lifted value iteration** — represent the value function as linear in the observables, $V(x) = w^\top \phi(x)$. Because of this, the expected next-state value for an action is available in closed form, $\mathbb{E}[V(x')] \approx w^\top K(u)\,\phi(x)$, without sampling next states.
3. **Policy extraction** — the policy is a softmax (Gibbs) distribution over `--num_actions` evenly spaced actions spanning the action space, with probabilities proportional to $\exp\bigl(-(c(x,u) + \gamma\, w^\top K(u)\,\phi(x))/\alpha\bigr)$; actions are sampled from it.

In every training epoch, the soft Bellman target (the policy-weighted expectation of cost, $\alpha \log \pi$ and discounted next-state value) is computed on a batch of `--batch_size` states from the tensor's training data, and the value weights $w$ are refit to it by least squares (`torch.linalg.lstsq` with the LAPACK driver `gelsd`). The linear structure of the Koopman operator is what makes both the expectation over next states and this regression step cheap.

## Running SKVI

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --env_id LinearSystem-v0
```

All supported environment IDs:

| Environment | `--env_id` |
|-------------|-----------|
| Linear System | `LinearSystem-v0` |
| Fluid Flow | `FluidFlow-v0` |
| Lorenz | `Lorenz-v0` |
| Double Well | `DoubleWell-v0` |

## Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--env_id` | `LinearSystem-v0` | Environment to train on |
| `--seed` | `1` | Random seed |
| `--total_timesteps` | `50000` | Environment steps of the evaluation rollout after training |
| `--num_paths` | `100` | Random-agent trajectories for Koopman tensor fitting |
| `--num_steps_per_path` | `300` | Steps per trajectory for Koopman tensor fitting |
| `--state_order` | `2` | Monomial order of the state observables $\phi(x)$ |
| `--action_order` | `2` | Monomial order of the action dictionary |
| `--regressor` | `ols` | Regressor for the Koopman tensor: `ols`, `ridge`, `sindy` or `rrr` |
| `--num_actions` | `101` | Number of evenly spaced actions the policy chooses from |
| `--num_training_epochs` | `150` | Value-iteration epochs |
| `--batch_size` | `16384` | States per value-function fit |
| `--alpha` | `1.0` | Entropy regularisation coefficient |
| `--lr` | `1e-3` | Learning rate of the gradient-based value update (unused by the default least-squares fit) |

Precedence is CLI flag > `--config_file` > default: `--env_id`, `--seed`, `--total_timesteps`, `--num_paths`, `--num_steps_per_path`, `--state_order`, `--action_order`, `--lr` and `--num_training_epochs` are read from the config file when they are not given on the command line, and fall back to the defaults above when neither sets them. The other flags always use their CLI value or default.

Run `--help` to see the full list:

```bash
uv run -m koopmanrl.soft_koopman_value_iteration --help
```

## Using a pre-optimised config

```bash
uv run python -m koopmanrl.soft_koopman_value_iteration \
    --config_file configurations/skvi_lorenz_hparams.json
```

## Source

`koopmanrl/soft_koopman_value_iteration.py`
