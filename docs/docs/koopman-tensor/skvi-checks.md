---
id: skvi-checks
sidebar_position: 4
title: SKVI Interpretability and Sensitivity Checks
---

# SKVI Interpretability and Sensitivity Checks

Two scripts train SKVI with the tuned configurations in `configurations/` and check what it does with the Koopman tensor against references that do not use the tensor. Their results are in the electronic supplementary material of the paper (sections "Additional validation and interpretability material" and "Accuracy along the learned policy and sensitivity of SKVI"). Both were added in [PR #15](https://github.com/dynamicslab/KoopmanRL/pull/15).

Each run trains SKVI once per seed, so a full run takes a while. Each writes one JSON file per run into a git-ignored output folder, and `--summarize_only` prints the medians over seeds from the files already there.

## Reading the policy off the tensor

```bash
uv run -m koopmanrl_utils.skvi_policy_checks                                  # both environments, seeds 100-107
uv run -m koopmanrl_utils.skvi_policy_checks --environments LinearSystem --seeds 100 101
uv run -m koopmanrl_utils.skvi_policy_checks --summarize_only --plot
```

With the polynomial action dictionary $\psi(u) = (1, u, u^2, \dots)$, the fitted continuation $w^\top K^u \phi(x)$ is a polynomial in the action whose coefficients are functions of the state, read off the slices of the tensor. The deployed Gibbs policy can therefore be written down in closed form. The script compares that reading with:

- **Linear system:** the learned quadratic form of the value function against the Riccati matrix of the discounted LQR, and the gain of the policy's mean action against the discounted LQR gain.
- **Double well:** the value function pruned in two ways (odd terms removed; diagonal quadratic only), compared with the full policy by the returns of paired rollouts and the total-variation distance; the gain of the mean action near the origin; and the closed-form Gaussian mean against the grid policy.

| Flag | Default | Description |
|------|---------|-------------|
| `--environments` | `LinearSystem DoubleWell` | Environments to check |
| `--seeds` | `100 ... 107` | One SKVI run per seed |
| `--episodes` | `10` | Paired rollout episodes per variant (double well) |
| `--horizon` | `2000` | Steps per rollout episode (double well) |
| `--output_dir` | `skvi_policy_checks_results` | One JSON file per run, `<Environment>_<seed>.json` |
| `--summarize_only` | off | Skip the runs; summarise the files in `--output_dir` |
| `--plot` | off | Also draw `linear_costate.pdf` and `double_well_costate.pdf` |

## Accuracy along the policy and sensitivity of the control

```bash
uv run -m koopmanrl_utils.skvi_sensitivity_checks --study accuracy       # four benchmarks, seeds 100-107
uv run -m koopmanrl_utils.skvi_sensitivity_checks --study sensitivity    # double well and Lorenz
uv run -m koopmanrl_utils.skvi_sensitivity_checks --study sensitivity --environments Lorenz --seeds 100 101
uv run -m koopmanrl_utils.skvi_sensitivity_checks --summarize_only
```

The tensor is identified from random-agent data, while the trained policy visits other states.

- **`--study accuracy`** (all four benchmarks) evaluates the one-step dictionary error of [the validation scripts](./validation.md#the-error-measure) on transitions of the trained SKVI policy and on fresh random-agent transitions. The scale $s_j$ is taken from the random-agent targets for both distributions so the two errors are comparable; the error with each distribution's own scale is recorded too. It also records the persistence baseline, the state error in state units, the one-step Jacobian of the fitted model at the target against that of the environment, and the distance from the target of the closed loop and of the uncontrolled system.
- **`--study sensitivity`** (double well and Lorenz) retrains SKVI with other state-dictionary orders, with the action-cost weight $R$ multiplied by the time step ($R \cdot dt$, actions a hundred times cheaper), and after re-identifying the tensor on transitions of the trained policy. Each variant is compared with no control and with the LQR controller of the linearisation at the target, on the same episodes. Only SKVI is retrained.

| Flag | Default | Description |
|------|---------|-------------|
| `--study` | `accuracy` | `accuracy` or `sensitivity` |
| `--environments` | per study | All four benchmarks for `accuracy`; `DoubleWell Lorenz` for `sensitivity` |
| `--seeds` | `100 ... 107` | One SKVI training per seed and variant |
| `--accuracy_episodes` | `16` | Closed-loop episodes sampled for the accuracy study |
| `--random_agent_paths` | `20` | Random-agent trajectories for the reference distribution |
| `--evaluation_episodes` | `10` | Closed-loop episodes per controller in the sensitivity study |
| `--variants` | all | Run only the named variants of the sensitivity study (e.g. `"order 4, refit x1"`); each variant is seeded alone, so a subset gives the same numbers as a full run |
| `--output_dir` | `skvi_sensitivity_checks_results` | One JSON file per study, environment and seed, `<study>_<Environment>_<seed>.json` |
| `--summarize_only` | off | Skip the runs; summarise the files in `--output_dir` |

Each file records the settings it was made with. A run refuses to overwrite a file made with other settings, the summary refuses to combine runs made with different settings, and unknown variant names are rejected.

The variants are defined in `VARIANTS` at the top of [`skvi_sensitivity_checks.py`](https://github.com/dynamicslab/KoopmanRL/blob/main/koopmanrl_utils/skvi_sensitivity_checks.py): for the double well, state order 2 with the benchmark $R$ and with $R \cdot dt$, and orders 4, 5 and 6 with $R \cdot dt$; for the Lorenz system, order 3 with zero or one round of re-identification and order 4 with zero, one or two, plus two order-3 variants that separate refitting the tensor from adding the policy's states to the value fit.
