---
id: validation
sidebar_position: 2
title: Validating the Tensor
---

# Validating the Tensor

Two scripts measure how well the Koopman tensor predicts on held-out random-agent data. Both identify the tensor with the tuned SKVI dictionary orders and identification budgets of `configurations/skvi_<benchmark>_hparams.json`, and report results across seeds. The results are in the paper (section "Evaluation") and its electronic supplementary material.

## The error measure

SKVI and SAKC use the tensor only through the one-step prediction $K^u \phi(x)$ of the dictionary; neither iterates it nor decodes it back to the state. The one-step error is therefore measured in dictionary space, as the root mean square over held-out transitions $n$ and non-constant dictionary functions $j$ of

$$
\frac{\big(K^{u_n}\phi(x_n) - \mathbb{E}[\phi(x'_n) \mid x_n, u_n]\big)_j}{s_j},
$$

where $s_j$ is the root mean square of the target for function $j$, so that high-order monomials do not dominate. The conditional mean is $\phi(x'_n)$ for the deterministic benchmarks; for the stochastic double well it is computed over the one-step noise. Because the time step is small, the same error of the persistence prediction $\phi(x_n)$ ("no change") is reported alongside.

## Prediction validation

```bash
uv run -m koopmanrl_utils.koopman_prediction_validation                       # four benchmarks, seeds 123-130
uv run -m koopmanrl_utils.koopman_prediction_validation --seeds 2 --seed0 0   # two seeds starting at 0
```

The tensor is fitted on 80% of the trajectories and evaluated on the other 20%, split by whole trajectory. Results are medians and quartiles across seeds.

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds` | `8` | Number of seeds |
| `--seed0` | `123` | First seed; seeds are `seed0`, `seed0 + 1`, ... |
| `--output_dir` | `koopman_prediction_validation_results` | Output folder (git-ignored) |

| Output | Content |
|--------|---------|
| `onestep_error.csv`, `onestep_by_degree.csv`, `onestep_by_degree_long.csv`, `onestep_table.tex` | One-step error, persistence baseline and their ratio; error per monomial degree; a LaTeX table. The decoded-state error is kept for reference. |
| `multistep_<label>.csv` | State error along fresh trajectories against the horizon (up to 500 steps), rolling the model forward in the lifted space and with re-projection after every step. A diagnostic only: the algorithms never roll the model forward. |
| `train_vs_heldout_<label>.csv` | Training and held-out one-step error against the number of identification transitions, on nested subsamples. |

`<label>` is `Linear`, `Lorenz`, `FluidFlow` or `DoubleWell`. A run is reproducible from its seed up to floating-point rounding (differences of order 1e-12, which reach the leading digits only on the linear system, whose errors are at machine precision).

## Regressor comparison

```bash
uv run -m koopmanrl_utils.koopman_regressor_comparison                  # both studies, about 40 minutes on one core
uv run -m koopmanrl_utils.koopman_regressor_comparison --study accuracy
uv run -m koopmanrl_utils.koopman_regressor_comparison --study budget --environments DoubleWell
uv run -m koopmanrl_utils.koopman_regressor_comparison --regressors ols lasso_cv --accuracy_seeds 2
uv run -m koopmanrl_utils.koopman_regressor_comparison --summarize_only # rewrite tables and figures
```

Compares thirteen regressors on held-out transitions, reported relative to the persistence predictor (1 means "no better than predicting no change"):

- **As shipped:** `ols` (the package's torch `KoopmanTensor` class), `ols_numpy` (the NumPy class, which solved the normal equations), and `ridge`, `sindy`, `rrr` as they were when the results of the paper were produced (frozen copies of the code up to commit `b06d974`).
- **Alternatives:** `ols_scaled`, `tsvd`, `ridge_cv`, `stlsq_cv`, `lasso_cv`, `rrr_cv`, `tls` and `huber`. They use the increment target and column standardisation described in [Building a tensor](./building.md#regressors), with hyperparameters chosen on the last 20% of the identification transitions.

| Study | What it measures | Outputs |
|-------|------------------|---------|
| `accuracy` | Every regressor on every benchmark at the tuned budget, 12 seeds | `accuracy.json`, `accuracy_<benchmark>.dat`, `accuracy_seeds_<benchmark>.dat`, `accuracy_offscale_<benchmark>.dat`, `accuracy_table.csv`, `regressor_accuracy.tikz` |
| `budget` | The same error against the identification budget (1%–100% of the tuned budget), 8 seeds | `budget.json`, `budget_<benchmark>.dat`, `regressor_budget.tikz` |

| Flag | Default | Description |
|------|---------|-------------|
| `--study` | `all` | `accuracy`, `budget` or `all` |
| `--environments` | `Linear Lorenz FluidFlow DoubleWell` | Benchmarks, by label |
| `--regressors` | all thirteen | Regressors to run |
| `--accuracy_seeds` | `12` | Seeds of the accuracy study |
| `--budget_seeds` | `8` | Seeds of the budget study |
| `--fractions` | `0.01 0.02 0.05 0.1 0.2 0.5 1.0` | Fractions of the tuned budget |
| `--output_dir` | `koopman_regressor_comparison_results` | Output folder (git-ignored) |
| `--summarize_only` | off | Rewrite tables and figures from the JSON files in `--output_dir` |

The held-out transitions are the same for every seed and regressor, and the matrix $A$ of the linear system is fixed, so the seeds of the linear system are data sets of one system. The `.tikz` files are pgfplots figures; compile the standalone wrappers with `pdflatex regressor_accuracy.tex` inside the output directory.

Both scripts use the `ols` of `koopmanrl/koopman_tensor/torch_tensor.py`, which keeps the default least-squares driver, so their results differ in the last digits from call to call; see [Least-squares driver](./least-squares-driver.md).
