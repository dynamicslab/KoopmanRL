---
id: installation
sidebar_position: 2
title: Installation
---

# Installation

KoopmanRL requires **Python 3.10**. There are two recommended installation paths.

## Option 1 — uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the full project environment from the lock file, guaranteeing exact reproducibility.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/dynamicslab/KoopmanRL.git
cd KoopmanRL
uv sync
```

All scripts are then run with the `uv run` prefix:

```bash
uv run -m koopmanrl.soft_koopman_value_iteration
```

## Option 2 — Virtual environment + pip

```bash
git clone https://github.com/dynamicslab/KoopmanRL.git
cd KoopmanRL

python3.10 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install .
```

After activating the environment, scripts are run directly:

```bash
python -m koopmanrl.soft_koopman_value_iteration
```

## Development install

To install with development dependencies (testing, linting):

```bash
# uv
uv sync --group dev

# pip
pip install ".[dev]"
```

## Verifying the installation

```bash
uv run python -c "import koopmanrl; print('KoopmanRL installed successfully')"
```

Or run the test suite:

```bash
uv run pytest
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.9` | Neural network training |
| `gym == 0.23.1` | RL environment interface |
| `cleanrl >= 1.2` | CleanRL RL algorithm implementations |
| `control >= 0.10` | Linear control theory utilities |
| `optuna >= 3.0` | Hyperparameter optimisation |
| `ray[tune] >= 2.53` | Distributed hyperparameter search |
| `scipy >= 1.15` | Scientific computing |
| `stable-baselines3 == 1.2` | Baseline RL algorithms |
| `tensorboard >= 2.20` | Experiment logging |
| `numpy >= 2.2` | Numerical arrays |
| `matplotlib >= 3.10` | Plotting |
