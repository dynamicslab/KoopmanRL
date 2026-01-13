# Koopman-Assisted Reinforcement Learning

## Getting Started

Blub

## Running Individual Experiments

### Linear Quadratic Regulator

```bash
python -m koopmanrl.linear_quadratic_regulator \
--env-id=LinearSystem-v0 \
--alpha=1 \
--total-timesteps=50000
```

### Soft Actor-Critic (Q)

```bash
python -m koopmanrl.sac_continuous_action \
--env-id=LinearSystem-v0 \
--alpha=1 \
--autotune=false \
--total-timesteps=50000
```

### Soft Actor-Critic (V)

```bash
python -m koopmanrl.value_based_sac_continuous_action \
--env-id=LinearSystem-v0 \
--alpha=1 \
--autotune=false \
--total-timesteps=50000
```

### Soft Koopman Value Iteration

```bash
python -m koopmanrl.soft_koopman_value_iteration \
--env-id=LinearSystem-v0 \
--total-timesteps=50000
```

### Soft Actor Koopman-Critic

```bash
python -m koopmanrl.soft_actor_koopman_critic \
--end-id=LinearSystem-v0 \
--total-timesteps=50000
```

