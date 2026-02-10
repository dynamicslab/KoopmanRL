#!/usr/bin/bash

# Test all environments for functionality
for i in LinearSystem-v0 FluidFlow-v0 Lorenz-v0 DoubleWell-v0
do
    # LQR
    uv run -m koopmanrl.linear_quadratic_regulator --env_id=$i

    # SAC (Q)
    uv run -m koopmanrl.sac_continuous_action --env_id=$i

    # SAC (V)
    uv run -m koopmanrl.value_based_sac_continuous_action --env_id=$i

    # SKVI
    uv run -m koopmanrl.soft_koopman_value_iteration --env_id=$i

    # SAKC
    uv run -m koopmanrl.soft_actor_koopman_critic --env_id=$i

    # SAKC Optimization
    uv run python -m koopmanrl.sakc_optuna_opt --num_samples=5 --env_id=$i

    # SKVI Optimization
    uv run python -m koopmanrl.skvi_optuna_opt --num_samples=5 --env_id=$i
done
