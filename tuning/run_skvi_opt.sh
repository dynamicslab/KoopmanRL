#!/usr/bin/bash

# Activate the virtual env
source /home/lpaehler/Work/ReinforcementLearning/KoopmanRLLaptop/KoopmanRL/BayesianOptimization/_optimization_env/bin/activate

# Run the HPO for the Linear System
echo "Starting the Hyperparameter Optimization for the Linear System!"
nohup python skvi_optuna_opt.py \
 --env_id LinearSystem-v0 \
 --output_file skvi_linear_system_hparams > linear_system_opt.txt &
wait
echo "Finished running the Hyperparameter Optimization for the Linear System!"

# Run the HPO for the Fluid Flow
echo "Starting the Hyperparameter Optimization for the Fluid Flow!"
nohup python skvi_optuna_opt.py \
 --env_id FluidFlow-v0 \
 --output_file skvi_fluid_flow_hparams > fluid_flow_opt.txt &
wait
echo "Finished running the Hyperparameter Optimization for Fluid Flow!"

# Run the HPO for Lorenz
echo "Starting the Hyperparameter Optimization for the Lorenz!"
nohup python skvi_optuna_opt.py \
 --env_id Lorenz-v0 \
 --output_file skvi_lorenz_hparams > lorenz_opt.txt &
wait
echo "Finished running the Hyperparameter Optimization for the Lorenz System!"

# Run the HPO for the Stochastic Double Well
echo "Starting the Hyperparameter Optimization for the Double Well!"
nohup python skvi_optuna_opt.py \
 --env_id DoubleWell-v0 \
 --output_file skvi_double_well_hparams > double_well_opt.txt &
wait
echo "Finished running the Hyperparameter Optimization for the Double Well!"
