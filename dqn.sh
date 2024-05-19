#!/bin/bash

######### Baseline
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# No reward

python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.0 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

## Machine learning parameters

# Importance replay
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 0.0 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# Batch size
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# Learning rate
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# Hidden size
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# Amsgrad
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb 

# Target network
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --wandb --amsgrad 

# Dropout rate
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0.1 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# Regularization
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0.0 --weight_decay 0.001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 


########## Heuristic parameters

# Balance between velocity and position
python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --w_velocity 0.5 --w_position 0.5

python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --w_velocity 0.9 --w_position 0.1

python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --w_velocity 0.1 --w_position 0.9

# Reward factor

python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 5e-4 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 2 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 


######## Best

# python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 

# python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 256 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 0.999 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad 