#!/bin/bash

python dqn.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.95 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0.1 --weight_decay 0.0001 --alpha 0 --intermediate_reward 1 --final_reward 100 --wandb --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99

python dqn.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.95 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0.1 --weight_decay 0.0001 --alpha 0 --intermediate_reward 1 --final_reward 100 --wandb --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --target_network_update 10_000

# python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.95 --replay_size 10_000 --logging_interval 10 --hidden_size 128 --dropout_rate 0.1 --weight_decay 0.01 --target_network_update 100 --alpha 0 --auxiliary False --intermediate_reward 0 --final_reward 0 --wandb

# python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.95 --replay_size 10_000 --logging_interval 10 --hidden_size 128 --dropout_rate 0.1 --weight_decay 0.01 --target_network_update 100 --alpha 1.5 --auxiliary False --intermediate_reward 0 --final_reward 0 --wandb

# python dqn.py --seed 0 --n_episodes 5_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.95 --replay_size 10_000 --logging_interval 10 --hidden_size 128 --dropout_rate 0.1 --weight_decay 0.01 --target_network_update 100 --alpha 0 --auxiliary True --intermediate_reward 1 --final_reward 50 --wandb

