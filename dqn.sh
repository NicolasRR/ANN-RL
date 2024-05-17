#!/bin/bash

python dqn.py --seed 0 --n_episodes 10_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 1 --final_reward 200 --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --wandb --amsgrad 

python dqn.py --seed 0 --n_episodes 10_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 1 --final_reward 200 --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --wandb --amsgrad


python dqn.py --seed 0 --n_episodes 10_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0.1 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 1 --final_reward 200 --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --wandb --amsgrad


python dqn.py --seed 0 --n_episodes 10_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0 --weight_decay 0.0001 --alpha 0 --intermediate_reward 1 --final_reward 200 --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --wandb --amsgrad

python dqn.py --seed 0 --n_episodes 10_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 64 --discount_factor 0.99 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0.1 --weight_decay 0.0001 --alpha 1.5 --intermediate_reward 1 --final_reward 200 --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --wandb --amsgrad

python dqn.py --seed 0 --n_episodes 10_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --logging_interval 10 --hidden_size 64 --dropout_rate 0 --weight_decay 0.001 --alpha 1.5 --intermediate_reward 1 --final_reward 200 --auxiliary --final_epsilon 0.05 --epsilon_decay 0.99 --target_network --wandb --amsgrad