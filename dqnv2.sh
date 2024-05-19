#!/bin/bash

# Baseline

python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 3 --running_window 1000 --predictor_learning_rate 1e-3 --env_reward 

python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 3 --running_window 1000 --predictor_learning_rate 1e-3

# Reward factor

python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 5 --running_window 1000 --predictor_learning_rate 1e-3 --env_reward

python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 2 --running_window 1000 --predictor_learning_rate 1e-3 --env_reward

# running window
python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 3 --running_window 100 --predictor_learning_rate 1e-3 --env_reward

python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 1.5 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 3 --running_window 10 --predictor_learning_rate 1e-3 --env_reward

# alpha

python dqnv2.py --seed 0 --n_episodes 3_000 --start_epsilon 0.9 --learning_rate 1e-3 --batch_size 128 --discount_factor 0.99 --replay_size 10_000 --hidden_size 128 --dropout_rate 0 --weight_decay 0.0001 --alpha 0 --final_epsilon 0.05 --epsilon_decay 0.95 --target_network --wandb --amsgrad --reward_hidden_size 128 --reward_factor 3 --running_window 1000 --predictor_learning_rate 1e-3 --env_reward