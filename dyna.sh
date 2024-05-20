#!/bin/bash

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --n_episodes 5_000

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --n_episodes 5_000 --init_val 0.5

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --n_episodes 5_000 --init_val 0.01

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --alpha 1.5 --n_episodes 5_000

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --decay 0.999 --n_episodes 5_000

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --decay 0.95 --n_episodes 5_000

python dyna.py --wandb --discr_pos 0.025 --discr_vel 0.005 --n_episodes 5_000

python dyna.py --wandb --discr_pos 0.025 --discr_vel 0.005 --k 512 --n_episodes 5_000

python dyna.py --wandb --discr_pos 0.025 --discr_vel 0.005 --k 1024 --n_episodes 5_000
