#!/bin/bash

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --alpha 1.5

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --decay 0.999

python dyna.py --wandb --discr_pos 0.1 --discr_vel 0.01 --decay 0.95

python dyna.py --wandb --discr_pos 0.025 --discr_vel 0.005

python dyna.py --wandb --discr_pos 0.025 --discr_vel 0.005 --k 512

python dyna.py --wandb --discr_pos 0.025 --discr_vel 0.005 --k 1024
