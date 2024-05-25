#!/bin/bash

# DQN with no auxiliary reward or with heuristic auxiliary reward
bash dqn.sh

# DQN with intrinsic reward
bash dqnv2.sh

# Dyna
bash dyna.sh

## Comparison

python comparison.py --n_episodes 3_000 --snapshot_interval 200 

python comparison.py --n_episodes 3_000 --snapshot_interval 200 --target_network