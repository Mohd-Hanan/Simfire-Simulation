#!/bin/bash
set -e
# Restore from iteration 126. We need to reach 4001, so 3875 iterations left.
# 126 + 3875 = 4001
conda run -n simfire_env env RAY_memory_monitor_refresh_ms=0 python main.py \
    --config-name train \
    cli.data_dir=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data \
    checkpoint.checkpoint_frequency=100 \
    stop_conditions.training_iteration=3875 \
    rollouts.num_rollout_workers=2 \
    resources.num_gpus=1 \
    algo.checkpoint_path=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/2026-09-22_01-05-22/checkpoints/checkpoint_100
