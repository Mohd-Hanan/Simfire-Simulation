#!/bin/bash
set -e
# Restore from checkpoint_10 of 00-10-59 (which is iteration 20).
# Run 5 iterations to reach 25.
CMD="conda run -n simfire_env env RAY_memory_monitor_refresh_ms=0 python main.py --config-name train cli.data_dir=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data checkpoint.checkpoint_frequency=5 stop_conditions.training_iteration=5 rollouts.num_rollout_workers=2 resources.num_gpus=1 algo.checkpoint_path=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/2026-09-22_00-10-59/checkpoints/checkpoint_10"
eval "$CMD"
