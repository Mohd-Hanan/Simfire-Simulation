#!/bin/bash
set -e

# Run with 2 rollout workers. Target is 3976 more iterations (to reach 4001 from 25).
CMD="conda run -n simfire_env env RAY_memory_monitor_refresh_ms=0 python main.py --config-name train cli.data_dir=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data checkpoint.checkpoint_frequency=100 stop_conditions.training_iteration=3976 rollouts.num_rollout_workers=2 resources.num_gpus=1 algo.checkpoint_path=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/2026-09-22_00-55-13/checkpoints/checkpoint_0"

echo "Running command: $CMD"
eval "$CMD"
