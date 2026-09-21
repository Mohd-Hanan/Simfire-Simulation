#!/bin/bash
set -e

# Find the absolute latest checkpoint across ALL experiments
LATEST_CKPT=$(ls -vd /home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/*/checkpoints/checkpoint_* 2>/dev/null | tail -n 1)

# Run with 2 rollout workers for max speed! Target iteration 1550.
CMD="conda run -n simfire_env env RAY_memory_monitor_refresh_ms=0 python main.py --config-name train cli.data_dir=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data checkpoint.checkpoint_frequency=20 stop_conditions.training_iteration=1550 rollouts.num_rollout_workers=2 resources.num_gpus=1"

if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming training from latest checkpoint: $LATEST_CKPT"
    CMD="$CMD algo.checkpoint_path=$LATEST_CKPT"
else
    echo "No existing checkpoints found. Starting fresh training..."
fi

echo "Running command: $CMD"
eval "$CMD"
