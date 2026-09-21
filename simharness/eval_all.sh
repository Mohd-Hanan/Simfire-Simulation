#!/bin/bash
set -e
cd /home/hari/PycharmProjects/Simfire-Simulation/simharness

LATEST_CKPT=$(ls -vd /home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/*/checkpoints/checkpoint_* 2>/dev/null | tail -n 1)

echo "Evaluating: $LATEST_CKPT"
conda run -n simfire_env python main.py --config-name watch cli.data_dir=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data algo.checkpoint_path=$LATEST_CKPT evaluation.evaluation_duration=8 simulation.fire_initial_position.sampler.sample_size.eval=8
