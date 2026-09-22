import os
import glob
import pickle
import subprocess
import sys

EXPERIMENTS_DIR = "/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments"
TARGET_ITER = 4001

def get_latest_checkpoint():
    exp_dirs = glob.glob(os.path.join(EXPERIMENTS_DIR, "2026-*"))
    exp_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for exp_dir in exp_dirs:
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_*"))
            if ckpts:
                # Sort checkpoints by modification time
                ckpts.sort(key=os.path.getmtime, reverse=True)
                for ckpt in ckpts:
                    state_file = os.path.join(ckpt, "algorithm_state.pkl")
                    if os.path.exists(state_file):
                        try:
                            with open(state_file, 'rb') as f:
                                state = pickle.load(f)
                            iter_num = state.get("training_iteration", 0)
                            return ckpt, iter_num
                        except Exception as e:
                            print(f"Error loading {state_file}: {e}")
                            continue
    return None, 0

latest_ckpt, current_iter = get_latest_checkpoint()
print(f"Latest checkpoint found: {latest_ckpt} at iteration {current_iter}")

if current_iter >= TARGET_ITER:
    print(f"Already reached target iteration {TARGET_ITER}. Exiting.")
    sys.exit(0)

if not latest_ckpt:
    print("Could not find any checkpoint!")
    sys.exit(1)

iters_left = TARGET_ITER - current_iter
print(f"Iterations left: {iters_left}")

cmd = [
    "conda", "run", "-n", "simfire_env", "env", "RAY_memory_monitor_refresh_ms=0",
    "python", "main.py", "--config-name", "train",
    "cli.data_dir=/home/hari/PycharmProjects/Simfire-Simulation/simharness_data",
    "checkpoint.checkpoint_frequency=100",
    f"stop_conditions.training_iteration={iters_left}",
    "rollouts.num_rollout_workers=2",
    "resources.num_gpus=1",
    f"algo.checkpoint_path={latest_ckpt}"
]

print("Launching:", " ".join(cmd))
with open("train_overnight.log", "w") as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
