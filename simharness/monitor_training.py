import json
import time
import os
import glob
import re

ARTIFACT_PATH = "/home/hari/.gemini/antigravity/brain/03666627-d24e-4417-8dc9-325260f3fbc9/training_progress.md"
EXPERIMENTS_DIR = "/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments"
LOG_FILE = "/home/hari/PycharmProjects/Simfire-Simulation/simharness/train_overnight.log"

def get_latest_experiment():
    dirs = glob.glob(os.path.join(EXPERIMENTS_DIR, "2026-*"))
    if not dirs: return None
    return max(dirs, key=os.path.getmtime)

def update_artifact(lines_to_write):
    header = "# Overnight Training Progress (Target: 4001 iterations)\n\n"
    header += "| Iteration | Reward | Entropy | Policy Loss | VF Loss | Episode Length | Avg Damaged / Saved | Mean Action |\n"
    header += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    with open(ARTIFACT_PATH, 'w') as f:
        f.write(header)
        for l in lines_to_write:
            f.write(l + "\n")

def run():
    print("Starting monitor...")
    # Wait for the new experiment directory
    time.sleep(10)
    exp_dir = get_latest_experiment()
    result_path = os.path.join(exp_dir, "result.json")
    print(f"Monitoring {result_path}")

    logged_iterations = set()
    lines_to_write = []

    update_artifact(lines_to_write)

    while True:
        try:
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result_lines = f.readlines()
                
                for line in result_lines:
                    d = json.loads(line)
                    iter_num = d.get('training_iteration')
                    if iter_num % 100 == 0 and iter_num not in logged_iterations:
                        reward = d.get("episode_reward_mean", 0)
                        entropy = d["info"]["learner"]["default_policy"]["learner_stats"]["entropy"]
                        policy_loss = d["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]
                        vf_loss = d["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"]
                        ep_len = d.get("episode_len_mean", 0)
                        
                        # Grep log file for last 20 max_unburned
                        avg_damaged = "N/A"
                        avg_saved = "N/A"
                        mean_action = "N/A"
                        
                        if os.path.exists(LOG_FILE):
                            with open(LOG_FILE, 'r') as lf:
                                content = lf.read()
                                unburned = re.findall(r"max_unburned=(\d+)", content)
                                if unburned:
                                    last_unburned = [int(x) for x in unburned[-20:]]
                                    avg_saved = sum(last_unburned) / len(last_unburned)
                                    avg_damaged = 1024 - avg_saved
                                
                                actions = re.findall(r"'actions': np\.ndarray.*?, mean=([\d\.]+)\)", content)
                                if actions:
                                    mean_action = actions[-1]

                        row = f"| **{iter_num}** | {reward:.2f} | {entropy:.3f} | {policy_loss:.3f} | {vf_loss:.3f} | {ep_len} | ~{avg_damaged} / {avg_saved} | {mean_action} |"
                        lines_to_write.append(row)
                        logged_iterations.add(iter_num)
                        update_artifact(lines_to_write)
                        print(f"Logged iteration {iter_num}")
                        
                        if iter_num >= 4001:
                            print("Reached target iteration 4001. Stopping monitor.")
                            return
        except Exception as e:
            print(f"Error: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    run()
