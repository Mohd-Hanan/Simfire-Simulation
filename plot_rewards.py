import json
import numpy as np
import math

def extract_rewards(json_path):
    rewards = []
    iters = []
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                r = data.get('episode_reward_mean')
                if r is not None and not math.isnan(r):
                    rewards.append(r)
                    iters.append(data.get('training_iteration', 0))
            except:
                pass
    return iters, rewards

i2, r2 = extract_rewards('/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/2026-09-21_06-56-05/result.json')

if len(r2) > 0:
    print(f"128x128: Start Reward: {np.mean(r2[:10]):.2f}, End Reward: {np.mean(r2[-10:]):.2f}, Trend: {r2[::50]}")
else:
    print("No valid rewards for 128x128 in this file.")
