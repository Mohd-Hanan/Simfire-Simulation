import ray
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
import torch

ray.init(ignore_reinit_error=True)

ckpt_path = "/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/2026-09-21_22-04-04/checkpoints/checkpoint_0"
algo = Algorithm.from_checkpoint(ckpt_path)

policy = algo.get_policy()
model = policy.model

obs_space = policy.observation_space
print("Observation Space Lows (sample):\n", obs_space.low[0,0,:])
print("Observation Space Highs (sample):\n", obs_space.high[0,0,:])

# Test raw unnormalized max bounds
obs_max = np.broadcast_to(obs_space.high[0,0,:], (32, 32, 6))
obs_batch_max = torch.tensor(np.expand_dims(obs_max, 0).astype(np.float32))

print("\n--- Model Output at Upper Bounds ---")
out_max, _ = model({"obs": obs_batch_max})
print("Logits min/max:", torch.min(out_max).item(), torch.max(out_max).item())

# Test raw unnormalized min bounds
obs_min = np.broadcast_to(obs_space.low[0,0,:], (32, 32, 6))
obs_batch_min = torch.tensor(np.expand_dims(obs_min, 0).astype(np.float32))
print("\n--- Model Output at Lower Bounds ---")
out_min, _ = model({"obs": obs_batch_min})
print("Logits min/max:", torch.min(out_min).item(), torch.max(out_min).item())

weights = policy.get_weights()
max_weight = max([np.max(np.abs(w)) for w in weights.values()])
has_nan = any([np.isnan(w).any() for w in weights.values()])
print(f"\nMax weight magnitude in checkpoint: {max_weight}")
print(f"Has NaN weights: {has_nan}")

ray.shutdown()
