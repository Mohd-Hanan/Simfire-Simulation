import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    ray.init(ignore_reinit_error=True)
    
    # We will just use the main.py logic to build the algo
    import sys
    sys.path.append("/home/hari/PycharmProjects/Simfire-Simulation/simharness")
    from main import _build_algo_cfg
    
    algo_cfg = _build_algo_cfg(cfg)
    algo = algo_cfg.build()
    
    policy = algo.get_policy()
    model = policy.model
    device = next(model.parameters()).device
    
    # 1. Before Update
    print("=== BEFORE FIRST PPO UPDATE ===")
    weights_before = policy.get_weights()
    max_w_before = max([np.max(np.abs(w)) for w in weights_before.values()])
    has_nan_before = any([np.isnan(w).any() or np.isinf(w).any() for w in weights_before.values()])
    print(f"Max Weight Magnitude: {max_w_before:.4f}")
    print(f"Has NaN/Inf Weights: {has_nan_before}")
    
    obs_space = policy.observation_space
    # Generate a dummy observation simulating unnormalized values
    # e.g., elevation at 3000, wind_direction at 3000
    dummy_obs = np.zeros(obs_space.shape, dtype=np.float32)
    dummy_obs[:, :, 0] = 5.0      # fire map
    dummy_obs[:, :, 1] = 3000.0   # elevation
    dummy_obs[:, :, 2] = 0.5      # wind speed
    dummy_obs[:, :, 3] = 3000.0   # wind direction
    dummy_obs[:, :, 4] = 1.0      # custom fire map
    dummy_obs[:, :, 5] = 0.5      # mitigation
    
    obs_tensor = torch.tensor(np.expand_dims(dummy_obs, 0)).to(device)
    
    with torch.no_grad():
        logits_before, _ = model({"obs": obs_tensor})
        vf_before = model.value_function()
    
    print(f"Logits (Sample): {logits_before[0, :5].cpu().numpy()}")
    print(f"Logits Min/Max: {torch.min(logits_before).item():.4f} / {torch.max(logits_before).item():.4f}")
    print(f"Value Output: {vf_before.item():.4f}")
    
    # 2. Train (Rollout + 1 PPO Update)
    print("\n=== RUNNING 1 PPO UPDATE ===")
    result = algo.train()
    
    mean_reward = result.get("episode_reward_mean", "N/A")
    # Advantage isn't directly in result usually, but we can print loss metrics
    learner_stats = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
    policy_loss = learner_stats.get("policy_loss", "N/A")
    vf_loss = learner_stats.get("vf_loss", "N/A")
    entropy = learner_stats.get("entropy", "N/A")
    
    print(f"Episode Reward Mean: {mean_reward}")
    print(f"Policy Loss: {policy_loss}")
    print(f"Value Loss: {vf_loss}")
    print(f"Entropy: {entropy}")
    
    # 3. After Update
    print("\n=== AFTER FIRST PPO UPDATE ===")
    weights_after = policy.get_weights()
    max_w_after = max([np.max(np.abs(w)) for w in weights_after.values()])
    has_nan_after = any([np.isnan(w).any() or np.isinf(w).any() for w in weights_after.values()])
    print(f"Max Weight Magnitude: {max_w_after:.4f}")
    print(f"Has NaN/Inf Weights: {has_nan_after}")
    
    with torch.no_grad():
        logits_after, _ = model({"obs": obs_tensor})
        vf_after = model.value_function()
        
    print(f"Logits (Sample): {logits_after[0, :5].cpu().numpy()}")
    print(f"Logits Min/Max: {torch.min(logits_after).item():.4f} / {torch.max(logits_after).item():.4f}")
    print(f"Value Output: {vf_after.item():.4f}")
    
    ray.shutdown()

if __name__ == "__main__":
    main()
