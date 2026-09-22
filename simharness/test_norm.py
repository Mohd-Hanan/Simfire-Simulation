import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from simharness2.environments.reactive import ReactiveHarness
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    ray.init(ignore_reinit_error=True)
    
    import sys
    sys.path.append("/home/hari/PycharmProjects/Simfire-Simulation/simharness")
    from main import _build_algo_cfg
    
    algo_cfg = _build_algo_cfg(cfg)
    algo = algo_cfg.build()
    
    policy = algo.get_policy()
    model = policy.model
    device = next(model.parameters()).device
    
    obs_space = policy.observation_space
    print("Observation Space Lows:", obs_space.low[0,0,:])
    print("Observation Space Highs:", obs_space.high[0,0,:])
    
    # Generate random obs
    dummy_obs = np.random.uniform(low=obs_space.low, high=obs_space.high)
    print("Dummy obs shape:", dummy_obs.shape)
    print("Dummy obs mean per channel:", dummy_obs.mean(axis=(0,1)))
    
    obs_tensor = torch.tensor(np.expand_dims(dummy_obs, 0)).to(device)
    
    with torch.no_grad():
        logits_before, _ = model({"obs": obs_tensor})
        vf_before = model.value_function()
    
    print(f"Logits Min/Max: {torch.min(logits_before).item():.4f} / {torch.max(logits_before).item():.4f}")
    
    print("\n=== RUNNING 1 PPO UPDATE ===")
    result = algo.train()
    
    learner_stats = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
    policy_loss = learner_stats.get("policy_loss", "N/A")
    vf_loss = learner_stats.get("vf_loss", "N/A")
    entropy = learner_stats.get("entropy", "N/A")
    
    print(f"Policy Loss: {policy_loss}")
    print(f"Entropy: {entropy}")
    
    # 3. After Update
    print("\n=== AFTER FIRST PPO UPDATE ===")
    with torch.no_grad():
        logits_after, _ = model({"obs": obs_tensor})
        
    print(f"Logits Min/Max: {torch.min(logits_after).item():.4f} / {torch.max(logits_after).item():.4f}")
    
    ray.shutdown()

if __name__ == "__main__":
    main()
