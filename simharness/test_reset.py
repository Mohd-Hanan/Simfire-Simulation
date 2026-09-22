import hydra
from omegaconf import DictConfig
import ray
from ray.rllib.algorithms.algorithm import Algorithm

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    ray.init(ignore_reinit_error=True)
    import sys
    sys.path.append("/home/hari/PycharmProjects/Simfire-Simulation/simharness")
    from main import _build_algo_cfg
    
    algo_cfg = _build_algo_cfg(cfg)
    algo_cfg.environment(disable_env_checking=True) # disable env checking
    algo = algo_cfg.build()
    
    env = algo.workers.local_worker().env
    print("Does bench analytics exist?", env.harness_analytics.benchmark_sim_analytics is not None)
    
    obs, info = env.reset()
    
    print("Length of damaged:", len(env.harness_analytics.benchmark_sim_analytics.data.damaged))
    print("Is benchmark_sim active?", env.benchmark_sim.active)
    print("Elapsed steps:", env.benchmark_sim.elapsed_steps)
    
    ray.shutdown()

if __name__ == "__main__":
    main()
