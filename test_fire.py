import numpy as np
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config = Config("simharness/conf/simulation/simfire/default.yaml")
sim = FireSimulation(config)
for i in range(20):
    sim.run(1)
    print(f"Step {i+1}: Burning {np.sum(sim.fire_map == 1)}, Burned {np.sum(sim.fire_map == 2)}")

config.yaml_data["fire"]["max_fire_duration"] = 500
sim2 = FireSimulation(config)
print("--- With max_fire_duration=500 ---")
for i in range(20):
    sim2.run(1)
    print(f"Step {i+1}: Burning {np.sum(sim2.fire_map == 1)}, Burned {np.sum(sim2.fire_map == 2)}")
