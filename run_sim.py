import os
from simfire.sim.simulation import Fire
from simfire.utils.config import Config

def run_test():
    # Load configuration (using functional config to bypass the internet API error)
    config_path = os.path.join(os.path.dirname(__file__), "simfire/configs/functional_config.yml")
    print(f"Loading config from {config_path}")
    
    config = Config(config_path)
    sim = FireSimulation(config)
    
    # Enable the visual PyGame window
    sim.rendering = True
    
    # Run a 12-hour simulation (this will take longer and let you watch the fire spread!)
    print("Running visual simulation for 12h...")
    sim.run("12h")
    print("Simulation finished successfully.")

if __name__ == "__main__":
    run_test()
