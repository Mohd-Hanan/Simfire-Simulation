import requests.models
import os

# Save the original json method
original_json = requests.models.Response.json

# Patch it to print the raw response if it fails
def patched_json(self, **kwargs):
    try:
        return original_json(self, **kwargs)
    except Exception as e:
        print(f"\n--- API ERROR DIAGNOSTIC ---")
        print(f"Status Code: {self.status_code}")
        print(f"URL: {self.url}")
        print(f"Raw Response Text: {self.text[:500]}...") # Print first 500 chars
        print(f"----------------------------\n")
        raise e

requests.models.Response.json = patched_json

from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config_path = "/home/hanan/Documents/rl_project/simfire/configs/operational_config.yml"
config = Config(config_path)

# This will trigger the Landfire API call
sim = FireSimulation(config)
