import pickle
import glob
import os

for path in glob.glob("/home/hari/PycharmProjects/Simfire-Simulation/simharness_data/simharness/experiments/**/algorithm_state.pkl", recursive=True):
    try:
        state = pickle.load(open(path, 'rb'))
        iteration = state.get('training_iteration')
        if iteration > 1000:
            print(f"Iteration: {iteration} Path: {path}")
    except Exception as e:
        pass
