
import numpy as np
import copy
import matplotlib.pyplot as plt
import time as time_module
import random

from multiprocessing import Pool, cpu_count

from agents.paired_state_agent import DQNAgent
from environments.grid_environment import GridEnvironment
from utils.experience_db import ExperienceDB
from agents.model_trainer import train_model
from agents.swap_scheduler import schedule_swaps
from utils.circuit_tools import generate_completely_random_circuit
from utils.realistic_test_set_tools import import_test_set

training_episodes = 100
should_train = True

test_set_circuits = import_test_set()

def train_model_on_random_circuits(model_number):
    model_name = "random_circuits_" + str(model_number)

    training_circuit_generation_function = lambda: generate_completely_random_circuit(16, 50).to_dqn_rep()

    environment = GridEnvironment(4,4,training_circuit_generation_function())
    agent = DQNAgent(environment)

    train_model(environment, agent, training_episodes=training_episodes, circuit_generation_function=training_circuit_generation_function, should_print=False)
    agent.save_model(model_name)

def perform_run(initial_locations, model_number):
    model_name = "random_circuits_" + str(model_number)

    start_time = time_module.clock()

    environment = GridEnvironment(4,4,test_set_circuits[0].to_dqn_rep())
    agent = DQNAgent(environment)
    agent.load_model(model_name)

    average_test_time = 0.0
    average_circuit_depth_overhead = 0.0
    average_circuit_depth_ratio = 0.0

    test_episodes = len(test_set_circuits)

    for e in range(test_episodes):
        circuit = test_set_circuits[e]
        qubit_locations = initial_locations[e]
        original_depth = circuit.depth()

        actions, circuit_depth = schedule_swaps(environment, agent, circuit=circuit, experience_db=None, qubit_locations=qubit_locations, safety_checks_on=True)
        average_test_time += (1.0/test_episodes) * len(actions)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - original_depth)
        average_circuit_depth_ratio += (1.0/test_episodes) * (float(circuit_depth)/float(original_depth))

    end_time = time_module.clock()

    total_time = end_time-start_time

    result = (model_number, average_test_time, average_circuit_depth_overhead, average_circuit_depth_ratio, total_time)

    print('Completed run:', result)

    return result

repeats = 5

random.seed(343)

initial_locations_sets = []

for _ in range(repeats):
    initial_locations = []

    for _ in range(len(test_set_circuits)):
        qubit_locations = list(range(16))
        random.shuffle(qubit_locations)
        initial_locations.append(qubit_locations)

    initial_locations_sets.append(initial_locations)

inputs = []

for i in range(repeats):
    for l in initial_locations_sets:
        inputs.append((l,i))

random.shuffle(inputs)

if __name__ == '__main__':
    p = Pool(cpu_count())

    if should_train:
        model_numbers = list(range(0,repeats))
        p.map(train_model_on_random_circuits, model_numbers)

    results = p.starmap(perform_run, inputs)

    results.sort(key=lambda r: r[0])

    print()
    for r in results:
        print(r)
    print()

    average_depth_ratios = {k: 0 for k in list(range(repeats))}

    for (model_number, test_time, depth_overhead, depth_ratio, total_time) in results:
        average_depth_ratios[model_number] += (1.0/repeats) * depth_ratio

    print(average_depth_ratios)
