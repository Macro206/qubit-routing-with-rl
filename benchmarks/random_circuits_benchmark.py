
import numpy as np
import copy
import matplotlib.pyplot as plt
import time as time_module
import random

from multiprocessing import Pool, cpu_count

from agents.paired_state_agent import DQNAgent
from environments.ibm_q20_tokyo import IBMQ20Tokyo
from environments.rigetti_19q_acorn import Rigetti19QAcorn
from utils.experience_db import ExperienceDB
from agents.model_trainer import train_model
from agents.swap_scheduler import schedule_swaps
from utils.circuit_tools import generate_completely_random_circuit

training_episodes = 100
test_episodes = 100
should_train = True

def train_model_on_random_circuits(model_number):
    model_name = "random_circuits_" + str(model_number)

    training_circuit_generation_function = lambda: generate_completely_random_circuit(20, 50).to_dqn_rep()

    environment = IBMQ20Tokyo(training_circuit_generation_function())
    agent = DQNAgent(environment)

    train_model(environment, agent, training_episodes=training_episodes, circuit_generation_function=training_circuit_generation_function, should_print=False)
    agent.save_model(model_name)

def perform_run(n_gates, model_number):
    model_name = "random_circuits_" + str(model_number)

    start_time = time_module.clock()

    test_circuit_generation_function = lambda: generate_completely_random_circuit(20, n_gates)

    environment = IBMQ20Tokyo(test_circuit_generation_function().to_dqn_rep())
    agent = DQNAgent(environment)
    agent.load_model(model_name)

    average_test_time = 0.0
    average_circuit_depth_overhead = 0.0
    average_circuit_depth_ratio = 0.0

    for e in range(test_episodes):
        circuit = test_circuit_generation_function()
        original_depth = circuit.depth()

        actions, circuit_depth = schedule_swaps(environment, agent, circuit=circuit.to_dqn_rep(), experience_db=None)
        average_test_time += (1.0/test_episodes) * len(actions)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - original_depth)
        average_circuit_depth_ratio += (1.0/test_episodes) * (float(circuit_depth)/float(original_depth))

    end_time = time_module.clock()

    total_time = end_time-start_time

    result = (n_gates, average_test_time, average_circuit_depth_overhead, average_circuit_depth_ratio, total_time)

    print('Completed run:', result)

    return result

repeats = 5

n_gates_list = [90, 110, 130, 150, 1000]
inputs = []

for i in range(repeats):
    for g in n_gates_list:
        inputs.append((g,i))

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

    average_depth_ratios = {k: 0 for k in n_gates_list}

    for (n_gates, test_time, depth_overhead, depth_ratio, total_time) in results:
        average_depth_ratios[n_gates] += (1.0/repeats) * depth_ratio

    print(average_depth_ratios)
