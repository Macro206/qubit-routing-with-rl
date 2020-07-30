
import numpy as np
import copy
import matplotlib.pyplot as plt
import time as time_module
import random

from multiprocessing import Pool, cpu_count

from agents_final.paired_state_agent import DQNAgent
from environments.grid_environment import GridEnvironment
from agents_final.model_trainer import train_model
from agents_final.swap_scheduler import schedule_swaps
from utils.circuit_tools import generate_full_layer_circuit


def perform_run(nrows, ncols, training_episodes):
    test_episodes = 100

    circuit_generation_function = lambda: generate_full_layer_circuit(nrows*ncols).to_dqn_rep()

    environment = GridEnvironment(nrows,ncols,circuit_generation_function())
    agent = DQNAgent(environment)

    start_time = time_module.clock()

    train_model(environment, agent, training_episodes=training_episodes, circuit_generation_function=circuit_generation_function, should_print=False)

    average_circuit_depth_overhead = 0.0

    for e in range(test_episodes):
        actions, circuit_depth = schedule_swaps(environment, agent, circuit=circuit_generation_function(), experience_db=None)
        average_circuit_depth_overhead += (1.0/test_episodes) * (circuit_depth - 1)

    end_time = time_module.clock()

    total_time = end_time-start_time

    datapoint = (nrows, ncols, average_circuit_depth_overhead, total_time)

    print('Completed run:', datapoint)

    return datapoint


repeats = 5

grid_sizes = [(4,4), (4,5), (5,5), (5,6), (6,6), (6,7), (7,7)]
inputs = [(4,4,50), (4,5,100), (5,5,120), (5,6,150), (6,6,200), (6,7,250), (7,7,300)] * repeats

random.shuffle(inputs)

if __name__ == '__main__':
    p = Pool(cpu_count())
    results = p.starmap(perform_run, inputs)

    results.sort(key=lambda r: (r[0]*r[1]))

    print()
    for r in results:
        print(r)
    print()

    average_depth_overheads = {s: 0 for s in grid_sizes}

    for (nrows, ncols, depth_overhead, total_time) in results:
        average_depth_overheads[(nrows,ncols)] += (1.0/repeats) * depth_overhead

    print(average_depth_overheads)
