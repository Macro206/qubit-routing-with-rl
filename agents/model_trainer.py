import numpy as np
import copy
from collections import deque

from environments.environment import Environment

def reset_environment_state(env, circuit_generation_function):
    if circuit_generation_function is not None:
        circuit = circuit_generation_function()
    else:
        circuit = None

    initial_state, gates_scheduled = env.generate_starting_state(circuit)

    while env.is_done(initial_state[1]):
        initial_state, gates_scheduled = env.generate_starting_state(circuit)

    return initial_state, gates_scheduled

def train_model(environment, agent, training_episodes=350, circuit_generation_function=None, should_print=True):
    num_actions_deque = deque(maxlen=50)
    batch_size = 32
    time_between_model_updates = 5

    # --- Fill up memory tree ---
    while agent.used_up_memory_capacity() < agent.memory_size:
        state, gates_scheduled = reset_environment_state(environment, circuit_generation_function)

        for time in range(500):
            action, _ = agent.act(state)
            next_state, reward, done, next_gates_scheduled = environment.step(action, state)

            agent.remember(state, reward, next_state, done)
            state = next_state
            gates_scheduled = next_gates_scheduled

            if done:
                num_actions = time + 1
                num_actions_deque.append(num_actions)
                break

    # --- Training ---
    for e in range(training_episodes):
        state, gates_scheduled = reset_environment_state(environment, circuit_generation_function)

        if should_print:
            print("Episode", e, "starting positions\n", np.reshape(state[0], (environment.rows, environment.cols)))

        for time in range(500):
            temp_state = copy.deepcopy(environment.copy_state(state))
            action, _ = agent.act(state)
            new_state = copy.deepcopy(environment.copy_state(state))

            if not np.array_equal(temp_state, new_state):
                print("Error: state not preserved when selecting action")
                exit()

            next_state, reward, done, next_gates_scheduled = environment.step(action, state)
            agent.remember(state, reward, next_state, done)
            state = next_state
            gates_scheduled = next_gates_scheduled

            if done:
                num_actions = time+1
                num_actions_deque.append(num_actions)
                avg_time = np.mean(num_actions_deque)

                if should_print:
                    print("Number of actions: {}, average: {:.5}".format(num_actions, avg_time))
                    print("Final positions\n", np.reshape(next_state[0][0:environment.number_of_nodes],(environment.rows,environment.cols)),'\n')

                break

            agent.replay(batch_size)

            if time % time_between_model_updates == 0:
                agent.update_target_model()
