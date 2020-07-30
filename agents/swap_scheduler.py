
import numpy as np

from environments.circuits import NodeCircuit
from environments.physical_environment import verify_circuit

def reset_environment_state(env, circuit, qubit_locations):
    initial_state, gates_scheduled = env.generate_starting_state(circuit, qubit_locations)

    while env.is_done(initial_state[1]):
        initial_state, gates_scheduled = env.generate_starting_state(circuit, qubit_locations)

    return initial_state, gates_scheduled

def schedule_swaps(environment, agent, circuit=None, experience_db=None, qubit_locations=None, safety_checks_on=False, decompose_cnots=False):
    total_gates_scheduled = []
    original_circuit = circuit
    circuit = circuit.to_dqn_rep()

    state, gates_scheduled = reset_environment_state(environment, circuit, qubit_locations)
    actions = []

    initial_qubit_locations = state[0][:]

    total_gates_scheduled.extend(list(map(lambda g: ('CnotGate',g[0],g[1]), gates_scheduled)))

    if experience_db is not None:
        experience_db.new_experience(state, environment.circuit)

    for time in range(1500):
        action, action_type = agent.act(state)
        next_state, reward, done, next_gates_scheduled = environment.step(action, state)

        edges_swapped = [environment.edge_list[e] for e in np.where(np.array(action) == 1)[0]]
        total_gates_scheduled.extend(list(map(lambda g: ('SwapGate',g[0],g[1]), edges_swapped)))
        total_gates_scheduled.extend(list(map(lambda g: ('CnotGate',g[0],g[1]), next_gates_scheduled)))

        actions.append(action)

        if experience_db is not None:
            experience_db.add_state_transition(next_state, next_state, action_type)

        state = next_state
        gates_scheduled = next_gates_scheduled

        if done:
            break

    if experience_db is not None:
        experience_db.save_experience()

    scheduled_circuit = NodeCircuit.from_gates(environment.number_of_nodes, total_gates_scheduled, decompose=decompose_cnots)

    if safety_checks_on:
        verify_circuit(original_circuit, scheduled_circuit, environment, initial_qubit_locations)

    return actions, scheduled_circuit.depth()
