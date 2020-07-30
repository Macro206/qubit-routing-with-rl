
import random

from pytket.routing import Architecture, route, convert_index_mapping, place_with_map
from pytket.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.transform import Transform

from environments.circuits import NodeCircuit
from environments.physical_environment import verify_circuit

def generate_architecture(environment):
    coupling_map = []

    a = environment.adjacency_matrix

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == 1:
                coupling_map.append((i,j))

    architecture = Architecture(coupling_map)

    return architecture

def assemble_timesteps_from_gates(number_of_nodes, gates):
    d = [0] * number_of_nodes
    timesteps = []

    for (gate_type,n1,n2) in gates:
        d_max = max(d[n1], d[n2])

        new_depth = d_max + 1

        d[n1] = new_depth
        d[n2] = new_depth

        if new_depth > len(timesteps):
            timesteps.append([(gate_type,n1,n2)])
        else:
            timesteps[new_depth-1].append((gate_type,n1,n2))

    return timesteps


def schedule_swaps(environment, circuit, qubit_locations=None, safety_checks_on=False, decompose_cnots=False):
    original_circuit = circuit

    circuit = qiskit_to_tk(circuit.to_qiskit_rep())
    architecture = generate_architecture(environment)

    if qubit_locations is None:
        qubit_locations = list(range(environment.number_of_nodes))
        random.shuffle(qubit_locations)

    initial_index_map = {qubit: node for node,qubit in enumerate(qubit_locations)}
    initial_map = convert_index_mapping(circuit, architecture, initial_index_map)

    initial_qubit_locations = [-1]*len(qubit_locations)

    for k, v in initial_map.items():
        q = k.index[0]
        n = v.index[0]

        initial_qubit_locations[n] = q

    place_with_map(circuit, initial_map)
    routed_circuit = route(circuit, architecture, swap_lookahead=1000, bridge_interactions=0, bridge_lookahead=0)

    node_circuit = NodeCircuit.from_qiskit_rep(tk_to_qiskit(routed_circuit), decompose=decompose_cnots)

    if decompose_cnots:
        Transform.DecomposeSWAPtoCX().apply(routed_circuit)
        #Transform.DecomposeBRIDGE().apply(routed_circuit)

    tket_depth = routed_circuit.depth()

    calculated_depth = node_circuit.depth()

    if tket_depth != calculated_depth:
        print('Tket depth:', tket_depth)
        print('Calculated depth:', calculated_depth)
        print()

        exit("Tket depth disagrees with calculated depth")

    layers = assemble_timesteps_from_gates(node_circuit.n_nodes, node_circuit.gates)

    if safety_checks_on:
        verify_circuit(original_circuit, node_circuit, environment, initial_qubit_locations)

    return layers, tket_depth
