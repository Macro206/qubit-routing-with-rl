
import random

import networkx as nx
import cirq
import cirq.contrib.routing as ccr

from environments.circuits import NodeCircuit

def generate_device_graph(environment):
    device_graph = nx.Graph()
    nodes = [cirq.NamedQubit('node' + str(n)) for n in range(environment.number_of_nodes)]

    a = environment.adjacency_matrix

    for n in nodes:
        device_graph.add_node(n)

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == 1:
                device_graph.add_edge(nodes[i],nodes[j])

    return nodes, device_graph

def convert_circuit_to_cirq_format(circuit, qubits):
    cirq_circuit = cirq.Circuit()

    for (q1,q2) in circuit.gates:
        cirq_circuit.append([cirq.CX(qubits[q1],qubits[q2])])

    return cirq_circuit

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


def schedule_swaps(environment, circuit, qubit_locations=None):
    unused_qubits = set()

    for q,interactions in enumerate(circuit.to_dqn_rep()):
        if len(interactions) == 0:
            unused_qubits.add(q)

    qubits = [cirq.NamedQubit('qubit' + str(n)) for n in range(circuit.n_qubits)]
    nodes, device_graph = generate_device_graph(environment)

    circuit = convert_circuit_to_cirq_format(circuit, qubits)

    if qubit_locations is None:
        qubit_locations = list(range(environment.number_of_nodes))
        random.shuffle(qubit_locations)

    initial_mapping = {nodes[n]: qubits[q] for n,q in list(filter(lambda p: p[1] not in unused_qubits, enumerate(qubit_locations)))}

    swap_network = ccr.greedy.route_circuit_greedily(circuit, device_graph, max_search_radius=2, initial_mapping=initial_mapping)
    routed_circuit = swap_network.circuit

    gates = []

    for op in routed_circuit.all_operations():
        op_code = 'SWAP' if 'Swap' in str(op.gate) else 'CNOT'
        n1 = int(op.qubits[0].name.replace('node', ''))
        n2 = int(op.qubits[1].name.replace('node', ''))

        gates.append((op_code, n1, n2))

    cirq_depth = len(routed_circuit.moments)

    node_circuit = NodeCircuit.from_gates(environment.number_of_nodes, gates)

    calculated_depth = node_circuit.depth()

    if cirq_depth != calculated_depth:
        print('Cirq depth:', cirq_depth)
        print('Calculated depth:', calculated_depth)
        print()

        exit("Cirq depth disagrees with calculated depth")

    layers = assemble_timesteps_from_gates(node_circuit.n_nodes, node_circuit.gates)

    return layers, cirq_depth
