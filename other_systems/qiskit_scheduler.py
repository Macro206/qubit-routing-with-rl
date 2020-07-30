
import random

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing import BasicSwap, LookaheadSwap, StochasticSwap
from qiskit.transpiler.passes.basis.decompose import Decompose
from qiskit.extensions.standard.swap import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

from environments.circuits import NodeCircuit
from environments.physical_environment import verify_circuit

MethodClass = StochasticSwap

def generate_coupling_map(environment):
    coupling_map = CouplingMap()

    a = environment.adjacency_matrix

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] == 1:
                coupling_map.add_edge(i,j)
                coupling_map.add_edge(j,i)

    if not coupling_map.is_symmetric:
        exit("Qiskit coupling map was not symmetric")

    return coupling_map

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

    if qubit_locations is None:
        qubit_locations = list(range(environment.number_of_nodes))
        random.shuffle(qubit_locations)

    initial_qubit_locations = qubit_locations[:]

    circuit = circuit.to_qiskit_rep(qubit_locations=qubit_locations)
    coupling_map = generate_coupling_map(environment)

    dag_circuit = circuit_to_dag(circuit)

    method_instance = MethodClass(coupling_map, trials=500)
    mapped_dag_circuit = method_instance.run(dag_circuit)
    mapped_circuit = dag_to_circuit(mapped_dag_circuit)

    node_circuit = NodeCircuit.from_qiskit_rep(mapped_circuit, decompose=decompose_cnots)

    if decompose_cnots:
        decomposition_pass = Decompose(type(SwapGate()))
        mapped_dag_circuit = decomposition_pass.run(mapped_dag_circuit)
        mapped_circuit = dag_to_circuit(mapped_dag_circuit)

    qiskit_depth = mapped_circuit.depth()

    calculated_depth = node_circuit.depth()

    if qiskit_depth != calculated_depth:
        print('Data:', mapped_circuit.data)
        print('Gates:', gates)
        print('Qiskit depth:', qiskit_depth)
        print('Calculated depth:', calculated_depth)
        print()

        exit("Qiskit depth disagrees with calculated depth")

    layers = assemble_timesteps_from_gates(node_circuit.n_nodes, node_circuit.gates)

    if safety_checks_on:
        verify_circuit(original_circuit, node_circuit, environment, initial_qubit_locations)

    return layers, qiskit_depth
