
import numpy as np

class PhysicalEnvironment:

    def __init__(self, original_circuit, scheduled_circuit, environment, initial_qubit_locations, verbose=False):
        if verbose:
            print('Original gates:', original_circuit.gates)

        self.circuit = original_circuit.to_dqn_rep()
        self.gates = scheduled_circuit.gates

        self.topology = environment.adjacency_matrix

        self.state = (initial_qubit_locations[:], [0]*original_circuit.n_qubits) # Qubit locations and circuit progress

        self.verbose = verbose

    def execute_swap(self, n1, n2):
        if self.topology[n1][n2] != 1:
            exit('Nodes ' + str(n1) +  ' and ' + str(n2) + ' not adjacent while executing SWAP')

        qubit_locations = self.state[0]

        q1 = qubit_locations[n1]
        q2 = qubit_locations[n2]

        qubit_locations[n1] = q2
        qubit_locations[n2] = q1

    def execute_cnot(self, n1, n2):
        if self.topology[n1][n2] != 1:
            exit('Nodes ' + str(n1) +  ' and ' + str(n2) + ' not adjacent while executing CNOT')

        qubit_locations, circuit_progress = self.state

        q1 = qubit_locations[n1]
        q2 = qubit_locations[n2]

        if not (self.circuit[q1][circuit_progress[q1]] == q2 and self.circuit[q2][circuit_progress[q2]] == q1):
            exit('Qubits ' + str(q1) + ' and ' + str(q2) + ' are not looking to interact')

        circuit_progress[q1] += 1
        circuit_progress[q2] += 1

    def execute_gate(self, n1, n2, type):
        if self.verbose:
            print('Executing gate:', (n1,n2,type))

        if 'swap' in type.lower():
            self.execute_swap(n1,n2)
        elif 'cx' in type.lower() or 'cnot' in type.lower():
            self.execute_cnot(n1,n2)
        else:
            exit('Unknown gate type "' + str(type) + '" in circuit')

    def execute_circuit(self):
        if self.verbose:
            print('Executing circuit on physical environment')
            print('Initial qubit locations:', self.state[0])

        for type,n1,n2 in self.gates:
            self.execute_gate(n1,n2,type)

        for q in range(len(self.circuit)):
            if self.state[1][q] != len(self.circuit[q]):
                exit('Circuit not complete')

        if self.verbose:
            print('All gates scheduled')
            print()


def verify_circuit(original_circuit, scheduled_circuit, environment, initial_qubit_locations, verbose=False):
    physical_env = PhysicalEnvironment(original_circuit, scheduled_circuit, environment, initial_qubit_locations, verbose)
    physical_env.execute_circuit()
