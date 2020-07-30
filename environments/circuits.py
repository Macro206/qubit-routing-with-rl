
from qiskit import QuantumCircuit as QiskitCircuit

class QubitCircuit:

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []

    ### BUILDERS ###

    @staticmethod
    def from_gates(n_qubits, gates):
        circuit = QubitCircuit(n_qubits)
        circuit.gates.extend(gates)
        return circuit


    ### GATES ###

    def cnot(self, q1, q2):
        if q1 >= self.n_qubits or q2 >= self.n_qubits:
            raise Exception('Tried to add a gate ' + str((q1,q2)) + \
                            ' but circuit only has ' + str(self.n_qubits) + ' qubits')

        self.gates.append((q1,q2))


    ### OTHER METHODS ###

    def depth(self):
        d = [0] * self.n_qubits

        for (q1,q2) in self.gates:
            d_max = max(d[q1], d[q2])

            d[q1] = d_max + 1
            d[q2] = d_max + 1

        return max(d)


    ### REP GENERATION ###

    def to_dqn_rep(self):
        dqn_rep = []

        for _ in range(self.n_qubits):
            dqn_rep.append([])

        for (q1,q2) in self.gates:
            dqn_rep[q1].append(q2)
            dqn_rep[q2].append(q1)

        return dqn_rep

    def to_qiskit_rep(self, qubit_locations=None):
        if qubit_locations is None:
            gates = self.gates
        else:
            qubit_to_node_map = [-1]*self.n_qubits

            for n,q in enumerate(qubit_locations):
                qubit_to_node_map[q] = n

            gates = list(map(lambda g: (qubit_to_node_map[g[0]], qubit_to_node_map[g[1]]), self.gates))

        qiskit_rep = QiskitCircuit(self.n_qubits)

        for (n1,n2) in gates:
            qiskit_rep.cnot(n1,n2)

        return qiskit_rep


class NodeCircuit:

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.gates = []

    ### BUILDERS ###

    @staticmethod
    def from_gates(n_nodes, gates, decompose=False):
        circuit = NodeCircuit(n_nodes)

        if decompose:
            gates = circuit.decompose_gates(gates)

        circuit.gates.extend(gates)

        return circuit

    @staticmethod
    def from_qiskit_rep(qiskit_rep, decompose=False):
        circuit = NodeCircuit(len(qiskit_rep.qubits))

        gates = []

        for gate_obj, qubits, _ in qiskit_rep.data:
            gate = (gate_obj.__class__.__name__, qubits[0].index, qubits[1].index)
            gates.append(gate)

        if decompose:
            gates = circuit.decompose_gates(gates)

        circuit.gates.extend(gates)

        return circuit


    ### OTHER METHODS ###

    def decompose_gates(self, gates):
        decomposed_gates = []
        for (type, n1, n2) in gates:
            if 'swap' in type.lower():
                decomposition = [('CnotGate',n1,n2), ('CnotGate',n2,n1), ('CnotGate',n1,n2)]
                decomposed_gates.extend(decomposition)
            elif 'cx' in type.lower() or 'cnot' in type.lower():
                decomposed_gates.append((type,n1,n2))
            else:
                exit('Unknown gate type "' + str(type) + '" in circuit when decomposing')

        return decomposed_gates

    def depth(self):
        d = [0] * self.n_nodes

        for (_,n1,n2) in self.gates:
            d_max = max(d[n1], d[n2])

            d[n1] = d_max + 1
            d[n2] = d_max + 1

        return max(d)
