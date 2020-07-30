
import numpy as np
import random

class Environment:

    def __init__(self, topology, circuit, qubit_locations=None):
        """
        :param topology: an adjacency matrix representing the topology of the target system.
        :param circuit: a list of lists representing the circuit to be scheduled.
        The ith row represents the sequence of interactions that qubit i will undergo during
        the course of the circuit.
        """

        # TODO: check that relevant arguments are indeed NumPy arrays
        # TODO: consider how to deal with circuits that require fewer qubits than
        # available on the target topology

        self.gate_reward = 20
        self.distance_reduction_reward = 2
        self.negative_reward = -10
        self.circuit_completion_reward = 100

        self.alternative_reward_delivery = False

        self.number_of_nodes = len(topology)
        self.number_of_qubits = len(circuit)
        self.adjacency_matrix = np.copy(topology)
        self.circuit = np.copy(circuit) if circuit is not None else None

        self.edge_list = self.generate_edge_list()
        self.distance_matrix = self.generate_distance_matrix()

    @staticmethod
    def generate_random_circuit(number_of_qubits, number_of_gates):
        circuit = []

        for _ in range(number_of_qubits):
            circuit.append([])

        for _ in range(number_of_gates):
            q1 = random.randint(0, number_of_qubits-1)
            q2 = random.randint(0, number_of_qubits-1)

            while q1 == q2:
                q1 = random.randint(0, number_of_qubits-1)
                q2 = random.randint(0, number_of_qubits-1)

            circuit[q1].append(q2)
            circuit[q2].append(q1)

        return circuit

    def generate_starting_state(self, circuit=None, qubit_locations=None):
        if circuit is not None:
            self.circuit = np.copy(circuit)

        if qubit_locations is None:
            qubit_locations = list(np.arange(self.number_of_nodes))
            random.shuffle(qubit_locations)
        else:
            qubit_locations = qubit_locations[:]

        qubit_targets = [interactions[0] if len(interactions) > 0 else -1 \
                         for interactions in self.circuit]

        circuit_progress = [0] * self.number_of_qubits

        gates_to_schedule, protected_nodes = self.next_gates_to_schedule_between_nodes(qubit_targets, qubit_locations)

        starting_state = (qubit_locations, qubit_targets, circuit_progress, protected_nodes)

        return starting_state, gates_to_schedule

    def generate_edge_list(self):
        temp = np.where(self.adjacency_matrix == 1)
        return sorted(list(filter(lambda edge: edge[0] < edge[1], zip(temp[0], temp[1]))))

    def generate_distance_matrix(self):
        """
        Uses the Floyd-Warshall algorithm to generate a matrix of distances
        between physical nodes in the target topology.
        """

        dist = np.full((self.number_of_nodes, self.number_of_nodes), np.inf)

        for (u,v) in self.edge_list:
            dist[u][v] = 1
            dist[v][u] = 1

        for v in range(0,self.number_of_nodes):
            dist[v][v] = 0

        for k in range(0,self.number_of_nodes):
            for i in range(0,self.number_of_nodes):
                for j in range(0,self.number_of_nodes):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    def schedule_gates(self, state):
        reward = 0

        qubit_locations, qubit_targets, circuit_progress, _ = state
        circuit = self.circuit

        for (q1,q2) in self.next_gates_to_schedule(qubit_targets, qubit_locations):
            circuit_progress[q1] += 1
            circuit_progress[q2] += 1

            qubit_targets[q1] = circuit[q1][circuit_progress[q1]] \
                                if circuit_progress[q1] < len(circuit[q1]) \
                                else -1
            qubit_targets[q2] = circuit[q2][circuit_progress[q2]] \
                                if circuit_progress[q2] < len(circuit[q2]) \
                                else -1

            reward += self.gate_reward

        return reward

    def next_gates(self, qubit_targets):
        gates = [(q,qubit_targets[q]) if q == qubit_targets[qubit_targets[q]] and q < qubit_targets[q]
                                      else None for q in range(0,len(qubit_targets))]

        return list(filter(lambda gate: gate is not None and gate[0] < gate[1], gates))

    def next_gates_to_schedule(self, qubit_targets, qubit_locations):
        next_gates = self.next_gates(qubit_targets)

        return list(filter(lambda gate: self.calculate_gate_distance(gate, qubit_locations) == 1, next_gates))

    def next_gates_to_schedule_between_nodes(self, qubit_targets, qubit_locations):
        next_gates_to_schedule = self.next_gates_to_schedule(qubit_targets, qubit_locations)
        next_gates_to_schedule_between_nodes = []

        for (q1,q2) in next_gates_to_schedule:
            (n1,n2) = (np.where(np.array(qubit_locations) == q1)[0][0], \
                       np.where(np.array(qubit_locations) == q2)[0][0])
            gate_between_nodes = (n1,n2) if n1 < n2 else (n2,n1)
            next_gates_to_schedule_between_nodes.append(gate_between_nodes)

        protected_nodes = set()

        for (n1,n2) in next_gates_to_schedule_between_nodes:
            protected_nodes.add(n1)
            protected_nodes.add(n2)

        return next_gates_to_schedule_between_nodes, protected_nodes

    def calculate_gate_distance(self, gate, qubit_locations):
        (q1,q2) = gate

        node1 = np.where(np.array(qubit_locations) == q1)[0][0]
        node2 = np.where(np.array(qubit_locations) == q2)[0][0]

        return self.distance_matrix[node1][node2]

    def calculate_distances(self, qubit_locations, qubit_targets):
        distances = [0]*self.number_of_qubits

        for q in range(self.number_of_qubits):
            target_qubit = qubit_targets[q]

            if target_qubit == -1:
                distances[q] = np.inf
                continue

            node = np.where(np.array(qubit_locations) == q)[0][0]
            target_node = np.where(np.array(qubit_locations) == qubit_targets[q])[0][0]

            distances[q] = self.distance_matrix[node][target_node]

        return distances

    def is_done(self, qubit_targets):
        """
        Returns True iff each qubit has completed all of its interactions
        """
        return all([target == -1 for target in qubit_targets])

    def copy_state(self, state):
        qubit_locations, qubit_targets, circuit_progress, protected_nodes = state
        return (qubit_locations[:], qubit_targets[:], circuit_progress[:], set(protected_nodes))

    def step(self, action, state):
        qubit_locations, qubit_targets, circuit_progress, protected_nodes = self.copy_state(state)

        pre_swap_reward = self.schedule_gates((qubit_locations, qubit_targets, circuit_progress, protected_nodes)) # can serve reward here

        # total_pre_swap_distance = sum([self.calculate_gate_distance(gate, qubit_locations) for gate in self.next_gates(qubit_targets)])

        pre_swap_distances = self.calculate_distances(qubit_locations, qubit_targets)

        swap_edge_indices = np.where(np.array(action) == 1)[0]
        swap_edges = [self.edge_list[i] for i in swap_edge_indices]

        for (node1,node2) in swap_edges:
            temp = qubit_locations[node1]
            qubit_locations[node1] = qubit_locations[node2]
            qubit_locations[node2] = temp

        post_swap_distances = self.calculate_distances(qubit_locations, qubit_targets)

        distance_reduction_reward = 0

        for q in range(self.number_of_qubits):
            if post_swap_distances[q] < pre_swap_distances[q]:
                distance_reduction_reward += self.distance_reduction_reward

        # total_post_swap_distance = sum([self.calculate_gate_distance(gate, qubit_locations) for gate in self.next_gates(qubit_targets)])

        # state_before_scheduling = self.copy_state((qubit_locations, qubit_targets, circuit_progress))

        gates_scheduled, protected_nodes = self.next_gates_to_schedule_between_nodes(qubit_targets, qubit_locations)
        post_swap_reward = len(gates_scheduled) * self.gate_reward

        # Give rewards, based on num_matches (matches = gates) and total distances
        # if scheduling_reward > 0:
        #     reward = scheduling_reward
        # elif total_post_swap_distance < total_pre_swap_distance:
        #     reward = self.distance_reduction_reward
        # else:
        #     reward = self.negative_reward

        # if self.is_done(qubit_targets):
        #     reward += self.circuit_completion_reward # reward doesn't matter if only annealing

        # state_after_scheduling = self.copy_state((qubit_locations, qubit_targets, circuit_progress))

        reward = pre_swap_reward if self.alternative_reward_delivery else post_swap_reward + distance_reduction_reward

        next_state = self.copy_state((qubit_locations, qubit_targets, circuit_progress, protected_nodes))

        return next_state, reward, self.is_done(qubit_targets), gates_scheduled

    def get_neighbour_edge_nums(self, edge_num):
        """
        Finds edges that share a node with input edge.
        :param edge_num: index of input edge (used to get input edge from self.edge_list)
        :return: neighbour_edge_nums: indices of neighbouring edges.
        """
        node1, node2 = self.edge_list[edge_num]
        neighbour_edge_nums = []
        for edge in self.edge_list:
            if node1 in edge or node2 in edge:
                neighbour_edge_nums.append(self.edge_list.index(edge))

        neighbour_edge_nums.remove(edge_num)
        return neighbour_edge_nums
