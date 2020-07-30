
import numpy as np
import random


def generate_force_gates_action(environment, current_state, version=3):
    dist = environment.distance_matrix
    qubit_node_map = np.array(current_state[0])

    gates_between_qubits = environment.next_gates(current_state[1])

    gates = [(np.where(qubit_node_map == q1)[0][0], np.where(qubit_node_map == q2)[0][0]) \
             for (q1,q2) in gates_between_qubits]

    gates_to_force = list(filter(lambda gate: dist[gate[0]][gate[1]] == 2, gates))

    if len(gates) == 1 and version == 2:
        return generate_sparse_state_action(environment, gates[0])

    elif len(gates_to_force) == 0:
        return None

    random.shuffle(gates_to_force)

    action = np.array([0] * len(environment.edge_list))  # an action representing an empty layer of swaps

    protected_nodes = current_state[3]

    available_edges = set(filter(lambda e: e[0] not in protected_nodes and e[1] not in protected_nodes, environment.edge_list))
    edge_index_map = {edge: index for index,edge in enumerate([(n1,n2) for (n1,n2) in environment.edge_list])}

    for n1,n2 in gates_to_force:
        n1_neighbours = np.where(dist[n1] == 1)[0]
        n2_neighbours = np.where(dist[n2] == 1)[0]
        intermediate_nodes = np.intersect1d(n1_neighbours, n2_neighbours)
        np.random.shuffle(intermediate_nodes)

        for n3 in intermediate_nodes:
            possible_edges = [((n1,n3) if n1 < n3 else (n3,n1)), \
                              ((n2,n3) if n2 < n3 else (n3,n2))]
            random.shuffle(possible_edges)

            for edge_to_swap in possible_edges:
                if edge_to_swap not in available_edges:
                    continue

                action[edge_index_map[edge_to_swap]] = 1

                fixed_node = n1 if n1 not in edge_to_swap else n2

                edges = [(m1,m2) for (m1,m2) in environment.edge_list]

                for edge in edges:
                    if edge_to_swap[0] in edge or edge_to_swap[1] in edge or fixed_node in edge:
                        if edge in available_edges:
                            available_edges.remove(edge)

    force_mask = [1 if val == 1 else -1 if environment.edge_list[i] not in available_edges else 0 for i,val in enumerate(action)]

    if version == 3:
        return action, force_mask
    else:
        return action
