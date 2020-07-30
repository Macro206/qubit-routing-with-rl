
import numpy as np

def remove_edge(available_edges_mask, edge_index_map, edge):
    if edge[0] == edge[1]:
        return
    elif edge[0] > edge[1]:
        edge = (edge[1], edge[0])

    if edge in edge_index_map:
        edge_index = edge_index_map[edge]
        available_edges_mask[edge_index] = 0

def swappable_edges(current_action, current_state, forced_mask, edge_list, n_nodes):
    num_edges = len(edge_list)

    edge_index_map = {edge: index for index,edge in enumerate(edge_list)}

    available_edges_mask = [1]*num_edges

    # We block any edges connected to nodes already involved in a swap
    for i,val in enumerate(current_action):
        if val == 1:
            (n1,n2) = edge_list[i]

            for n in range(n_nodes):
                remove_edge(available_edges_mask, edge_index_map, (n1,n))
                remove_edge(available_edges_mask, edge_index_map, (n2,n))

    # We add back the edges in the current action, since we're allowed to reverse these
    for i,val in enumerate(current_action):
        if val == 1:
            available_edges_mask[i] = 1

    # We remove all forced and protected edges
    for i,val in enumerate(forced_mask):
        if val == 1 or val == -1:
            available_edges_mask[i] = 0

    # If the current action has only one swap, we disallow reversing that one
    # swap_indices = np.where(np.array(current_action) == 1)[0]
    # if len(swap_indices) == 1:
    #     swap_index = swap_indices[0]
    #     available_edges_mask[swap_index] = 0

    # We disallow swapping edges whose nodes are both done interacting
    for i,edge in enumerate(edge_list):
        (n1,n2) = edge

        if  current_state[1][current_state[0][n1]] == -1 \
        and current_state[1][current_state[0][n2]] == -1:
            available_edges_mask[i] = 0

    available_edges = list(np.where(np.array(available_edges_mask) == 1)[0])

    return available_edges
