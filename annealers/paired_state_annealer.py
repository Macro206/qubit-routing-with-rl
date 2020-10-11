
import random
import math
import numpy as np
import copy
from collections import deque
import utils.static_heuristics as static_heuristics
import utils.action_edge_translation as action_edge_translation


class Annealer:
    def __init__(self, agent, environment):
        self.initial_temperature = 60.0
        self.min_temperature = 0.1
        self.cooling_multiplier = 0.95
        self.environment = environment
        self.agent = agent

        self.reversed_gates_deque = deque(maxlen=20)

        self.safety_checks_on = True
        self.speed_over_optimality = False

    def get_neighbour_solution(self, current_solution, current_state, forced_mask):
        neighbour_solution = copy.copy(current_solution)
        edge_list = self.environment.edge_list
        n_nodes = self.environment.number_of_nodes

        available_edges = action_edge_translation.swappable_edges(neighbour_solution, current_state, forced_mask, edge_list, n_nodes)

        if not available_edges:
            exit("Ran out of edges to swap")

        edge_index_to_swap = random.sample(available_edges, 1)[0]

        neighbour_solution[edge_index_to_swap] = (neighbour_solution[edge_index_to_swap] + 1) % 2

        if self.safety_checks_on and not self.check_valid_solution(neighbour_solution, forced_mask):
            print([self.environment.edge_list[e] for e in np.where(np.array(forced_mask) == 1)[0]])
            print([self.environment.edge_list[e] for e in np.where(np.array(neighbour_solution) == 1)[0]])
            exit("Solution not safe")

        # print("Current solution:")
        # print([self.environment.edge_list[e] for e in np.where(np.array(current_solution) == 1)[0]])
        # print("Available edges:")
        # print([self.environment.edge_list[e] for e in available_edges])
        # print("Neighbour solution:")
        # print([self.environment.edge_list[e] for e in np.where(np.array(neighbour_solution) == 1)[0]])

        return neighbour_solution

    def check_valid_solution(self, solution, forced_mask):
        for i in range(len(solution)):
            if forced_mask[i] == 1 and solution[i] == 1:
                return False

        if 1 in solution:
            swap_edge_indices = np.where(np.array(solution) == 1)[0]
            swap_edges = [self.environment.edge_list[index] for index in swap_edge_indices]
            swap_nodes = [node for edge in swap_edges for node in edge]

            # return False if repeated swap nodes
            seen = set()
            for node in swap_nodes:
                if node in seen:
                    return False
                seen.add(node)
            return True

        return True  # TODO should all zero be valid action?

    def acceptance_probability(self, current_energy, new_energy, temperature):
        if new_energy < current_energy:
            return 1
        else:
            energy_diff = new_energy - current_energy
            probability = math.exp(-energy_diff/temperature)
            return probability

    def get_energy(self, solution, current_state=None, action_chooser='model'):
        next_state_temp, _, _, _ = self.environment.step(solution, current_state)

        Qval = self.agent.get_quality(current_state, next_state_temp, action_chooser)

        # print("New solution: ", [self.environment.edge_list[e] for e in np.where(np.array(solution) == 1)[0]])
        # print("Energy: ", Qval)
        # print()

        return -Qval

    def generate_initial_solution(self, current_state, forced_mask):
        num_edges = len(self.environment.edge_list)
        initial_solution = [0]*num_edges

        available_edges = action_edge_translation.swappable_edges(initial_solution, current_state, forced_mask, self.environment.edge_list, self.environment.number_of_nodes)

        if not available_edges:
            return initial_solution

        edge_index_to_swap = random.sample(available_edges, 1)[0]

        initial_solution[edge_index_to_swap] = (initial_solution[edge_index_to_swap] + 1) % 2

        return initial_solution

    def generate_forced_mask(self, protected_nodes):
        return list(map(lambda e: 1 if e[0] in protected_nodes or e[1] in protected_nodes else 0, self.environment.edge_list))

    def calculate_reversed_gates_proportion(self, suggestion, solution):
        reversed = [suggestion[i] == 1 and solution[i] == 0 for i in range(len(suggestion))]

        if sum(suggestion) == 0 or sum(reversed) == 0:
            return 0.0

        return float(sum(reversed))/float(sum(suggestion))

    def simulated_annealing(self, current_state, action_chooser='model', search_limit=None):
        protected_nodes = current_state[3]

        forced_mask = self.generate_forced_mask(protected_nodes)

        current_solution = self.generate_initial_solution(current_state, forced_mask)

        # print("Initial solution: ", [self.environment.edge_list[e] for e in np.where(np.array(current_solution) == 1)[0]])
        # print("Initially available edges: ", [self.environment.edge_list[e] for e in available_edges])

        if current_solution == [0]*len(self.environment.edge_list):
            # There are no actions possible
            # Often happens when only one gate is left, and it's already been scheduled
            if action_chooser == 'model':
                return current_solution, -np.inf
            else:
                return current_solution, 0

        T = self.initial_temperature
        current_energy = self.get_energy(current_solution, current_state=current_state, action_chooser=action_chooser)

        best_solution = copy.copy(current_solution) #.copy()
        best_energy = current_energy

        iterations_since_best = 0
        iterations = 0

        while T > self.min_temperature:
            if self.speed_over_optimality and iterations_since_best > 40:
                break
            elif search_limit is not None and iterations > search_limit:
                break

            new_solution = self.get_neighbour_solution(current_solution, current_state, forced_mask)
            new_energy = self.get_energy(new_solution, current_state=current_state, action_chooser=action_chooser)
            accept_prob = self.acceptance_probability(current_energy, new_energy, T)
            # print(accept_prob)

            if accept_prob > random.random():
                current_solution = new_solution
                current_energy = new_energy

                # Save best solution, so it can be returned if algorithm terminates at a sub-optimal solution
                if current_energy < best_energy:
                    best_solution = copy.copy(current_solution) #.copy()
                    best_energy = current_energy
                    # intervals.append(iterations_since_best)
                    iterations_since_best = 0

            T = T * self.cooling_multiplier
            iterations_since_best += 1
            iterations += 1

        # print("Best solution: ", [self.environment.edge_list[e] for e in np.where(np.array(best_solution) == 1)[0]])
        # print("Energy: ", -best_energy)
        # print()

        # intervals.append(iterations_since_best)
        # print("Intervals: ", intervals)

        return best_solution, best_energy
