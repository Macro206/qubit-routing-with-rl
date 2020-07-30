
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
            exit("Solution not safe")

        return neighbour_solution

    def check_valid_solution(self, solution, forced_mask):
        for i in range(len(solution)):
            if (forced_mask[i] ==  1 and solution[i] == 0) or \
               (forced_mask[i] == -1 and solution[i] == 1):
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
        next_state_temp_NN_input = self.agent.obtain_target_nodes(next_state_temp)

        # print(next_state_temp_NN_input)

        if action_chooser == 'model':
            Qval = self.agent.current_model.predict(next_state_temp_NN_input)[0]
        elif action_chooser == 'target':
            Qval = self.agent.target_model.predict(next_state_temp_NN_input)[0]

        # print("New solution: ", [self.environment.edge_list[e] for e in np.where(np.array(solution) == 1)[0]])
        # print("Energy: ", Qval)
        # print()

        return -Qval

    def generate_initial_solution(self, current_state):
        protected_nodes = current_state[3]

        force_gates_action = static_heuristics.generate_force_gates_action(self.environment, current_state, version=3)

        if force_gates_action is None:
            num_edges = len(self.environment.edge_list)
            initial_solution = [0]*num_edges
            protected_mask = self.generate_protected_mask(current_state[3])

            available_edges = action_edge_translation.swappable_edges(initial_solution, current_state, protected_mask, self.environment.edge_list, self.environment.number_of_nodes)

            if not available_edges:
                return initial_solution, "None", protected_mask

            edge_index_to_swap = random.sample(available_edges, 1)[0]

            initial_solution[edge_index_to_swap] = (initial_solution[edge_index_to_swap] + 1) % 2

            return initial_solution, "Random", protected_mask
        else:
            return list(force_gates_action[0]), "Forced", force_gates_action[1]

    def generate_protected_mask(self, protected_nodes):
        return list(map(lambda e: -1 if e[0] in protected_nodes or e[1] in protected_nodes else 0, self.environment.edge_list))

    def simulated_annealing(self, current_state, action_chooser='model'):
        current_solution, method, forced_mask = self.generate_initial_solution(current_state)

        # print("Base action:")
        # print([(current_state[1][current_state[0][x]], current_state[1][current_state[0][y]]) for (_,(x,y)) in filter(lambda p: current_solution[p[0]] == 1, enumerate(self.environment.edge_list))])
        # print("Can't move:")
        # print([(current_state[1][current_state[0][x]], current_state[1][current_state[0][y]]) for (_,(x,y)) in filter(lambda p: forced_mask[p[0]] == 1, enumerate(self.environment.edge_list))])
        # print("Can move:")
        # print([(current_state[1][current_state[0][x]], current_state[1][current_state[0][y]]) for (_,(x,y)) in filter(lambda p: forced_mask[p[0]] == 0, enumerate(self.environment.edge_list))])

        available_edges = action_edge_translation.swappable_edges(current_solution, current_state, forced_mask, self.environment.edge_list, self.environment.number_of_nodes)

        if not available_edges or current_solution == [0]*len(self.environment.edge_list):
            # There are no actions possible
            # Often happens when only one gate is left, and it's already been scheduled
            return current_solution, -np.inf

        T = self.initial_temperature
        current_energy = self.get_energy(current_solution, current_state=current_state, action_chooser=action_chooser)

        best_solution = copy.copy(current_solution) #.copy()
        best_energy = current_energy

        iterations_since_best = 0

        while T > self.min_temperature:
            if self.speed_over_optimality and iterations_since_best > 40:
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

        if method == "Forced":
            # i.e. we ensure this is chosen, and not Best Effort
            best_energy = -np.inf

        # print("Best solution: ", [self.environment.edge_list[e] for e in np.where(np.array(best_solution) == 1)[0]])
        # print("Energy: ", best_energy)
        # print()

        # intervals.append(iterations_since_best)
        # print("Intervals: ", intervals)

        return best_solution, best_energy
