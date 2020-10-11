
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

from annealers.paired_state_annealer import Annealer
from utils.PER_memory_tree import Memory

class DQNAgent:

    def __init__(self, environment, memory_size=500):
        self.environment = environment
        self.furthest_distance = int(np.amax(self.environment.distance_matrix))
        self.max_node_degree = int(np.max(np.sum(self.environment.adjacency_matrix, axis=1)))
        self.memory_size = memory_size

        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9
        self.learning_rate = 0.001

        self.fix_learning_bug = True

        self.NN_state_size = self.furthest_distance+1+self.max_node_degree+1
        self.current_model = self.build_model(self.NN_state_size)
        self.target_model = self.build_model(self.NN_state_size)

        self.update_target_model()

        self.memory_tree = Memory(memory_size)
        self.annealer = Annealer(self, environment)

    def build_model(self, furthest_distance):
        """
        Build the neural network model for this agent
        """

        input_size = furthest_distance * 2

        model = Sequential()
        model.add(Dense(32, input_dim=input_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Copy weights from the current model to the target model
        """
        self.target_model.set_weights(self.current_model.get_weights())

    def save_model(self, model_name=None):
        # Serialize model to JSON
        model_json = self.current_model.to_json()

        if model_name is not None:
            filepath = "./models/" + model_name
        else:
            filepath = "./models/agent_model"

        with open(filepath + ".json", "w") as json_file:
            json_file.write(model_json)

        # Serialize weights to HDF5
        self.current_model.save_weights(filepath + ".h5")
        print("Saved model to disk")

    def load_model(self, model_name=None):
        self.epsilon = self.epsilon_min

        if model_name is not None:
            filepath = "./models/" + model_name
        else:
            filepath = "./models/agent_model"

        # Load json and create model
        json_file = open(filepath + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.current_model = model_from_json(loaded_model_json)

        # Load weights into new model
        self.current_model.load_weights(filepath + ".h5")
        self.current_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.update_target_model()
        print("Loaded model from disk")

    def remember(self, state, reward, next_state, done):
        """
        Store experience in the memory tree
        """
        self.memory_tree.store((state, reward, next_state, done))

    def generate_random_action(self, protected_nodes):
        """
        Generates a random layer of swaps
        Care is taken to ensure that all swaps can occur in parallel
        That is, no two neighbouring edges undergo a swap simultaneously
        """

        action = np.array([0] * len(self.environment.edge_list))  # an action representing an empty layer of swaps

        edges = [(n1,n2) for (n1,n2) in self.environment.edge_list]

        if not self.fix_learning_bug:
            edges = list(filter(lambda e: e[0] not in protected_nodes and e[1] not in protected_nodes, edges))

        edge_index_map = {edge: index for index,edge in enumerate(edges)}

        if self.fix_learning_bug:
            edges = list(filter(lambda e: e[0] not in protected_nodes and e[1] not in protected_nodes, edges))

        while len(edges) > 0:
            edge = random.sample(edges, 1)[0]
            action[edge_index_map[edge]] = 1

            # This also removes the sampled edge
            edges = [e for e in edges if e[0] not in edge and e[1] not in edge]

        return action

    def obtain_distance_vector(self, current_state):
        """
        Obtains a vector that summarises the different distances
        from qubits to their targets.

        More precisely, x_i represents the number of qubits that are
        currently a distance of i away from their targets.

        If there are n qubits, then the length of this vector
        will also be n.
        """

        qubit_locations, qubit_targets, _, protected_nodes = current_state

        nodes_to_target_qubits = \
            [qubit_targets[qubit_locations[n]] for n in range(0,len(qubit_locations))]

        nodes_to_target_nodes = [next(iter(np.where(np.array(qubit_locations) == q)[0]), -1) \
                                 for q in nodes_to_target_qubits]

        distance_vector = [0 for _ in range(self.furthest_distance+1)]

        for n in range(len(nodes_to_target_nodes)):
            target = nodes_to_target_nodes[n]

            if target == -1:
                continue

            d = int(self.environment.distance_matrix[n][target])

            if d > 1 or nodes_to_target_nodes[target] != n:
                distance_vector[d] += 1
            else:
                distance_vector[d-1] += 1


        best_swaps_vector = [0 for _ in range(self.max_node_degree+1)]

        for node, target in enumerate(nodes_to_target_nodes):
            if target == -1:
                continue

            dist = self.environment.distance_matrix[node][target]

            if dist == 1:
                continue

            neighbours = np.where(self.environment.adjacency_matrix[node] == 1)[0]
            candidate_neighbours = []

            for neighbour in neighbours:
                if self.environment.distance_matrix[neighbour][target] == dist-1 \
                    and neighbour not in protected_nodes:
                    candidate_neighbours.append(neighbour)

            # print('Node ' + str(node) + ' with target ' + str(target) + ' has candidate neighbours: ' + str(candidate_neighbours))

            best_swaps_vector[len(candidate_neighbours)] += 1


        return distance_vector + best_swaps_vector

    def get_NN_input(self, current_state, next_state):
        current_state_distance_vector = self.obtain_distance_vector(current_state)
        next_state_distance_vector = self.obtain_distance_vector(next_state)

        return np.reshape(np.array(current_state_distance_vector + next_state_distance_vector), \
                            (1,len(current_state_distance_vector)*2))

    def get_quality(self, current_state, next_state, action_chooser='model'):
        neural_net_input = self.get_NN_input(current_state, next_state)

        if action_chooser == 'model':
            Qval = self.current_model.predict(neural_net_input)[0]
        elif action_chooser == 'target':
            Qval = self.target_model.predict(neural_net_input)[0]

        return Qval

    def act(self, current_state):
        """
        Chooses an action to perform in the environment and returns it
        (i.e. does not alter environment state)
        """

        protected_nodes = current_state[3]

        if np.random.rand() <= self.epsilon:
            action = self.generate_random_action(protected_nodes)
            return action, "Random"

        # Choose an action using the agent's current neural network
        action, _ = self.annealer.simulated_annealing(current_state, action_chooser='model')
        return action, "Model"

    def replay(self, batch_size, print_experiences=False):
        """
        Learns from past experiences
        """

        tree_index, minibatch, ISweights = self.memory_tree.sample(batch_size)
        minibatch_with_weights = zip(minibatch,ISweights)
        absolute_errors = []

        for experience, ISweight in minibatch_with_weights:
            [state, reward, next_state, done] = experience[0]

            NN_input = self.get_NN_input(state, next_state)
            Qval = self.get_quality(state, next_state)

            if done:
                target = reward
            else:
                _, energy = self.annealer.simulated_annealing(next_state, action_chooser='target', search_limit=10)
                bonus = -energy
                target = reward + self.gamma * bonus

            absolute_error = abs(Qval - target)
            absolute_errors.append(absolute_error)

            target_exp = np.sum(NN_input[:,:self.furthest_distance]) == 2

            if print_experiences and target_exp:
                print()
                print(np.reshape(self.obtain_targets(state), (self.environment.rows, self.environment.cols)))
                print()
                print(np.reshape(self.obtain_targets(next_state), (self.environment.rows, self.environment.cols)))
                print()
                print('Rep:', NN_input)
                print()
                print('Prediction:', Qval)
                print('Reward:', reward)
                print('Bonus:', target - reward)
                print('Total:', target)
                print('------')
                print()

            self.current_model.fit(NN_input, [target], epochs=1, verbose=0, sample_weight=ISweight)

        self.memory_tree.batch_update(tree_index, absolute_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def obtain_targets(self, current_state):
        """
        Obtains a list that maps nodes to their targets
        More precisely, a node n1 targets another node n2
        iff n1 holds q1 and n2 holds q2 and q1 targets q2
        """

        qubit_locations = np.array(current_state[0])
        qubit_targets = current_state[1]

        nodes_to_target_qubits = \
            [qubit_targets[qubit_locations[n]] for n in range(0,len(qubit_locations))]

        return np.reshape(np.array(nodes_to_target_qubits), (1,len(qubit_locations)))

    def used_up_memory_capacity(self):
        return self.memory_tree.tree.used_up_capacity
