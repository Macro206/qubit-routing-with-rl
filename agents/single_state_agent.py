
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
from keras import backend as K

from annealers.single_state_annealer import Annealer
from utils.PER_memory_tree import Memory

class DQNAgent:

    def __init__(self, environment, memory_size=500):
        self.environment = environment
        self.environment.alternative_reward_delivery = True
        self.furthest_distance = int(np.amax(self.environment.distance_matrix))
        self.max_node_degree = int(np.amax(np.sum(self.environment.adjacency_matrix, axis=1)))
        self.memory_size = memory_size

        self.gamma = 0.8
        self.epsilon_decay = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.learning_rate = 0.001

        self.current_model = self.build_model(self.furthest_distance)#+self.max_node_degree+1)
        self.target_model = self.build_model(self.furthest_distance)#+self.max_node_degree+1)

        self.update_target_model()

        self.memory_tree = Memory(memory_size)
        self.annealer = Annealer(self, environment)

    def build_model(self, input_size):
        """
        Build the neural network model for this agent
        """
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
        edges = list(filter(lambda e: e[0] not in protected_nodes and e[1] not in protected_nodes, edges))
        edge_index_map = {edge: index for index,edge in enumerate(edges)}

        while len(edges) > 0:
            edge = random.sample(edges, 1)[0]
            action[edge_index_map[edge]] = 1

            # This also removes the sampled edge
            edges = [e for e in edges if e[0] not in edge and e[1] not in edge]

        return action

    def obtain_target_nodes(self, current_state): # TODO: rename
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

        distance_vector = [0 for _ in range(self.furthest_distance)]

        for n in range(len(nodes_to_target_nodes)):
            target = nodes_to_target_nodes[n]

            if target == -1:
                continue

            d = int(self.environment.distance_matrix[n][target])
            distance_vector[d-1] += 1  # the vector is effectively indexed from 1

        return np.reshape(np.array(distance_vector), (1,self.furthest_distance))

        # n_available_edges = len(list(filter(lambda e: e[0] not in protected_nodes and e[1] not in protected_nodes, self.environment.edge_list)))
        #
        # available_swaps_vector = [0 for _ in range(self.max_node_degree+1)]
        #
        # for n in range(len(nodes_to_target_nodes)):
        #     if n in protected_nodes:
        #         continue
        #
        #     neighbours = np.where(self.environment.adjacency_matrix[n] == 1)[0]
        #     a = len(list(filter(lambda n: n not in protected_nodes, neighbours)))
        #     available_swaps_vector[a] += 1

        # return np.reshape(np.array(distance_vector + [n_available_edges]), (1,self.furthest_distance+1))

        # return np.reshape(np.array(distance_vector + available_swaps_vector), (1,self.furthest_distance+self.max_node_degree+1))

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

    def replay(self, batch_size):
        """
        Learns from past experiences
        """

        tree_index, minibatch, ISweights = self.memory_tree.sample(batch_size)
        minibatch_with_weights = zip(minibatch,ISweights)
        absolute_errors = []

        for experience, ISweight in minibatch_with_weights:
            [state, reward, next_state, done] = experience[0]

            target_nodes = self.obtain_target_nodes(state)
            next_target_nodes = self.obtain_target_nodes(next_state)

            Qval = self.current_model.predict(target_nodes)[0]

            if done:
                target = reward
            else:
                target = reward + self.gamma * self.target_model.predict(next_target_nodes)[0]

            absolute_error = abs(Qval - target)
            absolute_errors.append(absolute_error)

            self.current_model.fit(target_nodes, [target], epochs=1, verbose=0, sample_weight=ISweight)

        self.memory_tree.batch_update(tree_index, absolute_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def used_up_memory_capacity(self):
        return self.memory_tree.tree.used_up_capacity
