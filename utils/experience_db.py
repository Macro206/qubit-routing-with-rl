
import numpy as np
import pickle

class ExperienceDB:

    def __init__(self):
        self.experiences = []

        self.current_experience = None
        self.current_circuit = None

    ### SAVING EXPERIENCES ###

    def new_experience(self, initial_state, circuit):
        self.current_experience = [(None, initial_state, None)]
        self.current_circuit = circuit

    def add_state_transition(self, next_state_before_scheduling, next_state_after_scheduling, action_type):
        self.current_experience.append((next_state_before_scheduling, next_state_after_scheduling, action_type))

    def save_experience(self):
        self.experiences.append((self.current_experience, self.current_circuit))

        self.current_experience = None
        self.current_circuit = None


    ### DISK OPS ###

    def write_to_disk(self, experiment_name=None):
        if experiment_name is not None:
            filepath = "./experience_data/" + experiment_name + ".p"
        else:
            filepath = "./experience_data/experiences.p"

        pickle.dump(self.experiences, open(filepath, "wb"))

    def load_from_disk(self, experiment_name=None):
        if experiment_name is not None:
            filepath = "./experience_data/" + experiment_name + ".p"
        else:
            filepath = "./experience_data/experiences.p"

        self.experiences = pickle.load(open(filepath, "rb"))


    ### STATE UTILS ###

    def obtain_targets(self, current_state):
        qubit_locations = np.array(current_state[0])
        qubit_targets = current_state[1]

        nodes_to_target_qubits = \
            [qubit_targets[qubit_locations[n]] for n in range(0,len(qubit_locations))]

        return np.reshape(np.array(nodes_to_target_qubits), (1,len(qubit_locations)))

    def obtain_target_nodes(self, current_state):
        qubit_locations = np.array(current_state[0])
        qubit_targets = current_state[1]

        nodes_to_target_qubits = \
            [qubit_targets[qubit_locations[n]] for n in range(0,len(qubit_locations))]

        nodes_to_target_nodes = [next(iter(np.where(np.array(qubit_locations) == q)[0]), -1) \
                                 for q in nodes_to_target_qubits]

        return np.reshape(np.array(nodes_to_target_nodes), (1,len(qubit_locations)))

    ### DISPLAYING DATA ###

    def display_experiences_info(self):
        for index, experience in enumerate(self.experiences):
            print("Experience #" + str(index) + ": test time " + str(len(experience[0])-1))

    def display_experience(self, index, display_circuit=False, rows=4, cols=4, target_nodes=False):
        if target_nodes:
            display_function = self.obtain_target_nodes
        else:
            display_function = self.obtain_targets

        if index < 0 or index >= len(self.experiences):
            print("Experience not found")
        else:
            experience, circuit = self.experiences[index]

            if display_circuit:
                print("Circuit:")
                print(circuit)
                print()

                print("Experience:")
                print()

            for (state1, state2, action_type) in experience:
                print("Action type:", action_type)

                if state1 is not None and state2 is not None \
                    and np.array_equal(display_function(state1), display_function(state2)):
                    print(np.reshape(display_function(state1), (rows,cols)))
                    print()

                else:
                    if state1 is not None:
                        print(np.reshape(display_function(state1), (rows,cols)))
                        print()

                    if state2 is not None:
                        print(np.reshape(display_function(state2), (rows,cols)))
                        print()

                print()
