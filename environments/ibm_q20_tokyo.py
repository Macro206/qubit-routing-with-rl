
import numpy as np
from environments.environment import Environment

class IBMQ20Tokyo(Environment):

    def __init__(self, circuit, qubit_locations=None):
        topology = self.generate_grid_topology(4, 5)
        self.adjust_topology(topology)
        super().__init__(topology, circuit, qubit_locations)
        self.rows = 4
        self.cols = 5

    def generate_grid_topology(self, rows, columns):
        topology = [[0] * (rows*columns) for _ in range(0,rows*columns)]

        for i in range(0,rows):
            for j in range(0,columns):
                node_index = i*columns + j

                if node_index >= columns: # up
                    topology[node_index][node_index-columns] = 1

                if node_index < columns*(rows-1): # down
                    topology[node_index][node_index+columns] = 1

                if node_index % columns > 0: # left
                    topology[node_index][node_index-1] = 1

                if node_index % columns < columns-1: # right
                    topology[node_index][node_index+1] = 1

        return np.array(topology)

    def adjust_topology(self, topology):
        bonus_links = [(1,7), (2,6), (3,9), (4,8), (5,11), (6,10), (7,13), (8,12), (11,17), (12,16), (13,19), (14,18)]

        for (n1,n2) in bonus_links:
            topology[n1][n2] = 1
            topology[n2][n1] = 1
