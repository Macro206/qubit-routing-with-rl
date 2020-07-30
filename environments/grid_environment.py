
import numpy as np
from environments.environment import Environment

class GridEnvironment(Environment):

    def __init__(self, rows, columns, circuit, qubit_locations=None):
        topology = self.generate_grid_topology(rows, columns)
        super().__init__(topology, circuit, qubit_locations)
        self.rows = rows
        self.cols = columns

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
