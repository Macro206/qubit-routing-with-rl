
import numpy as np
from environments.environment_v2 import Environment

class Rigetti19QAcorn(Environment):

    def __init__(self, circuit, qubit_locations=None):
        topology = self.generate_acorn_topology()
        super().__init__(topology, circuit, qubit_locations)
        self.rows = 4
        self.cols = 5

    def generate_acorn_topology(self):
        topology = [[0] * 20 for _ in range(20)]

        links  = [(0,5), (0,6), (1,6), (1,7), (2,7), (2,8), (3,8), (3,9), (4,9)]
        links += [(5,10), (6,11), (7,12), (8,13), (9,14)]
        links += [(10,15), (10,16), (11,16), (11,17), (12,17), (12,18), (13,18), (13,19), (14,19)]

        for (n1,n2) in links:
            topology[n1][n2] = 1
            topology[n2][n1] = 1

        return np.array(topology)
