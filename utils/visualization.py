
import cairo
import math
import numpy as np

from environments.ibm_q20_tokyo import IBMQ20Tokyo
from environments.rigetti_19q_acorn import Rigetti19QAcorn
from utils.experience_db import ExperienceDB


# GLOBAL DRAWING PARAMS
WIDTH = 30
HEIGHT = 20
PIXEL_SCALE = 100


class StateVisualizer:

    def __init__(self, n_nodes, EnvironmentClass):
        placeholder_circuit = []
        for _ in range(n_nodes):
            placeholder_circuit.append([])

        self.reset_drawing_state()
        self.environment = EnvironmentClass(placeholder_circuit)
        self.rows = self.environment.rows
        self.cols = self.environment.cols

        self.node_radius = min(WIDTH/self.cols, HEIGHT/self.rows) / 3

    def reset_drawing_state(self):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24,
                                     WIDTH*PIXEL_SCALE,
                                     HEIGHT*PIXEL_SCALE)

        ctx = cairo.Context(surface)
        ctx.scale(PIXEL_SCALE, PIXEL_SCALE)

        ctx.rectangle(0, 0, WIDTH, HEIGHT)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()

        self.surface = surface
        self.context = ctx


    def save_image(self, path):
        self.surface.write_to_png(path)

    def calculate_swaps(self, qubit_locations, next_qubit_locations):
        swaps = set()

        for n,q in enumerate(qubit_locations):
            future_locs = np.where(np.array(next_qubit_locations) == q)[0]

            if future_locs:
                future_node = future_locs[0]

                if next_qubit_locations[future_node] == q \
                    and next_qubit_locations[n] == qubit_locations[future_node]:

                    swap = (n, future_node) if n < future_node else (future_node, n)
                    swaps.add(swap)

        return swaps

    def draw_node(self, x, y, r, qubit, target):
        ctx = self.context

        ctx.move_to(x,y)

        ctx.arc(x, y, r, 0, 2*math.pi)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(0.6)

        ctx.stroke()

        ctx.arc(x, y, r, 0, 2*math.pi)
        if target == -1:
            ctx.set_source_rgb(0.5, 0.5, 0.5)
        else:
            ctx.set_source_rgb(1, 1, 1)

        ctx.fill()

        ctx.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(1)
        ctx.set_source_rgb(0, 0, 0)

        (_, _, text_width, text_height, _, _) = ctx.text_extents(str(qubit))
        ctx.move_to(x-text_width/1.8, y-text_height/1.5)
        ctx.show_text(str(qubit))

        (_, _, text_width, text_height, _, _) = ctx.text_extents(str(target))
        ctx.move_to(x-text_width/1.8, y+text_height*1.5)
        ctx.show_text(str(target))

    def draw_link(self, x1, y1, x2, y2, interaction_type):
        ctx = self.context

        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)

        if interaction_type == "gate":
            ctx.set_source_rgb(0.4, 1, 0.4)
        elif interaction_type == "swap":
            ctx.set_source_rgb(1, 0.4, 0.4)
        else:
            ctx.set_source_rgb(0, 0, 0)

        ctx.set_line_width(0.3)
        ctx.stroke()

    def draw_topology(self, state, next_state=None):
        qubit_locations, qubit_targets, circuit_progress, _ = state
        swaps = self.calculate_swaps(qubit_locations, next_state[0]) \
                if next_state is not None else set()

        horizontal_separation_dist = WIDTH / self.cols
        vertical_separation_dist = HEIGHT / self.rows

        if horizontal_separation_dist < vertical_separation_dist:
            vertical_separation_dist = horizontal_separation_dist
            horizontal_offset = 0
            vertical_offset = (HEIGHT - horizontal_separation_dist*self.rows) / 2
        else:
            horizontal_separation_dist = vertical_separation_dist
            horizontal_offset = (WIDTH - vertical_separation_dist*self.cols) / 2
            vertical_offset = 0

        for row in range(self.rows):
            for col in range(self.cols):
                x = horizontal_separation_dist/2 + col*horizontal_separation_dist + horizontal_offset
                y = vertical_separation_dist/2 + row*vertical_separation_dist + vertical_offset
                node_number = row*self.cols + col
                qubit_number = qubit_locations[node_number]
                qubit_target = qubit_targets[qubit_number]

                for n in range(node_number+1, self.rows*self.cols):
                    if self.environment.adjacency_matrix[node_number][n] == 1:
                        dest_row = int(n/self.cols)
                        dest_col = n % self.cols

                        dest_x = horizontal_separation_dist/2 + dest_col*horizontal_separation_dist + horizontal_offset
                        dest_y = vertical_separation_dist/2 + dest_row*vertical_separation_dist + vertical_offset

                        is_gate = qubit_targets[qubit_number] == qubit_locations[n] \
                              and qubit_targets[qubit_locations[n]] == qubit_number

                        is_swap = (node_number, n) in swaps

                        interaction_type = "swap" if is_swap else "gate" if is_gate else "none"

                        self.draw_link(x, y, dest_x, dest_y, interaction_type)

                self.draw_node(x, y, self.node_radius, qubit_number, qubit_target)


state_viz = StateVisualizer(20, Rigetti19QAcorn)
db = ExperienceDB()
db.load_from_disk("cqc_scaling_with_depth")

experience_number = 97

experience = db.experiences[experience_number][0]

for i in range(len(experience)):
    state = experience[i][1]
    next_state = experience[i+1][1] if i < len(experience)-1 else None
    state_viz.draw_topology(state, next_state)
    state_viz.save_image("../../Visualizations/temp/" + str(i) + ".png")
