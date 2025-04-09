from .utils import fig2data

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def render(obs, mode='human', close=False, goal_override=None):
    layout = [[0 for _ in range(9)] for _ in range(9)]
    for lit in obs:
        if lit.predicate.name == 'value':
            # (value cell-0-2 value-4)
            cell_x = int(lit.variables[0].name.split("-")[1])
            cell_y = int(lit.variables[0].name.split("-")[2])
            value = int(lit.variables[1].name.split("-")[1])
            layout[cell_x][cell_y] = value
    return np.array(layout)
