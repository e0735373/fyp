from .utils import get_asset_path, render_from_layout, render_from_layout_crisp

import matplotlib.pyplot as plt
import numpy as np

NUM_OBJECTS = 5
EMPTY, EDGE, START, GOAL, START_AT_GOAL = range(5)

def build_layout(obs):
    start_node = None
    goal_node = None

    edges = []
    node_names = set()
    for lit in obs:
        if lit.predicate.name == 'goal-at':
            assert len(lit.variables) == 1
            assert goal_node is None
            goal_node = lit.variables[0]
        elif lit.predicate.name == 'at':
            assert len(lit.variables) == 1
            assert start_node is None
            start_node = lit.variables[0]
        elif lit.predicate.name == 'edge':
            assert len(lit.variables) == 2
            u, v = lit.variables
            assert u != v
            edges.append((u, v))
            node_names.add(u)
            node_names.add(v)

    assert start_node is not None
    assert goal_node is not None
    assert start_node in node_names
    assert goal_node in node_names

    node_names_to_idx = {name: i for i, name in enumerate(sorted(node_names))}
    num_nodes = len(node_names_to_idx)

    adjacency = [[EMPTY for _ in range(num_nodes)] for _ in range(num_nodes)]
    for u, v in edges:
        u_idx = node_names_to_idx[u]
        v_idx = node_names_to_idx[v]
        adjacency[u_idx][v_idx] = EDGE

    if start_node == goal_node:
        adjacency[node_names_to_idx[start_node]][node_names_to_idx[start_node]] = START_AT_GOAL
    else:
        adjacency[node_names_to_idx[start_node]][node_names_to_idx[start_node]] = START
        adjacency[node_names_to_idx[goal_node]][node_names_to_idx[goal_node]] = GOAL

    return np.array(adjacency)

def render(obs, mode='human', close=False):
    return build_layout(obs)
