import random
import os
import sys
import argparse

def generate_problem(num_nodes, seed):
    random.seed(seed)

    nodes = ["node{}".format(i) for i in range(num_nodes)]

    # generate a random tree
    tree = {}
    for i in range(num_nodes):
        tree[i] = (i + 1) % num_nodes
    edges = [(tree[i], i) for i in range(num_nodes)]

    # shuffle labels
    random.shuffle(nodes)

    edges_str = []
    for edge in edges:
        edges_str.append(f"(edge {nodes[edge[0]]} {nodes[edge[1]]})")
        edges_str.append(f"(edge {nodes[edge[1]]} {nodes[edge[0]]})")
    edges_str = sorted(edges_str)

    start_node = random.choice(nodes)
    goal_node = random.choice(nodes)
    while goal_node == start_node:
        start_node = random.choice(nodes)
        goal_node = random.choice(nodes)

    return f"""(define (problem tree{num_nodes})
    (:domain path)
    (:objects {" ".join(nodes)} - NODE)
    (:init {" ".join(edges_str)} (at {start_node}) (goal-at {goal_node}))
    (:goal (at {goal_node}))
)"""

def main():
    parser = argparse.ArgumentParser(description='Generate a path problem')
    parser.add_argument('-n', type=int, help='Number of nodes')
    parser.add_argument('-s', type=int, help='Random seed')
    args = parser.parse_args()
    print(generate_problem(args.n, args.s))

if __name__ == '__main__':
    main()
