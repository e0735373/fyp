import random
import os
import sys
import argparse
import csv

# (define (domain sudoku)
# (:requirements :typing)
# (:types LOCATION NUMBER)
# (:predicates
#   (at ?l - LOCATION)
#   (value ?l - LOCATION ?n - NUMBER)
#   (next-location ?l1 - LOCATION ?l2 - LOCATION)
#   (move ?n - NUMBER)
#   (official-solution ?l - LOCATION ?n - NUMBER)
# )

# ; (:actions move)

# (:action move
#   :parameters (?l1 - LOCATION ?l2 - LOCATION ?n - NUMBER)
#   :precondition (and (move ?n) (at ?l1) (next-location ?l1 ?l2) )
#   :effect (and (at ?l2) (not (at ?l1)) (value ?l1 ?n))
# )
# )

def generate_problem(sudoku_grid, sudoku_soln, id):
    cells = ["cell-{}-{}".format(i, j) for i in range(9) for j in range(9)]
    values = ["value-{}".format(i) for i in range(0, 10)]

    cells.append("cell-goal")

    assert len(sudoku_grid) == 81
    assert len(sudoku_soln) == 81
    assert all(c in "0123456789" for c in sudoku_grid)
    assert all(c in "123456789" for c in sudoku_soln)

    init_predicates = []
    for i in range(9):
        for j in range(9):
            if sudoku_grid[i*9+j] != "0":
                init_predicates.append(f"(value {cells[i*9+j]} {values[int(sudoku_grid[i*9+j])]})")

    init_predicates.append(f"(at {cells[0]})")
    for i in range(81):
        init_predicates.append(f"(next-location {cells[i]} {cells[i+1]})")

    for i in range(81):
        init_predicates.append(f"(official-solution {cells[i]} {values[int(sudoku_soln[i])]})")

    goal_predicates = [f"(at {cells[-1]})"]

    # every cell must have a value
    for i in range(81):
        cur_pred = []
        for j in range(1, 10):
            cur_pred.append(f"(value {cells[i]} {values[j]})")
        goal_predicates.append(f"(or {' '.join(cur_pred)})")

    # every cell have at most one value
    for i in range(81):
        for j in range(1, 10):
            for k in range(j+1, 10):
                goal_predicates.append(f"(or (not (value {cells[i]} {values[j]})) (not (value {cells[i]} {values[k]})))")

    edges = []
    for i in range(9):
        for j in range(9):
            for k in range(j+1, 9):
                # same row
                edges.append((i*9+j, i*9+k))

                # same column
                edges.append((j*9+i, k*9+i))

            # same 3x3 box
            box_i = i // 3
            box_j = j // 3
            for k in range(3):
                for l in range(3):
                    ni = 3*box_i + k
                    nj = 3*box_j + l
                    if i < ni or (i == ni and j < nj):
                        edges.append((i*9+j, ni*9+nj))

    for edge in edges:
        for i in range(1, 10):
            goal_predicates.append(f"(or (not (value {cells[edge[0]]} {values[i]})) (not (value {cells[edge[1]]} {values[i]})))")

    return f"""(define (problem sudoku-{id})
    (:domain sudoku)
    (:objects {" ".join(cells)} - LOCATION {" ".join(values[1:])} - NUMBER)
    (:init {" ".join(init_predicates)})
    (:goal (and {" ".join(goal_predicates)}))
)"""

def main():
    parser = argparse.ArgumentParser(description='Generate a path problem')
    parser.add_argument('-id', type=int, help='Index of the problem')
    args = parser.parse_args()

    with open("sudoku.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        reader = list(reader)
        sudoku_grid, sudoku_soln = reader[args.id+1]

    print(generate_problem(sudoku_grid, sudoku_soln, args.id))

if __name__ == '__main__':
    main()
