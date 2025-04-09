import random
import os
import sys
import argparse
import csv

def generate_problem(sudoku_grid, sudoku_soln, id):
    sudoku_grid = sudoku_soln

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

    # every cell have at most one value
    for i in range(81):
        for j in range(1, 10):
            if sudoku_grid[i] == str(j):
                goal_predicates.append(f"(value {cells[i]} {values[j]})")
            else:
                goal_predicates.append(f"(not (value {cells[i]} {values[j]}))")

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
