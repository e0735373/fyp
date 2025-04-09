import csv
import gymnasium
import numpy as np
import os
import time


class AltEnvSudoku(gymnasium.Env):
    def __init__(self, env_name):
        super().__init__()
        self.problem_index = None
        self.problem = None
        self.solution = None
        self.state = (None, None)
        self.sudoku_csv = self.init_sudoku_csv()

        assert "PDDLEnvMysudoku" in env_name
        if "PDDLEnvMysudokuTest" in env_name:
            self.start_problem_index = int(5e5)
        else:
            self.start_problem_index = 0

    def fix_problem_index(self, problem_index):
        self.problem_index = problem_index

    def reset(self):
        index = self.problem_index
        if index is None:
            index = np.random.randint(0, 1000)

        sudoku_grid, sudoku_soln = self.get_sudoku(index)
        self.problem = sudoku_grid
        self.solution = sudoku_soln
        self.state = ("", 0)

        assert len(self.problem) == 81
        assert len(self.solution) == 81

        return self.state, {}

    def step(self, action):
        act_ = None
        for i in range(1, 10):
            if f"value-{i}" in str(action):
                assert act_ is None
                act_ = str(i)
        assert act_ is not None

        if self.state[1] < 81:
            assert len(act_) == 1
            self.state = (self.state[0] + act_, self.state[1] + 1)

        if self.is_state_good():
            return self.state, 1, True, False, {}
        else:
            return self.state, 0, False, False, {}

    def render(self, mode="human"):
        grid = [[0 for _ in range(9)] for _ in range(9)]
        for i in range(len(self.state[0])):
            grid[i // 9][i % 9] = int(self.state[0][i])
        for i in range(len(self.problem)):
            if self.problem[i] != '0':
                grid[i // 9][i % 9] = int(self.problem[i])
        return np.array(grid)

    def init_sudoku_csv(self):
        with open(f"sudoku.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            reader = list(reader)
        return reader

    def get_sudoku(self, index):
        sudoku_grid, sudoku_soln = self.sudoku_csv[index + 1 + self.start_problem_index]
        return sudoku_grid, sudoku_soln

    def is_state_good(self):
        if len(self.state[0]) != 81:
            return False

        # Check basic constraints
        for i in range(81):
            if self.state[0][i] == '0':
                return False
            if self.problem[i] != '0' and self.problem[i] != self.state[0][i]:
                return False

        # Check all rows
        for i in range(9):
            row = self.state[0][i * 9:(i + 1) * 9]
            if len(set(row)) != len(row):
                return False

        # Check all columns
        for i in range(9):
            col = self.state[0][i::9]
            if len(set(col)) != len(col):
                return False

        # Check all 3x3 boxes (subgrids)
        for i in range(3):
            for j in range(3):
                box = []
                for k in range(3):
                    for l in range(3):
                        box.append(self.state[0][(i * 3 + k) * 9 + (j * 3 + l)])
                if len(set(box)) != len(box):
                    return False

        return True


with open(f"sudoku.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    reader = list(reader)
