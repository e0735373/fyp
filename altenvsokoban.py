import csv
import gymnasium
import numpy as np
import os
import time
import copy
import pddlgym


class AltEnvSokoban(gymnasium.Env):
    def __init__(self, env_name):
        super().__init__()
        self.problem_index = None
        self.state = None

        assert "PDDLEnvMysokoban" in env_name
        self.env = pddlgym.make(env_name)

    def fix_problem_index(self, problem_index):
        self.problem_index = problem_index

    def reset(self):
        index = self.problem_index
        if index is None:
            raise ValueError("Problem index is not set. Use fix_problem_index() to set it.")

        self.env.fix_problem_index(index)
        obs, _ = self.env.reset()

        self.state = self.build_complete_layout(obs.literals)
        self.state_shape = self.state["is_wall"].shape
        # print("State shape:", self.state_shape)
        return copy.deepcopy(self.state), {}

    def step(self, action):
        act_i, act_j = 0, 0
        if "move(down:dir)" in str(action):
            act_i = 1
        elif "move(up:dir)" in str(action):
            act_i = -1
        elif "move(left:dir)" in str(action):
            act_j = -1
        elif "move(right:dir)" in str(action):
            act_j = 1
        else:
            raise ValueError(f"Invalid action: {action}")

        def is_player(i, j):
            return copy.deepcopy(self.state)["is_player"][i][j] == 1 # player

        def is_box(i, j):
            return copy.deepcopy(self.state)["is_box"][i][j] == 1

        def is_wall(i, j):
            if i < 0 or i >= self.state_shape[0] or j < 0 or j >= self.state_shape[1]:
                return True
            return copy.deepcopy(self.state)["is_wall"][i][j] == 1 # wall

        def is_goal(i, j):
            return copy.deepcopy(self.state)["is_goal"][i][j] == 1

        def is_finished():
            cnt_good_boxes = 0
            for i in range(self.state_shape[0]):
                for j in range(self.state_shape[1]):
                    if is_box(i, j) and is_goal(i, j):
                        cnt_good_boxes += 1
            return cnt_good_boxes >= 2

        if is_finished():
            return copy.deepcopy(self.state), 1, True, False, {}

        # Move player
        player_i, player_j = None, None
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if is_player(i, j):
                    assert player_i is None and player_j is None
                    player_i, player_j = i, j

        assert player_i is not None and player_j is not None

        new_player_i = player_i + act_i
        new_player_j = player_j + act_j

        if is_wall(new_player_i, new_player_j):
            return copy.deepcopy(self.state), 0, False, False, {}
        elif is_box(new_player_i, new_player_j):
            new_box_i = new_player_i + act_i
            new_box_j = new_player_j + act_j

            if is_wall(new_box_i, new_box_j):
                return copy.deepcopy(self.state), 0, False, False, {}
            elif is_box(new_box_i, new_box_j):
                return copy.deepcopy(self.state), 0, False, False, {}
            else:
                # Move box
                self.state["is_box"][new_box_i][new_box_j] = 1
                self.state["is_box"][new_player_i][new_player_j] = 0

                # Move player
                self.state["is_player"][new_player_i][new_player_j] = 1
                self.state["is_player"][player_i][player_j] = 0
        else:
            # Move player
            self.state["is_player"][new_player_i][new_player_j] = 1
            self.state["is_player"][player_i][player_j] = 0

        if is_finished():
            return copy.deepcopy(self.state), 1, True, False, {}
        else:
            return copy.deepcopy(self.state), 0, False, False, {}

    def render(self, mode="human"):
        NUM_OBJECTS = 6
        CLEAR, PLAYER, STONE, STONE_AT_GOAL, GOAL, WALL = range(NUM_OBJECTS)

        layout = CLEAR * np.ones_like(self.state["is_wall"], dtype=np.uint8)

        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if self.state["is_goal"][i][j] == 1:
                    layout[i][j] = GOAL
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if self.state["is_box"][i][j] == 1:
                    if self.state["is_goal"][i][j] == 1:
                        layout[i][j] = STONE_AT_GOAL
                    else:
                        layout[i][j] = STONE
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if self.state["is_player"][i][j] == 1:
                    layout[i][j] = PLAYER
        for i in range(self.state_shape[0]):
            for j in range(self.state_shape[1]):
                if self.state["is_wall"][i][j] == 1:
                    layout[i][j] = WALL

        return layout

    def build_complete_layout(self, obs):
        def loc_str_to_loc(loc_str):
            r, c = loc_str.split('-')
            return (int(r[1:]), int(c[:-1]))

        def get_locations(obs, thing):
            locs = []
            for lit in obs:
                if lit.predicate.name == thing:
                    for v in lit.variables:
                        if v.name.startswith('f') and v.name.endswith('f'):
                            locs.append(loc_str_to_loc(v.name))
            return locs

        # Get location boundaries
        max_r, max_c = -np.inf, -np.inf
        for lit in obs:
            # print(lit, vars(lit))
            for v in lit.variables:
                if v.name.startswith('f') and v.name.endswith('f'):
                    r, c = loc_str_to_loc(v.name)
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)

        is_wall = np.zeros((max_r+1, max_c+1), dtype=np.uint8)
        is_box = np.zeros((max_r+1, max_c+1), dtype=np.uint8)
        is_player = np.zeros((max_r+1, max_c+1), dtype=np.uint8)
        is_goal = np.zeros((max_r+1, max_c+1), dtype=np.uint8)

        # Put things in the layout
        # Also track seen locs and goal locs
        seen_locs = set()
        goal_locs = set()

        # Find goals
        for r, c in get_locations(obs, 'is-goal'):
            is_goal[r, c] = 1
            seen_locs.add((r, c))
            goal_locs.add((r, c))

        for r, c in get_locations(obs, 'at'):
            if (r, c) in goal_locs:
                is_box[r, c] = 1
            else:
                is_box[r, c] = 1
            seen_locs.add((r, c))

        for r, c in get_locations(obs, 'at-robot'):
            is_player[r, c] = 1
            seen_locs.add((r, c))

        for r, c in get_locations(obs, 'clear'):
            seen_locs.add((r, c))

        # Add walls
        for r in range(max_r+1):
            for c in range(max_c+1):
                if (r, c) in seen_locs:
                    continue
                is_wall[r, c] = 1

        return {
            "is_wall": is_wall,
            "is_box": is_box,
            "is_player": is_player,
            "is_goal": is_goal
        }
