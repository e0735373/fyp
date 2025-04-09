from typing import Dict
import torch
import numpy as np
import copy
import os
import time
import pddlgym
import multiprocessing
from pddlgym_planners.fd import FD
from ..common.pytorch_util import dict_apply
from ..common.replay_buffer import ReplayBuffer
from ..common.sampler import SequenceSampler, get_val_mask
from .base_dataset import BaseLowdimDataset

class ActionMyHanoi:
    @staticmethod
    def _init():
        print("Initializing actions")
        env = pddlgym.make("PDDLEnvMyhanoiTest-v0")
        planner = FD()

        ActionMyHanoi.ACTIONS = [None for _ in range(ActionMyHanoi.N_PEGS * ActionMyHanoi.N_PEGS)]
        def is_all_actions_initialized():
            cntNone = 0
            for a in ActionMyHanoi.ACTIONS:
                if a is None:
                    cntNone += 1
            assert cntNone >= ActionMyHanoi.N_PEGS
            return cntNone == ActionMyHanoi.N_PEGS # all actions except identity action

        while not is_all_actions_initialized():
            obs, debug_info = env.reset()
            plan = planner(env.domain, obs)
            for act in plan:
                a = ActionMyHanoi.action_to_int(act)
                if ActionMyHanoi.ACTIONS[a] is None:
                    ActionMyHanoi.ACTIONS[a] = copy.deepcopy(act)

        env.close()
        print("Actions initialized")

    @staticmethod
    def action_to_int(action):
        for i in range(ActionMyHanoi.N_PEGS):
            for j in range(ActionMyHanoi.N_PEGS):
                if f"p{i+1}:peg,p{j+1}:peg" in str(action):
                    return i * ActionMyHanoi.N_PEGS + j
        raise ValueError("Invalid action")

    @staticmethod
    def int_to_action(a):
        assert a >= 0 and a < ActionMyHanoi.N_PEGS * ActionMyHanoi.N_PEGS
        return copy.deepcopy(ActionMyHanoi.ACTIONS[a])

    @staticmethod
    def str_to_action(action):
        return ActionMyHanoi.int_to_action(ActionMyHanoi.action_to_int(action))

    ACTIONS = None
    N_PEGS = 3

ActionMyHanoi._init()

def generate_myhanoi_dataset(
    zaar_path,
    pddl_env_name,
    n_episodes,
    n_workers=1,
    episode_length_limit=70,
    episode_idx_start = 0
):
    def generate_episode(episode_idx, thread_queue):
        env = pddlgym.make(pddl_env_name)
        planner = FD()
        while episode_idx < n_episodes:
            env.fix_problem_index(episode_idx)
            print("Generating episode", episode_idx)
            episode_idx += n_workers

            try:
                obs, debug_info = env.reset()
                plan = planner(env.domain, obs)

                if len(plan) > episode_length_limit:
                    print("Episode", episode_idx - n_workers, "too long, skipping")
                    continue

                goal = env.render('layout_goal', goal_override=obs.goal.literals)

                obs_all = []
                action_all = []
                goal_all = []
                for i, act in enumerate(plan):
                    print("Episode", episode_idx - n_workers, "Step", i, "Action", act)

                    obs_ = env.render('layout')
                    action_ = ActionMyHanoi.action_to_int(act)

                    obs_all.append(obs_.flatten())
                    action_all.append(action_)
                    goal_all.append(goal.flatten())

                    obs, reward, done, truncated, debug_info = env.step(act)
                    if done:
                        break

                data = dict()
                data['obs'] = np.stack(obs_all, axis=0)
                data['action'] = np.array(action_all)
                data['goal'] = np.stack(goal_all, axis=0)

                thread_queue.put((0, data))
            except:
                print("Error in episode", episode_idx - n_workers)
                continue

        env.close()
        thread_queue.put((1, None))
        print("Thread", episode_idx % n_workers, "exiting")

    replay_buffer = ReplayBuffer.create_empty_numpy()
    thread_queue = multiprocessing.Queue()

    threads = []
    for i in range(n_workers):
        thread = multiprocessing.Process(
            target=generate_episode, args=(episode_idx_start + i, thread_queue))
        thread.start()
        threads.append(thread)

    cnt_active = n_workers
    while cnt_active > 0:
        l, data = thread_queue.get()

        if l == 1:
            cnt_active -= 1
            print(f"Thread exited, {cnt_active} remaining")
            continue

        replay_buffer.extend(data)
        print("Received episode", i)

    print("All episodes received")
    replay_buffer.save_to_path(zaar_path)
    print("Replay buffer saved to", zaar_path)

    for i, thread in enumerate(threads):
        thread.join()
        print("Thread", i, "joined!")

    print("All threads joined")

class MyhanoiLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='obs',
            action_key='action',
            goal_key='goal',
            seed=42,
            val_ratio=0.0,
            max_n_episodes=None
            ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key, goal_key])

        if max_n_episodes is not None:
            while self.replay_buffer.n_episodes > max_n_episodes:
                self.replay_buffer.pop_episode()

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key] # T, D_o
        data = {
            'obs': obs,
            'action': sample[self.action_key], # T, D_a
            'goal': sample['goal'][0] # D_g
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
