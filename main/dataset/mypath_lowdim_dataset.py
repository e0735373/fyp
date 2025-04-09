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

class ActionMyPath:
    @staticmethod
    def _init():
        print("Initializing actions")
        env = pddlgym.make("PDDLEnvMypathTest-v0")
        planner = FD()

        ActionMyPath.ACTIONS = [None for _ in range(ActionMyPath.N_NODES)]
        def is_all_actions_initialized():
            for a in ActionMyPath.ACTIONS:
                if a is None:
                    return False
            return True

        while not is_all_actions_initialized():
            obs, debug_info = env.reset()
            plan = planner(env.domain, obs)
            for act in plan:
                a = ActionMyPath.action_to_int(act)
                if ActionMyPath.ACTIONS[a] is None:
                    ActionMyPath.ACTIONS[a] = copy.deepcopy(act)

        env.close()
        print("Actions initialized")

    @staticmethod
    def action_to_int(action):
        for i in range(ActionMyPath.N_NODES):
            if f"node{i}:node" in str(action):
                return i
        raise ValueError("Invalid action")

    @staticmethod
    def int_to_action(a):
        assert a >= 0 and a < ActionMyPath.N_NODES
        return copy.deepcopy(ActionMyPath.ACTIONS[a])

    ACTIONS = None
    N_NODES = 10
    N_STEPS = N_NODES

ActionMyPath._init()

def generate_mypath_dataset(
    zaar_path,
    pddl_env_name,
    n_episodes,
    n_workers=1,
    episode_length_limit=60,
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

                obs_all = []
                action_all = []
                for i, act in enumerate(plan):
                    print("Episode", episode_idx - n_workers, "Step", i, "Action", act)

                    obs_ = env.render('layout')
                    action_ = ActionMyPath.action_to_int(act)

                    obs_all.append(obs_.flatten())
                    action_all.append(action_)

                    obs, reward, done, truncated, debug_info = env.step(act)
                    if done:
                        break

                while len(obs_all) < ActionMyPath.N_STEPS:
                    obs_ = env.render('layout')
                    goal_node = [i for i in range(ActionMyPath.N_NODES) if obs_[i][i] == 4]
                    assert len(goal_node) == 1

                    goal_node = goal_node[0]
                    action_ = -100

                    obs_all.append(obs_.flatten())
                    action_all.append(action_)

                assert len(obs_all) == ActionMyPath.N_STEPS
                assert len(action_all) == ActionMyPath.N_STEPS

                data = dict()
                data['obs'] = np.stack(obs_all, axis=0)
                data['action'] = np.array(action_all)

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

class MypathLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='obs',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_n_episodes=None
            ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key])

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
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
