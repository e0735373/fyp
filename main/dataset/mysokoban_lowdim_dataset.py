from typing import Dict
import torch
import numpy as np
import copy
import os
import time
import pddlgym
import multiprocessing
from pddlgym_planners.ff import FF
from ..common.pytorch_util import dict_apply
from ..common.replay_buffer import ReplayBuffer
from ..common.sampler import SequenceSampler, get_val_mask
from .base_dataset import BaseLowdimDataset
from .altenvsokoban import AltEnvSokoban

class ActionMySokoban:
    @staticmethod
    def _init():
        print("Initializing actions")
        env = pddlgym.make("PDDLEnvMysokoban_init-v0")
        planner = FF()

        ActionMySokoban.ACTIONS = [None, None, None, None]
        def is_all_actions_initialized():
            for a in ActionMySokoban.ACTIONS:
                if a is None:
                    return False
            return True

        while not is_all_actions_initialized():
            obs, debug_info = env.reset()
            plan = planner(env.domain, obs)
            for act in plan:
                a = ActionMySokoban.action_to_int(act)
                if ActionMySokoban.ACTIONS[a] is None:
                    ActionMySokoban.ACTIONS[a] = copy.deepcopy(act)

        env.close()
        print("Actions initialized")

    @staticmethod
    def action_to_int(action):
        if "right:dir" in str(action):
            return 0
        elif "down:dir" in str(action):
            return 1
        elif "left:dir" in str(action):
            return 2
        elif "up:dir" in str(action):
            return 3
        else:
            raise ValueError("Invalid action")

    @staticmethod
    def int_to_action(a):
        assert a >= 0 and a < 4
        return copy.deepcopy(ActionMySokoban.ACTIONS[a])

    ACTIONS = [None, None, None, None]

ActionMySokoban._init()

def generate_mysokoban_dataset(
    zaar_path,
    pddl_env_name,
    n_episodes,
    n_workers=1,
    episode_length_limit=100,
    episode_idx_start = 0
):
    def generate_episode(episode_idx, thread_queue):
        print("Thread", episode_idx % n_workers, "starting env...", flush=True)
        env = AltEnvSokoban(pddl_env_name)
        print("Thread", episode_idx % n_workers, "started env", flush=True)
        planner = FF()
        print("Thread", episode_idx % n_workers, "finished instantiating FF", flush=True)

        while episode_idx < n_episodes:
            print("Generating episode", episode_idx, flush=True)

            env.fix_problem_index(episode_idx)
            episode_idx += n_workers

            try:
                print("Planning episode", episode_idx - n_workers, flush=True)
                obs, debug_info = env.reset()
                plan = planner(env.env.domain, obs)

                if len(plan) > episode_length_limit:
                    print("Episode", episode_idx - n_workers, "too long, skipping", flush=True)
                    continue

                obs_all = []
                action_all = []
                done = False
                for i, act in enumerate(plan):
                    print("Episode", episode_idx - n_workers, "Step", i, "Action", act, flush=True)

                    obs_ = env.render('layout')
                    action_ = ActionMySokoban.action_to_int(act)

                    obs_all.append(obs_.flatten())
                    action_all.append(action_)

                    obs, reward, done, truncated, debug_info = env.step(act)
                    if done:
                        break

                assert done

                data = dict()
                data['obs'] = np.stack(obs_all, axis=0)
                data['action'] = np.array(action_all)

                thread_queue.put((0, data))
            except:
                print("Error in episode", episode_idx - n_workers, flush=True)
                continue

        thread_queue.put((1, episode_idx % n_workers))
        print("Thread", episode_idx % n_workers, "exiting", flush=True)
        # env.close()

    replay_buffer = ReplayBuffer.create_empty_numpy()
    thread_queue = multiprocessing.Queue()

    print("Generating", n_episodes, "episodes with", n_workers, "workers")

    threads = []
    for i in range(n_workers):
        thread = multiprocessing.Process(
            target=generate_episode, args=(episode_idx_start + i, thread_queue))
        thread.start()
        threads.append(thread)

    print("Threads started", flush=True)

    cnt_active = n_workers
    while cnt_active > 0:
        if thread_queue.empty():
            time.sleep(0.1)
            continue

        l, data = thread_queue.get()

        if l == 1:
            cnt_active -= 1
            print(f"Thread {data} exited, {cnt_active} remaining", flush=True)
            continue

        replay_buffer.extend(data)
        print("Received episode", flush=True)

    print("All episodes received")
    replay_buffer.save_to_path(zaar_path)
    print("Replay buffer saved to", zaar_path)

    for i, thread in enumerate(threads):
        thread.join()
        print("Thread", i, "joined!")

    print("All threads joined")

class MysokobanLowdimDataset(BaseLowdimDataset):
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

        self.n_episodes = self.replay_buffer.n_episodes
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
