import os
import torch
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import time

from main.common.pytorch_util import dict_apply, optimizer_to
from main.dataset.base_dataset import BaseLowdimDataset
from main.common.checkpoint_util import TopKCheckpointManager
from main.common.json_logger import JsonLogger
from main.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
from transformeragent import TransformerAgent, DiffusionAgent, D3PMAgent
from config import MysokobanConfig, MyhanoiConfig

from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading

import pddlgym
import multiprocessing

class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, output_dir: Optional[str]=None):
        self.output_dir = output_dir
        self._saving_thread = None

    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None,
            include_keys=None,
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
            exclude_keys=None,
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


class TrainTransformerWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, output_dir: Optional[str]=None, seed=42, model_type=None, config_type=None):
        super().__init__(output_dir)

        self.cfg : MysokobanConfig | MyhanoiConfig
        self.cfg = config_type

        self.ignore_index = -100

        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model : TransformerAgent | DiffusionAgent | D3PMAgent
        self.model = model_type(self.cfg)

        # configure training state
        self.optimizer = self.model.get_optimizer(
            learning_rate=self.cfg.OPT_LEARNING_RATE,
            weight_decay=self.cfg.OPT_WEIGHT_DECAY,
            betas=self.cfg.OPT_BETAS
        )

        self.global_step = 0
        self.epoch = 0

        print("number of parameters", sum(p.numel() for p in self.model.parameters()))

    def train(self, training_resume=True, zarr_path="./storage/mysokoban-train", dataset_cls:BaseLowdimDataset=None):
        DEBUG = False

        cfg = self.cfg
        HAS_GOAL_CONDITION = (cfg.HAS_GOAL_CONDITION == 1)

        # resume training
        if training_resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset_horizon = cfg.N_OBS_STEPS + cfg.HORIZON - 1
        dataset = dataset_cls(
            zarr_path,
            horizon=dataset_horizon,
            pad_before=(0 if cfg.HAS_NO_PADDING == 1 else cfg.N_OBS_STEPS-1),
            pad_after=(0 if cfg.HAS_NO_PADDING == 1 else cfg.HORIZON-1),
            val_ratio=0.1
        )

        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset,
                                      batch_size=cfg.DATALOADER_BATCH_SIZE, shuffle=cfg.DATALOADER_SHUFFLE)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=cfg.DATALOADER_BATCH_SIZE, shuffle=cfg.DATALOADER_SHUFFLE)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.LR_SCHEDULER,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.LR_WARMUP_STEPS,
            num_training_steps=
                len(train_dataloader) * cfg.NUM_EPOCHS,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            monitor_key='val_loss',
            k=int(1e7) # basically infinite
        )

        # device transfer
        device = torch.device('cuda')
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        validation_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        # create file and directories if not exist
        log_path = pathlib.Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.NUM_EPOCHS):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = copy.deepcopy(batch)

                        # compute loss
                        loss = self.model.compute_loss(batch)
                        loss.backward()

                        # step optimizer
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        # logging
                        raw_loss_cpu = loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if DEBUG:
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # run validation
                self.model.eval()
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                            leave=False) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if validation_sampling_batch is None:
                                validation_sampling_batch = copy.deepcopy(batch)

                            loss = self.model.compute_loss(batch)
                            val_losses.append(loss)

                            if DEBUG:
                                break

                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

                # run rollout on train_sampling_batch and validation_sampling_batch
                if True or (self.epoch % cfg.TRAINING_CHECKPOINT_EVERY) == 0:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(
                            [train_sampling_batch, validation_sampling_batch]
                        ):
                            assert batch is not None

                            obs = batch['obs']
                            actions = batch['action']

                            if HAS_GOAL_CONDITION:
                                goal = batch['goal']
                            else:
                                goal = None

                            obs = obs[:, :cfg.N_OBS_STEPS]
                            actions = actions[:, -cfg.HORIZON:]

                            actions_pred = self.model.predict(obs, do_sample=False, goal=goal)

                            acc = torch.sum(
                                actions_pred[actions != self.ignore_index] ==
                                actions[actions != self.ignore_index]).item()
                            tot = actions[actions != self.ignore_index].numel()

                            batch_name = 'train' if batch_idx == 0 else 'val'
                            step_log[f'{batch_name}_predict_acc'] = 0 if tot == 0 else acc / tot

                # checkpoint
                if (self.epoch % cfg.TRAINING_CHECKPOINT_EVERY) == 0:
                    # checkpointing
                    self.save_checkpoint()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def eval(self, ckpt_tag='latest', n_worker=8, n_episodes=100, first_k_moves=None, action_to_int_cls=None, env_name=None):
        assert (first_k_moves is None) or first_k_moves >= 1

        cfg = self.cfg
        HAS_GOAL_CONDITION = (cfg.HAS_GOAL_CONDITION == 1)

        # resume training
        lastest_ckpt_path = self.get_checkpoint_path(ckpt_tag)
        if lastest_ckpt_path.is_file():
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        self.model.to(self.model.device)
        print(f"first_k_moves={first_k_moves}")
        print(f"n_episodes={n_episodes}")
        print(f"env_name={env_name}")

        # configure dataset
        LIMIT = cfg.EVAL_UNSEEN_STEP_LIMIT
        final_plans = [None for _ in range(n_episodes)]

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            def worker_fn(w, in_que, out_que, fin_que):
                if cfg.OVERRIDE_PDDLGYM_ENVIRONMENT:
                    env = cfg.OVERRIDE_PDDLGYM_ENVIRONMENT_CLASS(env_name)
                else:
                    env = pddlgym.make(env_name)

                for i in range(w, n_episodes, n_worker):
                    env.fix_problem_index(i)
                    orig_obs, *_ = env.reset()

                    past_obs = list()
                    done = False
                    plan = list()

                    obs = env.render('layout').flatten()
                    oobs = obs
                    if HAS_GOAL_CONDITION:
                        goal = env.render('layout_goal', goal_override=orig_obs.goal.literals).flatten()
                    else:
                        goal = None

                    for _ in range(cfg.N_OBS_STEPS):
                        past_obs.append(obs)

                    while not done and len(plan) < LIMIT:
                        obs_ = np.array(past_obs[-cfg.N_OBS_STEPS:])
                        out_que.put((i, obs_, goal))
                        actions = in_que.get()
                        assert len(actions) == cfg.HORIZON

                        if first_k_moves is not None:
                            actions = actions[:first_k_moves]

                        for act_ in actions:
                            act = action_to_int_cls.int_to_action(act_)
                            plan.append(str(act))

                            print(f"Episode {i} Step {len(plan)} Action {act}", flush=True)

                            obs, reward, done, truncated, debug_info = env.step(act)
                            past_obs.append(env.render('layout').flatten())

                            if done:
                                break

                    step_log = {
                        'plan_length': len(plan),
                        'plan': str(plan),
                        'success': (done and len(plan) <= LIMIT),
                        'episode_idx': i,
                        'real-success': done,
                        'obs': np.array(oobs),
                    }

                    json_logger.log(step_log)
                    step_log['plan'] = plan

                    fin_que.put((i, step_log))

                env.close()

            threads = []
            in_queues = []
            out_queues = []
            fin_queues = []
            for i in range(n_worker):
                in_que = multiprocessing.Queue()
                out_que = multiprocessing.Queue()
                fin_que = multiprocessing.Queue()

                in_queues.append(in_que)
                out_queues.append(out_que)
                fin_queues.append(fin_que)

                t = multiprocessing.Process(target=worker_fn, args=(i, in_que, out_que, fin_que))
                t.start()

                threads.append(t)

            cnt_finished = 0
            while cnt_finished < n_episodes:
                time.sleep(0.1)

                indices = []
                obs_all = []
                goal_all = []
                for i in range(n_worker):
                    if out_queues[i].empty():
                        continue

                    idx, obs_, goal_ = out_queues[i].get()
                    assert idx < n_episodes
                    assert obs_.shape[0] == cfg.N_OBS_STEPS

                    indices.append(i)
                    obs_all.append(torch.tensor(obs_))
                    if HAS_GOAL_CONDITION:
                        goal_all.append(torch.tensor(goal_))

                if len(indices) > 0:
                    obs_all = torch.stack(obs_all, dim=0).to(self.model.device)
                    if not HAS_GOAL_CONDITION:
                        goal_all = None
                    else:
                        goal_all = torch.stack(goal_all, dim=0).to(self.model.device)

                    with torch.no_grad():
                        self.model.eval()
                        actions_all = self.model.predict(obs_all, do_sample=True, goal=goal_all)
                        assert actions_all.shape[0] == len(indices)

                    for i, idx in enumerate(indices):
                        actions = actions_all[i]
                        assert actions.shape[0] == cfg.HORIZON and len(actions.shape) == 1

                        actions = actions.cpu().numpy()
                        in_queues[idx].put(actions)

                for i in range(n_worker):
                    if fin_queues[i].empty():
                        continue

                    idx, step_log = fin_queues[i].get()
                    assert 0 <= idx < n_episodes
                    final_plans[idx] = step_log

                    cnt_finished += 1
                    print(f"Episode {idx} finished (success={step_log['success']}), progress={cnt_finished}/{n_episodes}", flush=True)

            for t in threads:
                t.join()

        return final_plans


    def eval_accuracy(self, ckpt_tag='latest', zarr_path="./storage/mysokoban-test", dataset_cls:BaseLowdimDataset=None):
        cfg = self.cfg
        HAS_GOAL_CONDITION = (cfg.HAS_GOAL_CONDITION == 1)

        # resume training
        lastest_ckpt_path = self.get_checkpoint_path(ckpt_tag)
        if lastest_ckpt_path.is_file():
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)

        self.model.to(self.model.device)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset_horizon = cfg.N_OBS_STEPS + cfg.HORIZON - 1
        dataset = dataset_cls(
            zarr_path,
            horizon=dataset_horizon,
            pad_before=(0 if cfg.HAS_NO_PADDING == 1 else cfg.N_OBS_STEPS-1),
            pad_after=(0 if cfg.HAS_NO_PADDING == 1 else cfg.HORIZON-1),
        )

        dataloader = DataLoader(dataset, batch_size=cfg.DATALOADER_BATCH_SIZE, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            acc = [0 for k in range(cfg.HORIZON+1)]
            tot = [0 for k in range(cfg.HORIZON+1)]
            with tqdm.tqdm(dataloader, desc="Evaluating accuracy", leave=False) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(self.model.device, non_blocking=True))

                    obs = batch['obs']
                    actions = batch['action']
                    if HAS_GOAL_CONDITION:
                        goal = batch['goal']
                    else:
                        goal = None

                    obs = obs[:, :cfg.N_OBS_STEPS]
                    actions = actions[:, -cfg.HORIZON:]

                    actions_pred = self.model.predict(obs, do_sample=False, goal=goal)

                    for k in range(1, cfg.HORIZON+1):
                        tmp_actions = actions[:, :k]
                        tmp_actions_pred = actions_pred[:, :k]

                        acc[k] += torch.sum(
                            tmp_actions_pred[tmp_actions != self.ignore_index] ==
                            tmp_actions[tmp_actions != self.ignore_index]).item()
                        tot[k] += tmp_actions[tmp_actions != self.ignore_index].numel()

        for k in range(1, cfg.HORIZON+1):
            print(f"Accuracy (first {k} moves): {acc[k]}/{tot[k]} = {-1 if tot[k] == 0 else acc[k]/tot[k]}")
