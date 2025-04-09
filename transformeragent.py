import os
import hydra
import torch
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import einops

from main.common.pytorch_util import dict_apply, optimizer_to
from main.dataset.base_dataset import BaseLowdimDataset
from main.common.checkpoint_util import TopKCheckpointManager
from main.common.json_logger import JsonLogger
from main.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
from main.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from main.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from main.policy.d3pm_transformer_lowdim_policy import D3PMTransformerLowdimPolicy
from typing import Tuple
from main.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from main.common.pytorch_util import one_hot_with_ignore_index

def TransformerForDiffusionFromConfig(cls, causal_attn=False):
    return TransformerForDiffusion(
        input_dim=cls.ACTION_DIM+cls.HAS_MASK_TOKEN,
        output_dim=cls.ACTION_DIM+cls.HAS_MASK_TOKEN,
        horizon=cls.HORIZON,
        n_obs_steps=(cls.N_OBS_STEPS + cls.HAS_GOAL_CONDITION) * (cls.IMAGE_SIZE),
        cond_dim=cls.COND_DIM,
        n_layer=cls.N_LAYER,
        n_head=cls.N_HEAD,
        n_emb=cls.N_EMB,
        p_drop_emb=cls.P_DROP_EMB,
        p_drop_attn=cls.P_DROP_ATTN,
        causal_attn=causal_attn,
        time_as_cond=cls.TIME_AS_COND,
        obs_as_cond=cls.OBS_AS_COND,
        n_cond_layers=cls.N_COND_LAYERS,
    )

def DDPMSchedulerFromConfig(cls):
    return DDPMScheduler(
        num_train_timesteps=cls.NOISESCHEDULER_NUM_TRAIN_TIMESTEPS,
        beta_start=cls.NOISESCHEDULER_BETA_START,
        beta_end=cls.NOISESCHEDULER_BETA_END,
        beta_schedule=cls.NOISESCHEDULER_BETA_SCHEDULE,
        variance_type=cls.NOISESCHEDULER_VARIANCE_TYPE,
        clip_sample=cls.NOISESCHEDULER_CLIP_SAMPLE,
        prediction_type=cls.NOISESCHEDULER_PREDICTION_TYPE,
    )

def DiffusionTransformerLowdimPolicyFromConfig(cls):
    return DiffusionTransformerLowdimPolicy(
        model=TransformerForDiffusionFromConfig(cls),
        noise_scheduler=DDPMSchedulerFromConfig(cls),
        horizon=cls.HORIZON,
        obs_dim=cls.PIXEL_OUT_DIM,
        action_dim=cls.ACTION_DIM+cls.HAS_MASK_TOKEN,
        n_action_steps=cls.N_ACTION_STEPS,
        n_obs_steps=(cls.N_OBS_STEPS + cls.HAS_GOAL_CONDITION) * (cls.IMAGE_SIZE),
        num_inference_steps=cls.NOISESCHEDULER_NUM_TRAIN_TIMESTEPS,
        obs_as_cond=cls.OBS_AS_COND,
        pred_action_steps_only=True
    )

def D3PMTransformerLowdimPolicyFromConfig(cls):
    return D3PMTransformerLowdimPolicy(
        model=TransformerForDiffusionFromConfig(cls),
        noise_scheduler=DDPMSchedulerFromConfig(cls),
        horizon=cls.HORIZON,
        obs_dim=cls.PIXEL_OUT_DIM,
        action_dim=cls.ACTION_DIM+cls.HAS_MASK_TOKEN,
        n_action_steps=cls.N_ACTION_STEPS,
        n_obs_steps=(cls.N_OBS_STEPS + cls.HAS_GOAL_CONDITION) * (cls.IMAGE_SIZE),
        num_inference_steps=cls.NOISESCHEDULER_NUM_TRAIN_TIMESTEPS,
        obs_as_cond=cls.OBS_AS_COND,
        pred_action_steps_only=True,
        loss_reweighting=cls.LOSS_REWEIGHTING,
    )


class TransformerAgent(torch.nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.transformer = TransformerForDiffusionFromConfig(cls, causal_attn=True)

        self.obs_emb = torch.nn.Linear(self.cls.PIXEL_DIM, self.cls.PIXEL_OUT_DIM)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda')

    def forward(self, obs, gt_action=None, generate=False, do_sample=True, goal=None):
        # obs: (B, n_obs_steps, n * n) of int[0, pixel_dim)
        # gt_action: (B, horizon) of int[0, action_dim)
        # output: (B, horizon, action_dim) of probabilities
        # goal: (B, n * n) of int[0, pixel_dim)

        obs = obs.long()

        # one-hot encoding of obs
        obs = torch.nn.functional.one_hot(obs, self.cls.PIXEL_DIM).float().to(self.device)
        # (B, n_obs_steps, n * n, pixel_dim)
        obs = self.obs_emb(obs)
        # (B, n_obs_steps, n * n, pixel_out_dim)
        obs = einops.rearrange(obs, "B n i p -> B (n i) p")
        # (B, n_obs_steps * n * n, pixel_out_dim)

        if goal is not None:
            goal = goal.long()

            # one-hot encoding of goal
            goal = torch.nn.functional.one_hot(goal, self.cls.PIXEL_DIM).float().to(self.device)
            # (B, n * n, pixel_dim)
            goal = self.obs_emb(goal)
            # (B, n * n, pixel_out_dim)

            obs = torch.cat([goal, obs], dim=1)
            # (B, (n_obs_steps+1) * n * n, pixel_out_dim)

        if not generate:
            # same shape as one-hot encoding of action
            action = one_hot_with_ignore_index(gt_action, self.cls.ACTION_DIM+self.cls.HAS_MASK_TOKEN).float().to(self.device)
            # (B, horizon, action_dim)

            output = self.transformer(
                sample=action,
                timestep=0,
                cond=obs)
            # (B, horizon+1, action_dim)
            output = output[:, :-1, :]

            loss = None
            if gt_action is not None:
                gt_action = gt_action.long()
                loss = self.ce_loss(
                    einops.rearrange(output, "B horizon action_dim -> (B horizon) action_dim"),
                    einops.rearrange(gt_action, "B horizon -> (B horizon)"))

            return output, loss
        else:
            return self.transformer.generate(
                timestep=0,
                cond=obs,
                do_sample=do_sample
            )

    def compute_loss(self, batch):
        obs = batch["obs"]
        gt_action = batch["action"]

        dataset_horizon = self.cls.N_OBS_STEPS + self.cls.HORIZON - 1
        assert len(obs.shape) == 3 and obs.shape[1] == dataset_horizon
        assert len(gt_action.shape) == 2 and gt_action.shape[1] == dataset_horizon

        obs = obs[:, :self.cls.N_OBS_STEPS, :]
        gt_action = gt_action[:, -self.cls.HORIZON:]

        if "goal" in batch:
            goal = batch["goal"]
        else:
            goal = None

        _, loss = self.forward(obs, gt_action, goal=goal)
        return loss

    def predict(self, obs, do_sample=False, goal=None):
        return self.forward(obs, generate=True, do_sample=do_sample, goal=goal)

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.transformer.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas)
        )

class DiffusionAgent(torch.nn.Module):
    def __init__(self, cls):
        super().__init__()

        self.cls = cls
        self.policy = DiffusionTransformerLowdimPolicyFromConfig(cls)
        self.obs_emb = torch.nn.Linear(self.cls.PIXEL_DIM, self.cls.PIXEL_OUT_DIM)
        self.device = torch.device('cuda')

        self.policy.set_normalizer(self.get_normalizer())

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.preprocess_action(self.get_all_actions()), last_n_dims=1, mode=mode, **kwargs)

        return normalizer

    def preprocess_obs(self, obs):
        # obs: (B, n_obs_steps, n * n) of int[0, pixel_dim)
        obs = obs.long()
        # one-hot encoding of obs
        obs = torch.nn.functional.one_hot(obs, self.cls.PIXEL_DIM).float().to(self.device)
        # (B, n_obs_steps, n * n, pixel_dim)
        obs = self.obs_emb(obs)
        # (B, n_obs_steps, n * n, pixel_out_dim)
        obs = einops.rearrange(obs, "B n i p -> B (n i) p")
        # (B, n_obs_steps * n * n, pixel_out_dim)
        return obs

    def get_all_actions(self):
        all_actions = torch.zeros((self.cls.HORIZON, self.cls.ACTION_DIM+self.cls.HAS_MASK_TOKEN), dtype=torch.long)
        for r in range(len(all_actions)):
            for c in range(self.cls.ACTION_DIM+self.cls.HAS_MASK_TOKEN):
                all_actions[r, c] = c
        return all_actions

    def preprocess_action(self, action):
        # action: (B, horizon) of int[0, action_dim)
        action = action.long()
        # one-hot encoding of action if not -1, else uniform
        action = one_hot_with_ignore_index(action, self.cls.ACTION_DIM+self.cls.HAS_MASK_TOKEN).float().to(self.device)
        # (B, horizon, action_dim)

        # transform probability to logits
        action = torch.log(action + 1e-8)
        return action

    def compute_loss(self, batch):
        obs = batch["obs"]
        gt_action = batch["action"]

        dataset_horizon = self.cls.N_OBS_STEPS + self.cls.HORIZON - 1
        assert len(obs.shape) == 3 and obs.shape[1] == dataset_horizon
        assert len(gt_action.shape) == 2 and gt_action.shape[1] == dataset_horizon

        # obs: (B, n_obs_steps, n * n) of int[0, pixel_dim)
        # gt_action: (B, horizon) of int[0, action_dim)
        # output: (B, horizon, action_dim) of probabilities

        obs = obs[:, :self.cls.N_OBS_STEPS, :]
        gt_action = gt_action[:, -self.cls.HORIZON:]

        if "goal" in batch:
            goal = batch["goal"]
            goal = goal.unsqueeze(1)
            obs = torch.cat([goal, obs], dim=1)

        batch["obs"] = self.preprocess_obs(obs)
        batch["action"] = self.preprocess_action(gt_action)

        return self.policy.compute_loss(batch)

    def predict(self, obs, do_sample=False, goal=None):
        if goal is not None:
            goal = goal.unsqueeze(1)
            obs = torch.cat([goal, obs], dim=1)

        obs = self.preprocess_obs(obs)
        action_logits = self.policy.predict_action({
            "obs": obs,
        })["action_pred"]

        if do_sample:
            action = torch.distributions.Categorical(logits=action_logits).sample()
        else:
            action = torch.argmax(action_logits, dim=-1)

        assert action.shape[1] == self.cls.HORIZON and len(action.shape) == 2
        return action

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.policy.get_optimizer(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas)
        )

class D3PMAgent(torch.nn.Module):
    def __init__(self, cls):
        super().__init__()

        self.cls = cls
        self.policy = D3PMTransformerLowdimPolicyFromConfig(cls)
        self.obs_emb = torch.nn.Linear(self.cls.PIXEL_DIM, self.cls.PIXEL_OUT_DIM)
        self.device = torch.device('cuda')

        self.policy.set_normalizer(self.get_normalizer())

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.preprocess_action(self.get_all_actions()), last_n_dims=1, mode=mode, **kwargs)

        return normalizer

    def preprocess_obs(self, obs):
        # obs: (B, n_obs_steps, n * n) of int[0, pixel_dim)
        obs = obs.long()
        # one-hot encoding of obs
        obs = torch.nn.functional.one_hot(obs, self.cls.PIXEL_DIM).float().to(self.device)
        # (B, n_obs_steps, n * n, pixel_dim)
        obs = self.obs_emb(obs)
        # (B, n_obs_steps, n * n, pixel_out_dim)
        obs = einops.rearrange(obs, "B n i p -> B (n i) p")
        # (B, n_obs_steps * n * n, pixel_out_dim)
        return obs

    def get_all_actions(self):
        all_actions = torch.zeros((self.cls.HORIZON, self.cls.ACTION_DIM+self.cls.HAS_MASK_TOKEN), dtype=torch.long)
        for r in range(len(all_actions)):
            for c in range(self.cls.ACTION_DIM+self.cls.HAS_MASK_TOKEN):
                all_actions[r, c] = c
        return all_actions

    def preprocess_action(self, action):
        # action: (B, horizon) of int[0, action_dim)
        action = action.long()
        return action

    def compute_loss(self, batch):
        obs = batch["obs"]
        gt_action = batch["action"]

        dataset_horizon = self.cls.N_OBS_STEPS + self.cls.HORIZON - 1
        assert len(obs.shape) == 3 and obs.shape[1] == dataset_horizon
        assert len(gt_action.shape) == 2 and gt_action.shape[1] == dataset_horizon

        # obs: (B, n_obs_steps, n * n) of int[0, pixel_dim)
        # gt_action: (B, horizon) of int[0, action_dim)
        # output: (B, horizon, action_dim) of probabilities

        obs = obs[:, :self.cls.N_OBS_STEPS, :]
        gt_action = gt_action[:, -self.cls.HORIZON:]

        if "goal" in batch:
            goal = batch["goal"]
            goal = goal.unsqueeze(1)
            obs = torch.cat([goal, obs], dim=1)

        batch["obs"] = self.preprocess_obs(obs)
        batch["action"] = self.preprocess_action(gt_action)

        return self.policy.compute_loss(batch)

    def predict(self, obs, do_sample=False, goal=None):
        if goal is not None:
            goal = goal.unsqueeze(1)
            obs = torch.cat([goal, obs], dim=1)

        obs = self.preprocess_obs(obs)
        action_pred = self.policy.predict_action({
            "obs": obs,
        })["action_pred"]

        return action_pred

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.policy.get_optimizer(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas)
        )
