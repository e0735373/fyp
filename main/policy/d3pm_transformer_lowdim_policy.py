import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange, reduce, einsum
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from main.common.pytorch_util import one_hot_with_ignore_index
from main.model.common.normalizer import LinearNormalizer
from main.policy.base_lowdim_policy import BaseLowdimPolicy
from main.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from main.model.diffusion.mask_generator import LowdimMaskGenerator
from main.policy.d3pm_interpolant import D3PM, UniformTimeDistribution, DiscreteMaskedPrior, DiscreteCosineNoiseSchedule, DiscreteUniformPrior

USE_MASK_DISCRETE_D3PM = False

class D3PMTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            hybrid_loss_coeff=0.001,
            loss_reweighting=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.loss_reweighting = loss_reweighting

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # D3PM stuff
        self.num_classes = action_dim

        self.d3pm_scheduler = D3PM(
            time_distribution=UniformTimeDistribution(
                discrete_time=True,
                nsteps=self.num_inference_steps
            ),
            prior_distribution=(
                DiscreteMaskedPrior(
                    num_classes=self.num_classes
                )
                if USE_MASK_DISCRETE_D3PM else
                DiscreteUniformPrior(
                    num_classes=self.num_classes
                )
            ),
            noise_schedule=DiscreteCosineNoiseSchedule(
                nsteps=self.num_inference_steps
            ),
            device=torch.device("cuda"),
        )

        self.ignore_index = -100

    def model_predict(self, traj, t, cond):
        # note: this returns unnormalized logits!
        assert len(traj.shape) == 2 and traj.shape[1] == self.n_action_steps
        traj_onehot = one_hot_with_ignore_index(traj, self.num_classes, ignore_index=self.ignore_index)
        return self.model(traj_onehot, t, cond)

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        scheduler = self.noise_scheduler

        if USE_MASK_DISCRETE_D3PM:
            trajectory = (torch.ones_like(condition_data, device=condition_data.device) * (self.num_classes - 1)).long()
            init_maskable_mask = ~condition_mask
        else:
            trajectory = torch.argmax(self._gen_noise(condition_data.shape, generator), dim=-1)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in tqdm.tqdm(scheduler.timesteps, desc="Inference", leave=False):
            # broadcast t
            timesteps = torch.broadcast_to(t, (trajectory.shape[0],)).to(trajectory.device)

            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = self.model_predict(trajectory, timesteps, cond)
            model_output = self.normalizer['action'].unnormalize(model_output)

            if USE_MASK_DISCRETE_D3PM:
                scores = torch.log_softmax(model_output, dim=-1)
                scores[:,:,-1] = -1000

                x0_scores, x0 = scores.max(-1)
                trajectory = topk_decoding(
                    x0,
                    x0_scores,
                    'stochastic0.5-linear',
                    init_maskable_mask,
                    t,
                    scheduler.num_inference_steps,
                    self.ignore_index
                )
            else:
                # 3. compute previous image: x_t -> x_t-1
                trajectory = self.d3pm_scheduler.step(
                    model_out=model_output,
                    t=timesteps,
                    xt=trajectory,
                    model_out_is_logits=True,
                )

            assert trajectory.shape == condition_data.shape

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype).argmax(dim=-1)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError("Not implemented yet")

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        action_pred = nsample

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            raise NotImplementedError("Not implemented yet")

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            raise NotImplementedError("Not implemented yet")
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        obs = self.normalizer['obs'].normalize(batch['obs'])
        action = batch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            assert self.pred_action_steps_only
            if self.pred_action_steps_only:
                # To = self.n_obs_steps
                # start = To - 1
                # end = start + self.n_action_steps
                # trajectory = action[:,start:end]
                trajectory = action # we do this work in transformeragent
        else:
            raise NotImplementedError("Not implemented yet")

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.d3pm_scheduler.interpolate(
            torch.where(trajectory != self.ignore_index,
                        trajectory,
                        torch.randint(0, self.num_classes, trajectory.shape, device=trajectory.device)),
            timesteps)

        assert noisy_trajectory.shape == trajectory.shape

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the x_0 trajectory logits
        predicted_x0_logits = self.model_predict(noisy_trajectory, timesteps, cond)

        # compute loss
        mask_loss = torch.where(trajectory != self.ignore_index,
                                torch.tensor(1.0, device=trajectory.device),
                                torch.tensor(0.0, device=trajectory.device))
        loss = self.d3pm_scheduler.loss(
            logits=predicted_x0_logits,
            target=trajectory,
            xt=noisy_trajectory,
            time=timesteps,
            mask=mask_loss,
            loss_reweight=self.loss_reweighting,
        )

        return loss.mean()

    # ========= D3PM stuff ============
    def _gen_noise(self, traj_shape, generator=None):
        return torch.rand(
            size=(*traj_shape, self.num_classes),
            dtype=torch.float64,
            device=self.device,
            generator=generator)



def topk_masking(scores, cutoff_len, stochastic=False, temp=0.5):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def topk_decoding(
        x0,
        x0_scores,
        decoding_strategy,
        init_maskable_mask,
        t,
        max_step,
        noise
    ):
        # decoding_strategy needs to take the form of "<topk_mode>-<schedule>"
        topk_mode, schedule = decoding_strategy.split("-")

        # select rate% not confident tokens, ~1 -> 0
        if schedule == "linear":
            rate = t / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
        # set the scores of unmaskable symbols to a large value so that they will never be selected
        _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError

        ### recovered tokens can also be remasked based on current scores
        masked_to_noise = lowest_k_mask
        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt