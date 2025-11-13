"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.logging_utils import get_validation_metrics_for_states
from .models.diffusion_transition import DiffusionTransitionModel, DiffusionStepPolicy


class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.z_shape = cfg.z_shape
        self.frame_stack = cfg.frame_stack
        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay**self.frame_stack
        self.x_stacked_shape = list(cfg.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.is_spatial = len(self.x_shape) == 3  # pixel
        self.gt_cond_prob = cfg.gt_cond_prob  # probability to condition one-step diffusion o_t+1 on ground truth o_t
        self.gt_first_frame = cfg.gt_first_frame
        self.context_frames = cfg.context_frames  # number of context frames at validation time
        self.chunk_size = cfg.chunk_size
        self.calc_crps_sum = cfg.calc_crps_sum
        self.external_cond_dim = cfg.external_cond_dim
        self.uncertainty_scale = cfg.uncertainty_scale
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.validation_step_outputs = []
        self.min_crps_sum = float("inf")
        self.learnable_init_z = cfg.learnable_init_z
        self.crps_group_size = getattr(cfg, "crps_group_size", None)

        super().__init__(cfg)

    def _build_model(self):
        self.transition_model = DiffusionTransitionModel(
            self.x_stacked_shape, self.z_shape, self.external_cond_dim, self.cfg.diffusion
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)
        if self.learnable_init_z:
            self.init_z = nn.Parameter(torch.randn(list(self.z_shape)), requires_grad=True)
        
        if self.cfg.schedule_matrix.build:
            self.matrix_model = DiffusionStepPolicy(
                self.x_stacked_shape, self.z_shape, self.cfg
            )
            self.train_k_step_outputs = []
            if self.cfg.schedule_matrix.positive_only:
                values = list(range(self.cfg.schedule_matrix.actions))           # [0,1]
            else:
                half = self.cfg.schedule_matrix.actions // 2
                values = list(range(-half, half + 1))                            # [-1,0,1] if actions=3
            self.deltas = torch.tensor(values, dtype=torch.long, device=self.device)

    def configure_optimizers(self):
        transition_params = list(self.transition_model.parameters())
        if self.learnable_init_z:
            transition_params.append(self.init_z)
        optimizer_dynamics = torch.optim.AdamW(
            transition_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )

        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        nonterminals = batch[-1]
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        if self.learnable_init_z:
            init_z = self.init_z[None].expand(batch_size, *self.z_shape)
        else:
            init_z = torch.zeros(batch_size, *self.z_shape)
            init_z = init_z.to(xs.device)

        return xs, conditions, masks, init_z

    def reweigh_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(weight, "(t fs) b ... -> t b fs ..." + " 1" * expand_dim, fs=self.frame_stack)
            loss = loss * weight

        return loss.mean()

    def training_step(self, batch, batch_idx):
        # training step for dynamics
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch) # (batch, video length, channels) --> (video length, batch, channels)
        # n_frames == length of the video
        n_frames, batch_size, _, *_ = xs.shape

        xs_pred = []
        loss = []
        z = init_z
        cum_snr = None
        for t in range(0, n_frames): # iterate over every frame in the video
            deterministic_t = None
            if random() <= self.gt_cond_prob or (t == 0 and random() <= self.gt_first_frame):
                deterministic_t = 0

            z_next, x_next_pred, l, cum_snr = self.transition_model(
                z, xs[t], conditions[t], deterministic_t=deterministic_t, cum_snr=cum_snr
            )

            z = z_next
            xs_pred.append(x_next_pred)
            loss.append(l)

        xs_pred = torch.stack(xs_pred)
        loss = torch.stack(loss)
        x_loss = self.reweigh_loss(loss, masks)
        loss = x_loss

        if batch_idx % 20 == 0:
            self.log_dict(
                {
                    "training/loss": loss,
                    "training/x_loss": x_loss,
                }
            )

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        output_dict = {
            "loss": loss,
            "xs_pred": self._unnormalize_x(xs_pred),
            "xs": self._unnormalize_x(xs),
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        if self.calc_crps_sum:
            # repeat batch for crps sum for time series prediction
            batch = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]
            
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        xs_pred_all = []
        z = init_z

        # context
        for t in range(0, self.context_frames // self.frame_stack):
            z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
            xs_pred.append(x_next_pred)

        # prediction
        while len(xs_pred) < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)

            chunk = [
                torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
            ]

            pyramid_height = self.sampling_timesteps + int(horizon * self.uncertainty_scale)
            pyramid = np.zeros((pyramid_height, horizon), dtype=int)
            for m in range(pyramid_height):
                for t in range(horizon):
                    pyramid[m, t] = m - int(t * self.uncertainty_scale)
            pyramid = np.clip(pyramid, a_min=0, a_max=self.sampling_timesteps, dtype=int)

            for m in range(pyramid_height):
                if self.transition_model.return_all_timesteps:
                    xs_pred_all.append(chunk)

                z_chunk = z.detach()
                for t in range(horizon):
                    print("Starting i is ")
                    i = min(pyramid[m, t], self.sampling_timesteps - 1)
                    print(i)
                    i_vecind = torch.full((batch_size,), i, device=z_chunk.device).long() # changed for testing new ddim
                    print("Leading to vecind")
                    print(i_vecind[:3])
                    chunk[t], z_chunk = self.transition_model.ddim_sample_step_vecind(
                        chunk[t], z_chunk, conditions[len(xs_pred) + t], i_vecind
                    )

                    print("the new chunk t is ", chunk[t][:3])

                    # theoretically, one shall feed new chunk[t] with last z_chunk into transition model again 
                    # to get the posterior z_chunk, and optionaly, with small noise level k>0 for stablization. 
                    # However, since z_chunk in the above line already contains info about updated chunk[t] in 
                    # our simplied math model, we deem it suffice to directly take this z_chunk estimated from 
                    # last z_chunk and noiser chunk[t]. This saves half of the compute from posterior steps. 
                    # The effect of the above simplification already contains stablization: we always stablize 
                    # (ddim_sample_step is never called with noise level k=0 above)

            z = z_chunk
            xs_pred += chunk

        xs_pred = torch.stack(xs_pred)
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweigh_loss(loss, masks)

        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all]
                limit = self.transition_model.sampling_timesteps
                for i in np.linspace(1, limit, 5, dtype=int):
                    xs_pred = xs_pred_all[i]
                    xs_pred = self._unnormalize_x(xs_pred)
                    metric_dict = get_validation_metrics_for_states(xs_pred, xs)
                    self.log_dict(
                        {f"{namespace}/{i}_sampling_steps_{k}": v for k, v in metric_dict.items()},
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )
            else:
                metric_dict = get_validation_metrics_for_states(xs_pred, xs)
                self.log_dict(
                    {f"{namespace}/{k}": v for k, v in metric_dict.items()},
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss
    
    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        self.validation_step_outputs.clear()

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return xs * std + mean
    
    def train_k_step(self, batch, batch_idx, namespace="train_k"):
        """ 
        On-policy training
        May need to take in optimizer.
        """
        batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
        deltas = self.deltas.to(self.device).long()
        if self.calc_crps_sum:
            batch_expanded = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]
        else:
            batch_expanded = batch
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch_expanded)
        n_frames, batch_size, *_ = xs.shape
        
        # hold predictions
        z = init_z
        xs_pred = []
        xs_pred_all = []

        # rl scores
        log_probs = []
        entropies = []
        values = []
        rewards = []

        # --------------------------------------------------
        # 1. Context frames (no RL)
        # --------------------------------------------------
        for t in range(self.context_frames // self.frame_stack):
            print("Test to see if this is on the GPU")
            z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
            xs_pred.append(x_next_pred)

        # --------------------------------------------------
        # 2. RL section — rollout future frames
        # --------------------------------------------------
        while len(xs_pred) < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)
            max_rl_steps = int((self.sampling_timesteps + int(horizon * self.uncertainty_scale)) * self.cfg.schedule_matrix.rollout_multiple)

            chunk = [
                torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
            ]

            # rollout multiple scales original length (to start)
            decision_tracker = torch.zeros(batch_size, device=self.device, dtype=torch.long) # what noise level is everything at right now; bounded 0<=dt< num_sampling_steps; in next version add horizon dimension
            max_idx = self.sampling_timesteps - 1
            for m in range(max_rl_steps): # ensures max num of rl steps finite
                print(f"Taking RL step {m}")
                if torch.all(decision_tracker >= max_idx):
                    break
                if self.transition_model.return_all_timesteps: # can keep these lines
                    xs_pred_all.append(chunk)
                z_chunk = z.detach() # can keep detachment
                for t in range(horizon):
                    # get logits for the informed schedule
                    noise_idx = decision_tracker.long().to(device=self.device)
                    print("Noise idx looks like", noise_idx)
                    logits, value = self.matrix_model(chunk[t], z_chunk, noise_idx)
                    dist = Categorical(logits=logits)

                    # sample a decision, append the results
                    action = dist.sample()
                    print("Actions ", action)
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)
                    entropies.append(dist.entropy())
                    action_delta = deltas.to(self.device)[action]
                    print("Actions mapped to ", action_delta)
                    # need latter three to be disjoint
                    can_denoise = (decision_tracker < max_idx)
                    denoise_step = (action_delta > 0) & can_denoise
                    noise_step = (action_delta < 0)
                    flat_step = ~(denoise_step | noise_step) # either we chose not to step OR (we chose not to/cannot denoise (aka ~denoise) AND chose not to noise (aka ~noise_step))

                    # broadcast mask
                    denoise_step_mask_x = denoise_step.view(batch_size, *([1] * (chunk[t].dim() - 1)))
                    denoise_step_mask_z = denoise_step.view(batch_size, *([1] * (z_chunk.dim() - 1)))
                    flat_step_mask_x = flat_step.view(batch_size, *([1] * (chunk[t].dim() - 1)))
                    flat_step_mask_z = flat_step.view(batch_size, *([1] * (z_chunk.dim() - 1)))
                    # ....

                    # renoise those who need it (not implemented yet -- or will this go after ddim?)
                    
                    # take a ddim step on everybody 
                    new_chunk_t, new_chunk_z = self.transition_model.ddim_sample_step_vecind(
                        chunk[t], z_chunk, conditions[len(xs_pred) + t],
                        index_vec=noise_idx
                    )

                    # keep non-updated the same noise level (option to hold value or resample)
                    print("The predicted chunk t is ", new_chunk_t[:5, :])
                    chunk[t] = torch.where(denoise_step_mask_x, new_chunk_t, chunk[t])
                    z_chunk = torch.where(denoise_step_mask_z, new_chunk_z, z_chunk)

                    # update the decision tracker
                    decision_tracker = decision_tracker + action_delta
                    decision_tracker = decision_tracker.clamp(0, max_idx)

                    # hygiene
                    del new_chunk_t, new_chunk_z
                    del noise_idx, action_delta
                    del denoise_step_mask_x, denoise_step_mask_z
                    del flat_step_mask_x, flat_step_mask_z
                    torch.cuda.empty_cache()

                z = z_chunk
            xs_pred += chunk

        # at this point xs_pred is list length n_frames; stack etc.
        xs_pred = torch.stack(xs_pred)      # (T, B, fs*C, ...)
        xs_gt = xs

        # -----------------------
        # 3. compute scalar reward (e.g. -CRPS or -MSE for now)
        # -----------------------
        # for a very first pass: simple MSE over prediction region
        loss_per_elem = F.mse_loss(xs_pred, xs_gt, reduction="none")
        loss = self.reweigh_loss(loss_per_elem, masks)
        reward = -loss.detach()   # higher reward = lower loss

        # -----------------------
        # 4. vanilla REINFORCE with entropy bonus (batch baseline)
        # -----------------------
        if log_probs:
            log_probs = torch.stack(log_probs, dim=0)    # (N_decisions, B)
            entropies = torch.stack(entropies, dim=0)    # (N_decisions, B)

            # simple baseline: mean reward over batch
            baseline = reward.mean()
            advantage = (reward - baseline).detach()

            # broadcast advantage across decisions
            # shape match: (N_decisions, B)
            adv = advantage.expand_as(log_probs)

            policy_loss = -(adv * log_probs).mean()
            entropy_loss = -self.cfg.schedule_matrix.entropy_beta * entropies.mean()

            total_loss = policy_loss + entropy_loss
            total_loss.backward()

        return {"xs_pred": self._unnormalize_x(rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)),
                "loss": loss}
    @property
    def device(self):
        ''' TEMPORARY -- WWHEN OVERHAUL PL NEED TO REMOVE'''
        try:
            return next(self.parameters()).device
        except StopIteration:
            # no parameters — fallback
            return torch.device("cpu")
