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
from algorithms.common.metrics import crps_quantile_sum

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
        # for learning the k schedule
        self.crps_group_size = getattr(cfg, "crps_group_size", None)
        self.log_k_matrix = getattr(cfg.schedule_matrix, "log_k_matrix", False)
        self.k_matrix_log_every = getattr(cfg.schedule_matrix, "k_matrix_log_every", 0)
        self.k_matrix_batches = getattr(cfg.schedule_matrix, "k_matrix_batches", 0)
        self.k_viz_max_steps = getattr(cfg.schedule_matrix, "k_viz_max_steps", 0)
        self.k_viz_summary_steps = getattr(cfg.schedule_matrix, "k_viz_summary_steps", [])
        self.raw_reward = getattr(cfg.schedule_matrix, "raw_reward", True)
        self.raw_reward_crps = getattr(cfg.schedule_matrix, "raw_reward_crps", False)
        self.step_reward = getattr(cfg.schedule_matrix, "step_reward", False)
        self.step_reward_crps = getattr(cfg.schedule_matrix, "step_reward_crps", False)
        self.difference_step_reward = getattr(cfg.schedule_matrix, "difference_step_reward", False)

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
        loss = torch.stack(loss) # we collected as we went...
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
        frameroller=0
        max_roller_mod = float('inf') if self.cfg.schedule_matrix.max_roller < 0 else self.cfg.schedule_matrix.max_roller

        while len(xs_pred) < n_frames and frameroller < max_roller_mod:
            
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)
            frameroller+=horizon

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
                    i = min(pyramid[m, t], self.sampling_timesteps - 1)

                    i_vecind = torch.full((batch_size,), i, device=z_chunk.device).long() # changed for testing new ddim
                    chunk[t], z_chunk = self.transition_model.ddim_sample_step_vecind(
                        chunk[t], z_chunk, conditions[len(xs_pred) + t], i_vecind
                    )

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
        loss = F.mse_loss(xs_pred, xs[:xs_pred.shape[0]], reduction="none") # Could not be calcd until we finished rolling out
        loss = self.reweigh_loss(loss, masks[:xs_pred.shape[0]])
        xs = rearrange(xs[:xs_pred.shape[0]], "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        xs_pred = rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack) # (T, B, fs*C, ...) is how we got it, send to (T * FS , B , C)

        xs = self._unnormalize_x(xs)
        xs_pred = self._unnormalize_x(xs_pred)

        if not self.is_spatial:
            if self.transition_model.return_all_timesteps:
                xs_pred_all = [torch.stack(item) for item in xs_pred_all[:xs_pred.shape[0]]] # need to ensure later.
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

    def train_k_step_multiple_densified(self, batch, batch_idx, namespace="train_k"):
        """ 
        On-policy training
        Current version we make one decision for each m and t which means we mave m*t decisions.
        """
        batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
        deltas = self.deltas.to(self.device).long()
        if self.calc_crps_sum:
            batch_expanded = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]
        else:
            batch_expanded = batch

        # Initialize relevant for the rollout
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch_expanded)
        n_frames, batch_size, *_ = xs.shape
        z = init_z
        xs_pred, log_probs, entropies, values, rewards, dense_rewards, rollout_lengths = [], [], [], [], [], [], []

        # Warm up the context
        with torch.no_grad():
            for t in range(self.context_frames // self.frame_stack):
                z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
                xs_pred.append(x_next_pred)

        # Execute the rollout, require grad if training
        frameroller = 0
        k_histories = []
        max_roller_mod = float('inf') if self.cfg.schedule_matrix.max_roller < 0 else self.cfg.schedule_matrix.max_roller
        while len(xs_pred) < n_frames and frameroller < max_roller_mod:
            
            # print(f"Rolling {frameroller}-th time")
            # Determine the horizon
            if self.chunk_size > 0:
                horizon = min(n_frames - len(xs_pred), self.chunk_size)
            else:
                horizon = n_frames - len(xs_pred)
            frameroller+=horizon
            
            chunk = [
                torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
            ]

            # Rollout this horizon
            decision_tracker = torch.zeros((horizon, batch_size), device=self.device, dtype=torch.long) # What noise are we at? Bounded by max_idx
            max_idx = self.sampling_timesteps - 1 # Do not exceed this noise level so that calls to DDIM are stable
            max_rl_steps = int((self.sampling_timesteps + int(horizon * self.uncertainty_scale)) * self.cfg.schedule_matrix.rollout_multiple) # Ensure finite horiuzon
            k_matrix_predicted = torch.zeros((max_rl_steps, horizon, batch_size))
            for m in range(max_rl_steps):
                # print(f"Taking RL step {m}")
                z_chunk = z  # already kept as a no-grad tensor from previous steps
                for t in range(horizon):

                    # Calculate logits
                    noise_idx = decision_tracker[t].clone().detach()  # already long + on device
                    logits, value = self.matrix_model(chunk[t], z_chunk, noise_idx) # [batch, action_space]
                    dist = Categorical(logits=logits) # Torch distro to sample from

                    # Take action and append reuslts
                    action = dist.sample() # Action indices simulated on policy 
                    action_delta = deltas[action] # Map to actual action (deltas already on device)
                    log_prob = dist.log_prob(action) # index using actual action
                    log_probs.append(log_prob)
                    entropies.append(dist.entropy())
                    
                    # Create latter three to be disjoint, our masks for RL transitions
                    can_denoise = (decision_tracker[t] < max_idx)
                    denoise_step = (action_delta > 0) & can_denoise
                    noise_step = (action_delta < 0)
                    flat_step = ~(denoise_step | noise_step) # either we chose not to step OR (we chose not to/cannot denoise (aka ~denoise) AND chose not to noise (aka ~noise_step))

                    # Broadcast masks for the later on torch.where
                    denoise_step_mask_x = denoise_step.view(batch_size, *([1] * (chunk[t].dim() - 1)))
                    denoise_step_mask_z = denoise_step.view(batch_size, *([1] * (z_chunk.dim() - 1)))
                    noise_step_mask_x = noise_step.view(batch_size, *([1] * (chunk[t].dim() - 1)))
                    noise_step_mask_z = noise_step.view(batch_size, *([1] * (z_chunk.dim() - 1)))

                    # DDIM Step can occur on everyone because we clamped noise_idx
                    with torch.no_grad():
                        new_chunk_t, new_chunk_z = self.transition_model.ddim_sample_step_vecind(
                            chunk[t], z_chunk, conditions[len(xs_pred) + t],
                            index_vec=noise_idx
                        )

                    if not self.cfg.schedule_matrix.positive_only:
                        # Make this sample noise
                        with torch.no_grad():
                            noised_chunk_t = self.transition_model.q_renoise_from_ddim_index(
                                x_t=chunk[t],
                                index_vec=noise_idx,
                                n=action_delta
                            )

                    # Keep non-updated the same noise level for now (option to "copy over" value or DDIM down, forward noise back up)
                    chunk[t] = torch.where(denoise_step_mask_x, new_chunk_t, chunk[t]) # Copy over logic
                    z_chunk = torch.where(denoise_step_mask_z, new_chunk_z, z_chunk) # Copy over logic
                    # Noise data where we need to. Keep latent via copy over.
                    if not self.cfg.schedule_matrix.positive_only:
                        chunk[t] = torch.where(noise_step_mask_x, noised_chunk_t, chunk[t]) # what to do with the latent? can keep it 

                    # Action deltas map to actual decision
                    decision_tracker[t] = decision_tracker[t] + action_delta
                    decision_tracker[t] = decision_tracker[t].clamp(0, max_idx)

                    # print(f"{m},{t} state dist is ", torch.unique(decision_tracker[t], return_counts=True))
                    # print(f"{m},{t} decision dist is ", torch.unique(action_delta, return_counts=True))

                    # Clean up GPU Utilization (delete when PL gets implemented)
                    del new_chunk_t, new_chunk_z
                    del denoise_step_mask_x, denoise_step_mask_z
                    del dist, logits, value, action, action_delta, noise_idx
                    # NOTE: no torch.cuda.empty_cache() here; it's very slow and doesn't reduce peak memory
                    
                    k_matrix_predicted[m][t] = decision_tracker[t]

                # get post m-step rewards now
                with torch.no_grad():
                    if self.step_reward:
                        stacked_chunk = torch.stack(chunk)
                        stacked_x = xs[len(xs_pred):len(xs_pred) + horizon] # (Horizon, B, C)

                        if not self.step_reward_crps:
                            
                            # we made t deicsions that round and will allocate 
                            step_loss = F.mse_loss(stacked_chunk, stacked_x, reduction="none")
                            rl_reweighed_step_loss = self.reweigh_loss_rl(step_loss, masks[len(xs_pred):len(xs_pred) + horizon])
                            rl_reweighed_step_loss = rl_reweighed_step_loss.unsqueeze(0).expand(horizon, -1)# (B) --> (1,B)--> (N_DEC_IN_M, B), assuming N_DEC equal to horizon
                            dense_rewards.append(-rl_reweighed_step_loss) # (N_DEC, B_eff)
                        else:
                            if self.calc_crps_sum:
                                # xs_pred: (T, B_eff, C)
                                # reshape to (S, T, B, C)
                                T, _, C = stacked_chunk.shape
                                S = self.calc_crps_sum
                                B0 = stacked_chunk.shape[1] // S # where S is number of "crps resamples"

                                xs_pred_samples = stacked_chunk.permute(1, 0, 2)           # (B_eff, T, C)
                                xs_pred_samples = xs_pred_samples.view(S, B0, T, C)  # (S, B0, T, C)
                                xs_pred_samples = xs_pred_samples.permute(0, 2, 1, 3)  # (S, T, B0, C)

                                xs_gt_b = stacked_x.permute(1, 0, 2)            # (B_eff, T, C)
                                xs_gt_b = xs_gt_b.view(S, B0, T, C)
                                xs_gt_b = xs_gt_b[0]                        # take the first sample (truth identical)
                                xs_gt_b = xs_gt_b.permute(1, 0, 2)          # (T, B0, C)
                                # have (S, T, B, C) and (T, B, C)
                                crps_per_item = self.crps_per_batch_element(xs_pred_samples, xs_gt_b, is_inference=True) # shape is (B) -- the sole reward per batch item gotten at the end of rollout
                                crps_per_item = crps_per_item.unsqueeze(0).expand(horizon, -1).unsqueeze(-1).expand(-1, -1, S).reshape(horizon, B0*S) # (T, B) instead of (T, B_eff)
                                dense_rewards.append(-crps_per_item)

            # print(k_matrix_predicted.mean(dim=-1))
            
            k_histories.append(k_matrix_predicted)
            xs_pred += chunk
            rollout_lengths.append(max_rl_steps) # later with early breaking append with actual length

        # At this point xs_pred is list length n_frames; stack etc.
        xs_pred = torch.stack(xs_pred)      # (T, B, fs*C, ...)
        xs_gt = xs[:xs_pred.shape[0]]
        masks = masks[:xs_pred.shape[0]]
        # Then, stack the resulting tensors
        k_histories = torch.stack(k_histories)
        if self.step_reward:
            dense_rewards = torch.stack(dense_rewards)
            if self.difference_step_reward:
                first = torch.zeros_like(dense_rewards[0:1])   # shape (1,n_dec,B)
                rest  = dense_rewards[1:] - dense_rewards[:-1] # shape (rest,n_dec,B)
                dense_diff = torch.cat([first, rest], dim=0)
                dense_rewards = dense_diff
            dense_reward = dense_rewards.reshape(-1, xs_gt.shape[1])

        # Unlike original self.validation_step we have not mapped (T, B, fs*C, ...) --> (T * fs , B , C) yet.
        with torch.no_grad():
            if self.calc_crps_sum:
                # xs_pred: (T, B_eff, C)
                # reshape to (S, T, B, C)
                T, _, C = xs_pred.shape
                S = self.calc_crps_sum
                B0 = xs_pred.shape[1] // S # where S is number of "crps resamples"

                xs_pred_samples = xs_pred.permute(1, 0, 2)           # (B_eff, T, C)
                xs_pred_samples = xs_pred_samples.view(S, B0, T, C)  # (S, B0, T, C)
                xs_pred_samples = xs_pred_samples.permute(0, 2, 1, 3)  # (S, T, B0, C)

                xs_gt_b = xs_gt.permute(1, 0, 2)            # (B_eff, T, C)
                xs_gt_b = xs_gt_b.view(S, B0, T, C)
                xs_gt_b = xs_gt_b[0]                        # take the first sample (truth identical)
                xs_gt_b = xs_gt_b.permute(1, 0, 2)          # (T, B0, C)

                crps_per_item = self.crps_per_batch_element(xs_pred_samples, xs_gt_b) # shape is (B) -- the sole reward per batch item gotten at the end of rollout
                crps_per_item = crps_per_item.repeat_interleave(S) # to make it shape (B_eff) so that we can use downstream in reward for all the decisions we made

                crps_batch = crps_quantile_sum(xs_pred_samples, xs_gt_b)

            loss_per_elem = F.mse_loss(xs_pred, xs_gt, reduction="none")
            loss = self.reweigh_loss(loss_per_elem, masks)
            rl_reweighed_loss = self.reweigh_loss_rl(loss_per_elem, masks)
            
        # Policy Gradient Steps
        if log_probs:
            # (N_decisions, B)
            log_probs = torch.stack(log_probs, dim=0)
            entropies = torch.stack(entropies, dim=0)

            # make reward negative because the raw metrics are meant to be minimized
            reward = - crps_per_item if self.raw_reward_crps else -rl_reweighed_loss
            reward = reward.unsqueeze(0).expand_as(log_probs) + dense_reward

            # reward: [B]
            baseline = reward.mean() # baseline: scalar
            advantage = (reward - baseline).detach()       # [B]

            # Expand advantage to match (N_decisions, B) as before it is just B.
            advantage_expanded = advantage

            # Vanilla REINFORCE objective. Minimize "policy_loss" to maximize reward
            # L = - E[ advantage * log π(a_t | s_t) ]
            policy_loss = -(advantage_expanded * log_probs).mean() # expanded to enable inplace mult

            # Entropy bonus (optional)
            entropy_loss = -self.cfg.schedule_matrix.entropy_beta * entropies.mean()

            total_loss = policy_loss + entropy_loss
            total_loss.backward()

            # print(f"Made {log_probs.shape[0]} decisions per batch")
            # print("Had the loss ", policy_loss.detach().cpu().item())
            # print(f"Had the advantage mean of {advantage.mean().item()} with std {advantage.std().item()}")

        
        del xs, xs_gt, chunk, conditions, init_z, z
        del log_probs, entropies, values, rewards, rollout_lengths

        return {"xs_pred": self._unnormalize_x(rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)),
                "loss": loss, "crps": crps_batch.item(), "total_loss": total_loss, "k_history": k_histories.squeeze(0)}


    def validate_k_step_multiple_densified(self, batch, batch_idx, namespace="train_k"):
        """ 
        On-policy training
        Current version we make one decision for each m and t which means we mave m*t decisions.
        """
        batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
        deltas = self.deltas.to(self.device).long()
        if self.calc_crps_sum:
            batch_expanded = [d[None].expand(self.calc_crps_sum, *([-1] * len(d.shape))).flatten(0, 1) for d in batch]
        else:
            batch_expanded = batch

        # Initialize relevant for the rollout
        xs, conditions, masks, *_, init_z = self._preprocess_batch(batch_expanded)
        n_frames, batch_size, *_ = xs.shape
        z = init_z
        xs_pred, log_probs, entropies, values, rewards, dense_rewards, rollout_lengths = [], [], [], [], [], [], []

        # Warm up the context
        with torch.no_grad():
            for t in range(self.context_frames // self.frame_stack):
                z, x_next_pred, _, _ = self.transition_model(z, xs[t], conditions[t], deterministic_t=0)
                xs_pred.append(x_next_pred)

        with torch.no_grad():
            # Execute the rollout
            frameroller = 0
            k_histories = []
            max_roller_mod = float('inf') if self.cfg.schedule_matrix.max_roller < 0 else self.cfg.schedule_matrix.max_roller
            while len(xs_pred) < n_frames and frameroller < max_roller_mod:
                
                # print(f"Rolling {frameroller}-th time")
                # Determine the horizon
                if self.chunk_size > 0:
                    horizon = min(n_frames - len(xs_pred), self.chunk_size)
                else:
                    horizon = n_frames - len(xs_pred)
                frameroller+=horizon
                
                chunk = [
                    torch.randn((batch_size,) + tuple(self.x_stacked_shape), device=self.device) for _ in range(horizon)
                ]

                # Rollout this horizon
                decision_tracker = torch.zeros((horizon, batch_size), device=self.device, dtype=torch.long) # What noise are we at? Bounded by max_idx
                max_idx = self.sampling_timesteps - 1 # Do not exceed this noise level so that calls to DDIM are stable
                max_rl_steps = int((self.sampling_timesteps + int(horizon * self.uncertainty_scale)) * self.cfg.schedule_matrix.rollout_multiple) # Ensure finite horiuzon
                k_matrix_predicted = torch.zeros((max_rl_steps, horizon, batch_size))
                for m in range(max_rl_steps):
                    # print(f"Taking RL step {m}")
                    z_chunk = z  # already kept as a no-grad tensor from previous steps
                    for t in range(horizon):

                        # Calculate logits
                        noise_idx = decision_tracker[t].clone().detach()  # already long + on device
                        logits, value = self.matrix_model(chunk[t], z_chunk, noise_idx) # [batch, action_space]
                        dist = Categorical(logits=logits) # Torch distro to sample from

                        # Take action and append reuslts
                        action = dist.sample() # Action indices simulated on policy 
                        action_delta = deltas[action] # Map to actual action (deltas already on device)
                        log_prob = dist.log_prob(action) # index using actual action
                        log_probs.append(log_prob)
                        entropies.append(dist.entropy())
                        
                        # Create latter three to be disjoint, our masks for RL transitions
                        can_denoise = (decision_tracker[t] < max_idx)
                        denoise_step = (action_delta > 0) & can_denoise
                        noise_step = (action_delta < 0)
                        flat_step = ~(denoise_step | noise_step) # either we chose not to step OR (we chose not to/cannot denoise (aka ~denoise) AND chose not to noise (aka ~noise_step))

                        # Broadcast masks for the later on torch.where
                        denoise_step_mask_x = denoise_step.view(batch_size, *([1] * (chunk[t].dim() - 1)))
                        denoise_step_mask_z = denoise_step.view(batch_size, *([1] * (z_chunk.dim() - 1)))
                        noise_step_mask_x = noise_step.view(batch_size, *([1] * (chunk[t].dim() - 1)))
                        noise_step_mask_z = noise_step.view(batch_size, *([1] * (z_chunk.dim() - 1)))

                        # DDIM Step can occur on everyone because we clamped noise_idx
                        with torch.no_grad():
                            new_chunk_t, new_chunk_z = self.transition_model.ddim_sample_step_vecind(
                                chunk[t], z_chunk, conditions[len(xs_pred) + t],
                                index_vec=noise_idx
                            )

                        if not self.cfg.schedule_matrix.positive_only:
                            # Make this sample noise
                            with torch.no_grad():
                                noised_chunk_t = self.transition_model.q_renoise_from_ddim_index(
                                    x_t=chunk[t],
                                    index_vec=noise_idx,
                                    n=action_delta
                                )

                        # Keep non-updated the same noise level for now (option to "copy over" value or DDIM down, forward noise back up)
                        chunk[t] = torch.where(denoise_step_mask_x, new_chunk_t, chunk[t]) # Copy over logic
                        z_chunk = torch.where(denoise_step_mask_z, new_chunk_z, z_chunk) # Copy over logic
                        # Noise data where we need to. Keep latent via copy over.
                        if not self.cfg.schedule_matrix.positive_only:
                            chunk[t] = torch.where(noise_step_mask_x, noised_chunk_t, chunk[t]) # what to do with the latent? can keep it 

                        # Action deltas map to actual decision
                        decision_tracker[t] = decision_tracker[t] + action_delta
                        decision_tracker[t] = decision_tracker[t].clamp(0, max_idx)

                        # print(f"{m},{t} state dist is ", torch.unique(decision_tracker[t], return_counts=True))
                        # print(f"{m},{t} decision dist is ", torch.unique(action_delta, return_counts=True))

                        # Clean up GPU Utilization (delete when PL gets implemented)
                        del new_chunk_t, new_chunk_z
                        del denoise_step_mask_x, denoise_step_mask_z
                        del dist, logits, value, action, action_delta, noise_idx
                        # NOTE: no torch.cuda.empty_cache() here; it's very slow and doesn't reduce peak memory
                        
                        k_matrix_predicted[m][t] = decision_tracker[t]

                    # get post m-step rewards now
                    with torch.no_grad():
                        if self.step_reward:
                            stacked_chunk = torch.stack(chunk)
                            stacked_x = xs[len(xs_pred):len(xs_pred) + horizon] # (Horizon, B, C)

                            if not self.step_reward_crps:
                                
                                # we made t deicsions that round and will allocate 
                                step_loss = F.mse_loss(stacked_chunk, stacked_x, reduction="none")
                                rl_reweighed_step_loss = self.reweigh_loss_rl(step_loss, masks[len(xs_pred):len(xs_pred) + horizon])
                                rl_reweighed_step_loss = rl_reweighed_step_loss.unsqueeze(0).expand(horizon, -1)# (B) --> (1,B)--> (N_DEC_IN_M, B), assuming N_DEC equal to horizon
                                dense_rewards.append(-rl_reweighed_step_loss) # (N_DEC, B_eff)
                            else:
                                if self.calc_crps_sum:
                                    # xs_pred: (T, B_eff, C)
                                    # reshape to (S, T, B, C)
                                    T, _, C = stacked_chunk.shape
                                    S = self.calc_crps_sum
                                    B0 = stacked_chunk.shape[1] // S # where S is number of "crps resamples"

                                    xs_pred_samples = stacked_chunk.permute(1, 0, 2)           # (B_eff, T, C)
                                    xs_pred_samples = xs_pred_samples.view(S, B0, T, C)  # (S, B0, T, C)
                                    xs_pred_samples = xs_pred_samples.permute(0, 2, 1, 3)  # (S, T, B0, C)

                                    xs_gt_b = stacked_x.permute(1, 0, 2)            # (B_eff, T, C)
                                    xs_gt_b = xs_gt_b.view(S, B0, T, C)
                                    xs_gt_b = xs_gt_b[0]                        # take the first sample (truth identical)
                                    xs_gt_b = xs_gt_b.permute(1, 0, 2)          # (T, B0, C)
                                    # have (S, T, B, C) and (T, B, C)
                                    crps_per_item = self.crps_per_batch_element(xs_pred_samples, xs_gt_b, is_inference=True) # shape is (B) -- the sole reward per batch item gotten at the end of rollout
                                    crps_per_item = crps_per_item.unsqueeze(0).expand(horizon, -1).unsqueeze(-1).expand(-1, -1, S).reshape(horizon, B0*S) # (T, B) instead of (T, B_eff)
                                    dense_rewards.append(-crps_per_item)

                # print(k_matrix_predicted.mean(dim=-1))
                
                k_histories.append(k_matrix_predicted)
                xs_pred += chunk
                rollout_lengths.append(max_rl_steps) # later with early breaking append with actual length

        # At this point xs_pred is list length n_frames; stack etc.
        xs_pred = torch.stack(xs_pred)      # (T, B, fs*C, ...)
        xs_gt = xs[:xs_pred.shape[0]]
        masks = masks[:xs_pred.shape[0]]
        # Then, stack the resulting tensors
        k_histories = torch.stack(k_histories)
        if self.step_reward:
            dense_rewards = torch.stack(dense_rewards)
            if self.difference_step_reward:
                first = torch.zeros_like(dense_rewards[0:1])   # shape (1,n_dec,B)
                rest  = dense_rewards[1:] - dense_rewards[:-1] # shape (rest,n_dec,B)
                dense_diff = torch.cat([first, rest], dim=0)
                dense_rewards = dense_diff
            dense_reward = dense_rewards.reshape(-1, xs_gt.shape[1])

        # Unlike original self.validation_step we have not mapped (T, B, fs*C, ...) --> (T * fs , B , C) yet.
        with torch.no_grad():
            if self.calc_crps_sum:
                # xs_pred: (T, B_eff, C)
                # reshape to (S, T, B, C)
                T, _, C = xs_pred.shape
                S = self.calc_crps_sum
                B0 = xs_pred.shape[1] // S # where S is number of "crps resamples"

                xs_pred_samples = xs_pred.permute(1, 0, 2)           # (B_eff, T, C)
                xs_pred_samples = xs_pred_samples.view(S, B0, T, C)  # (S, B0, T, C)
                xs_pred_samples = xs_pred_samples.permute(0, 2, 1, 3)  # (S, T, B0, C)

                xs_gt_b = xs_gt.permute(1, 0, 2)            # (B_eff, T, C)
                xs_gt_b = xs_gt_b.view(S, B0, T, C)
                xs_gt_b = xs_gt_b[0]                        # take the first sample (truth identical)
                xs_gt_b = xs_gt_b.permute(1, 0, 2)          # (T, B0, C)

                crps_per_item = self.crps_per_batch_element(xs_pred_samples, xs_gt_b) # shape is (B) -- the sole reward per batch item gotten at the end of rollout
                crps_per_item = crps_per_item.repeat_interleave(S) # to make it shape (B_eff) so that we can use downstream in reward for all the decisions we made

                crps_batch = crps_quantile_sum(xs_pred_samples, xs_gt_b)

            loss_per_elem = F.mse_loss(xs_pred, xs_gt, reduction="none")
            loss = self.reweigh_loss(loss_per_elem, masks)
            rl_reweighed_loss = self.reweigh_loss_rl(loss_per_elem, masks)
            
        # Policy Gradient Steps
        if log_probs:
            # (N_decisions, B)
            log_probs = torch.stack(log_probs, dim=0)
            entropies = torch.stack(entropies, dim=0)

            # make reward negative because the raw metrics are meant to be minimized
            reward = - crps_per_item if self.raw_reward_crps else -rl_reweighed_loss
            reward = reward.unsqueeze(0).expand_as(log_probs) + dense_reward

            # reward: [B]
            baseline = reward.mean() # baseline: scalar
            advantage = (reward - baseline).detach()       # [B]

            # Expand advantage to match (N_decisions, B) as before it is just B.
            advantage_expanded = advantage

            # Vanilla REINFORCE objective. Minimize "policy_loss" to maximize reward
            # L = - E[ advantage * log π(a_t | s_t) ]
            policy_loss = -(advantage_expanded * log_probs).mean() # expanded to enable inplace mult

            # Entropy bonus (optional)
            entropy_loss = -self.cfg.schedule_matrix.entropy_beta * entropies.mean()

            total_loss = policy_loss + entropy_loss
            
        del xs, xs_gt, chunk, conditions, init_z, z
        del log_probs, entropies, values, rewards, rollout_lengths

        return {"xs_pred": self._unnormalize_x(rearrange(xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)),
                "loss": loss, "crps": crps_batch.item(), "total_loss": total_loss, "k_history": k_histories.squeeze(0)}

    
    @property
    def device(self):
        ''' TEMPORARY -- WHEN OVERHAUL PL NEED TO REMOVE'''
        try:
            return next(self.parameters()).device
        except StopIteration:
            # no parameters — fallback
            return torch.device("cpu")


    def reweigh_loss_rl(self, loss, weight=None):
        """
        RL-specific loss reweighting, which instead of summing over all rollouts, guarantees credit assignment by maintaining that losses are split over the batch size. 

        loss:   [T, B, fs*C, ...]
        weight: [(T*fs), B, ...] or None

        Returns:
            per-sample (per-rollout) loss: [B]
        """

        # 1. Split frame-stack dimension: [T, B, fs*C, ...] -> [T, B, fs, C, ...]
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)

        # 2. Apply temporal masks (nonterminal mask over frames)
        if weight is not None:
            # weight originally [(T*fs), B, ...]
            # we want [T, B, fs, 1, 1, ...] so it can broadcast over C, H, W, etc.
            expand_dim = len(loss.shape) - len(weight.shape) - 1  # how many trailing singleton dims to add

            if expand_dim < 0:
                raise ValueError(
                    f"reweigh_loss_rl:  loss.shape={loss.shape}, weight.shape={weight.shape}, "
                    "cannot expand weight to match loss."
                )

            # Build einops pattern correctly (no '...1' garbage)
            # Example:
            #   expand_dim = 0: "(t fs) b ... -> t b fs ..."
            #   expand_dim = 2: "(t fs) b ... -> t b fs ... 1 1"
            suffix = ""
            if expand_dim > 0:
                suffix = " " + " ".join(["1"] * expand_dim)

            pattern = "(t fs) b ... -> t b fs ..." + suffix

            weight = rearrange(weight, pattern, fs=self.frame_stack)
            loss = loss * weight  # masked loss

        # 3. Reduce over all non-batch dims → keep only [B]
        # Current shape: [T, B, fs, C, ...]
        # Move batch first, flatten rest
        loss = rearrange(loss, "t b fs c ... -> b (t fs c ...)")
        # Now shape: [B, N_flat]
        loss = loss.mean(dim=1)  # → [B]

        return loss
    
    @torch.no_grad()
    def crps_per_batch_element(self, pred, truth, is_inference = False):
        """
        pred:  (samples, time, batch, feature)
        truth: (time, batch, feature)

        returns: tensor of shape (batch,)
        """
        S, T, B, C = pred.shape
        crps_vals = pred.new_zeros(B)

        for b in range(B):
            # Select single-batch slice while preserving expected dimensions
            # pred_b: (samples, time, batch=1, feature)
            pred_b = pred[:, :, b, :].unsqueeze(2)
            # truth_b: (time, batch=1, feature)
            truth_b = truth[:, b : b + 1, :]

            # Call CRPS
            if not is_inference:
                crps_b = crps_quantile_sum(pred_b[:, self.context_frames :], truth_b[self.context_frames: ]) # crps_sum_val = crps_quantile_sum(all_preds[:, self.context_frames :], gt[self.context_frames :])
            else:
                crps_b = crps_quantile_sum(pred_b[:, :], truth_b[:])
            crps_vals[b] = crps_b

        return crps_vals
    
