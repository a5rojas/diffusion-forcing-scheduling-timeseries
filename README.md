# RL for Controllable Uncertainty in Diffusion Forcing

### Branched from Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion (https://arxiv.org/abs/2407.01392)


# Project Instructions

## Setup

Create conda environment:

```
conda create python=3.10 -n diffusion_forcing
conda activate diffusion_forcing
```

Install dependencies for time series:

```
pip install -r requirements.txt
```


## Train your own base model

### Timeseries Prediction

Train model with command:
`python -m main +name=ts_exchange dataset=ts_exchange algorithm=df_prediction experiment=exp_prediction`

For any other dataset (ts_electricity, etc.) we just change name and dataset in this command. We can evaluate and test models with the pyramidal denoising schedule using the command (for example with 20 diffusion sampling steps):  

`python -m main +name=ts_exchange dataset=ts_exchange algorithm=df_prediction experiment=exp_prediction experiment.tasks=["validation", "test"] algorithm.diffusion.sampling_steps=20`

## Training RL Denoisers

There are numerous design choices in our environment that we can assign each one a separate hydra argument to and append it to `python -m main ... experiment.tasks=["training_schedule_matrix"]`, allowing us to ablate across many hyperparameters. 

### Reward-Based Configs

The choice of the reward function was a complex design space. While we first considered CRPS-based rewards, the data overhead (because it is an ensemble method) was not justified by improved performance, so we removed those flags. More recently, we considered MSE-based rewards (at the end of the K matrix and a denser per-row version), denoising-encouraging rewards, and entropy rewards. By default, we always have a (negative) MSE reward comparing the final synthesized vector (after denoising completes) to the ground truth. The variable portions are:

- Whether to include a step-wise (denser) reward after each row of the matrix, where we compare the noisy iterate to ground truth with MSE (` algorithm.schedule_matrix.step_reward=True/False`) and whether to difference the result of that (ie to see how MSE after 3 rows of K compares to MSE after 4 rows of K instead of the raw values, using ` algorithm.schedule_matrix.difference_step_reward=True/False`)
- Entropy of the policy distribution (with weight `algorithm.training_schedule_matrix.entropy_beta=0.01`
- Whether to use a reward that encourages taking positive actions, in order to encourage completing the denosing process (both quickly and entirely) because incomplete denoising will likely be detrimental (with `algorihtm.training_schedule_matrix.denoise_reward=True/False` `algorihtm.training_schedule_matrix.denoise_bonus=0.1`).

### Action-Based Configs

Typical pyramidal Diffuson Forcing inference can be interpreted as taking (deterministic) noise-delta actions $a \in  \mathcal{A}$ for $\mathcal{A} =$ $\{0,1\}$ -- where tokens not getting yet denoised are given $a=0$, while tokens in the midst of the pyramid step down one level with $a=1$. In our approach, we call this setting

- `algorithm.training_schedule_matrix.positive_only=True algorithm.training_schedule_matrix.actions=2`

But, we are interested in expanding the action space to allow (i) renoising for self correction and (ii) skipping noise levels for faster inference. We represent this as, for example $\mathcal{A} = \{-2, -1, 0,1, 2\}$

- `algorithm.training_schedule_matrix.positive_only=False algorithm.training_schedule_matrix.actions=5`


# Infra instructions

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.

All experiments can be launched via `python -m main [options]` where you can fine more details in the following paragraphs.

## Pass in arguments

We use [hydra](https://hydra.cc) instead of `argparse` to configure arguments at every code level. You can both write a static config in `configuration` folder or, at runtime,
[override part of yur static config](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) with command line arguments.

For example, arguments `algorithm=df_prediction algorithm.diffusion.network_size=32` will override the `network_size` variable in `configurations/algorithm/df_prediction.yaml`.

All static config and runtime override will be logged to wandb automatically.

## Resume a checkpoint & logging

All checkpoints and logs are logged to cloud automatically so you can resume them on another server. Simply append `resume=[wandb_run_id]` to your command line arguments to resume it. The run_id can be founded in a url of a wandb run in wandb dashboard.

On the other hand, sometimes you may want to start a new run with different run id but still load a prior ckpt. This can be done by setting the `load=[wandb_run_id / ckpt path]` flag.


