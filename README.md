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

For any other dataset (ts_electricity, etc.) we just change name and dataset in this command. We can evaluate and test models with 

`python -m main +name=ts_exchange dataset=ts_exchange algorithm=df_prediction experiment=exp_prediction experiment.tasks=["validation", "test"]`

## Training RL Denoisers

There are numerous design choices in our environment that we can assign each one a separate hydra argument to and append it to `python -m main ... experiment.tasks=["training_schedule_matrix"]`, allowing us to ablate across many hyperparameters. 

### Reward-Based Configs

$\frac{1}{z}$

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


