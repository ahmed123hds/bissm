# CT-BiSSM

CT-BiSSM is an offline reinforcement learning project for studying control under physics shift and irregular timing. The repository contains:

- offline dataset generation with timestamp jitter and physics-regime metadata
- SAC-based collector training for medium and expert datasets
- CT-BiSSM, fixed-step SSM, and time-aware transformer policy models
- training and evaluation scripts for MuJoCo-style continuous-control tasks
- utilities for rendering and live inspection of saved SAC policies

The codebase is set up to support experiments such as:

- train a collector on a control task
- turn the collector into an offline dataset
- train CT-BiSSM or a baseline on that dataset
- evaluate the learned policy on held-out physics regimes

## Repository Layout

The main pieces are:

- `ct_bissm/`: model definitions, datasets, training utilities, SAC helpers, and physics regime logic
- `train_sac_collector.py`: trains an SB3 SAC collector
- `generate_ct_bissm_data.py`: generates offline datasets from a scripted or saved policy
- `train_ct_bissm_cuda.py`: trains CT-BiSSM or a baseline on GPU or CPU
- `train_ct_bissm_tpu.py`: TPU entrypoint for `torch_xla`
- `evaluate_ct_bissm.py`: rollout evaluation for trained checkpoints
- `render_sac_rollout.py`: saves a SAC rollout as a GIF
- `watch_sac_live.py`: opens a live viewer for a saved SAC policy
- `run_ct_bissm_smoke_test.py`: quick end-to-end smoke test

## Environment Setup

Create or activate a Python environment first, then install the core dependencies:

```bash
pip install -U pip
pip install stable-baselines3 "gymnasium[mujoco]" mujoco tensorboard imageio pillow
```

For local TPU runs you will also need `torch_xla`.

All commands below assume you are running from the repository root.

## Quick Smoke Test

This is the fastest way to make sure the main pipeline is wired correctly:

```bash
python run_ct_bissm_smoke_test.py
```

If you want a tiny manual run instead:

```bash
python generate_ct_bissm_data.py \
  --env-id Pendulum-v1 \
  --output-dir artifacts/pendulum_dataset

python train_ct_bissm_cuda.py \
  --dataset-root artifacts/pendulum_dataset \
  --output-dir artifacts/pendulum_run \
  --device cpu

python evaluate_ct_bissm.py \
  --checkpoint artifacts/pendulum_run/best.pt \
  --env-id Pendulum-v1 \
  --device cpu
```

## Main Workflow

### 1. Train a SAC collector

Nominal SAC training:

```bash
python train_sac_collector.py \
  --env-id HalfCheetah-v5 \
  --output-dir collectors/halfcheetah_sac \
  --total-timesteps 1000000 \
  --device auto
```

Physics-randomized SAC training:

```bash
python train_sac_collector.py \
  --env-id Hopper-v5 \
  --output-dir collectors/hopper_sac_dr \
  --total-timesteps 2000000 \
  --device auto \
  --physics-randomization train \
  --eval-physics-split val
```

The collector script saves periodic checkpoints, a medium checkpoint when available, and a best-eval checkpoint under `best_model/`.

### 2. Generate an offline dataset

Expert dataset from a saved SAC collector:

```bash
python generate_ct_bissm_data.py \
  --env-id HalfCheetah-v5 \
  --output-dir data/halfcheetah_expert \
  --policy-name sb3 \
  --checkpoint-path collectors/halfcheetah_sac/best_model/best_model \
  --qualities expert \
  --policy-noise-scale 0.0 \
  --episodes-per-regime 50 \
  --max-steps 1000 \
  --jitter 0.2 \
  --seed 200
```

Medium dataset:

```bash
python generate_ct_bissm_data.py \
  --env-id HalfCheetah-v5 \
  --output-dir data/halfcheetah_medium \
  --policy-name sb3 \
  --checkpoint-path collectors/halfcheetah_sac/checkpoints/medium_model \
  --qualities medium \
  --policy-noise-scale 0.0 \
  --episodes-per-regime 50 \
  --max-steps 1000 \
  --jitter 0.2 \
  --seed 100
```

### 3. Train CT-BiSSM or a baseline

CT-BiSSM:

```bash
python train_ct_bissm_cuda.py \
  --dataset-root data/halfcheetah_expert \
  --output-dir runs/hc_ct_bissm_expert \
  --model-name ct_bissm \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --warmup-steps 5000 \
  --total-updates 25000 \
  --grad-clip 1.0 \
  --window-size 50 \
  --stride 10 \
  --lambda-bis 0.05 \
  --gamma 1.0 \
  --d-model 256 \
  --depth 4 \
  --dropout 0.1 \
  --projection-mode last \
  --eval-interval 1000 \
  --device auto
```

Fixed-step SSM baseline:

```bash
python train_ct_bissm_cuda.py \
  --dataset-root data/halfcheetah_expert \
  --output-dir runs/hc_fixed_ssm_expert \
  --model-name fixed_ssm \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --warmup-steps 5000 \
  --total-updates 25000 \
  --grad-clip 1.0 \
  --window-size 50 \
  --stride 10 \
  --lambda-bis 0.0 \
  --gamma 1.0 \
  --d-model 256 \
  --depth 4 \
  --dropout 0.1 \
  --projection-mode last \
  --eval-interval 1000 \
  --device auto
```

Time-aware transformer baseline:

```bash
python train_ct_bissm_cuda.py \
  --dataset-root data/halfcheetah_expert \
  --output-dir runs/hc_time_transformer_expert \
  --model-name time_transformer \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --warmup-steps 5000 \
  --total-updates 25000 \
  --grad-clip 1.0 \
  --window-size 50 \
  --stride 10 \
  --lambda-bis 0.0 \
  --gamma 1.0 \
  --d-model 256 \
  --depth 4 \
  --dropout 0.1 \
  --projection-mode last \
  --eval-interval 1000 \
  --device auto
```

### 4. Evaluate a trained checkpoint

```bash
python evaluate_ct_bissm.py \
  --checkpoint runs/hc_ct_bissm_expert/best.pt \
  --env-id HalfCheetah-v5 \
  --split test \
  --episodes-per-regime 10 \
  --max-steps 1000 \
  --jitter 0.2 \
  --target-return 10000 \
  --device auto
```

## Utilities

Render a saved SAC policy to a GIF:

```bash
python render_sac_rollout.py \
  --env-id HalfCheetah-v5 \
  --checkpoint-path collectors/halfcheetah_sac/best_model/best_model \
  --output-gif collectors/halfcheetah_sac/halfcheetah_expert.gif \
  --episodes 1 \
  --max-steps 1000 \
  --fps 30 \
  --frame-skip 4
```

Watch a SAC policy live:

```bash
python watch_sac_live.py \
  --env-id HalfCheetah-v5 \
  --checkpoint-path collectors/halfcheetah_sac/best_model/best_model \
  --episodes 1 \
  --max-steps 1000
```

## Practical Notes

- `generate_ct_bissm_data.py` appends to an existing dataset directory if `manifest.json` is already present. Use a fresh output directory when you want a clean dataset.
- For SB3 checkpoints, passing `.../best_model/best_model` is usually safer than passing `...zip` explicitly because some loaders append `.zip` internally.
- Recent training changes align the offline training inputs with rollout-time information more carefully:
  - return-to-go is computed from the full episode, then sliced into windows
  - time deltas are aligned to elapsed time available at decision time
- MuJoCo tasks require a working `gymnasium[mujoco]` and `mujoco` installation.
- If you are running in Colab after cloning this repository into `/content/bissm`, use commands like `python train_ct_bissm_cuda.py ...` from that repo root rather than prefixing paths with `BiSSM/`.

## Current Scope

This repository is still experiment-oriented rather than fully polished as a package. The goal is to keep the main research loop easy to run, modify, and debug while the method and benchmarks are still evolving.
