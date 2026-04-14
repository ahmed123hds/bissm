# CT-BiSSM

This directory now contains a full first-pass implementation of the idea described in:

- `BiSSM/ct_bissm_impl_plan.tex`
- `BiSSM/ct_bissm_icml_draft.tex`

Implemented pieces:

- physics-randomized offline dataset generation with irregular timestamps
- custom dataset storage/manifest format
- context-window loading and bisimulation pair mining
- CT-SSM backbone with timestamp-conditioned decay
- fixed-step SSM and time-aware transformer ablation baselines
- SB3 SAC collector training for medium/expert policy checkpoints
- Gaussian action head and trajectory-level bisimulation loss
- CUDA/CPU training entrypoint
- TPU training entrypoint for `torch_xla`
- evaluation script and a local smoke test

Quick start:

```bash
source /home/filliones/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
python BiSSM/generate_ct_bissm_data.py --env-id Pendulum-v1 --output-dir BiSSM/artifacts/pendulum_dataset
python BiSSM/train_ct_bissm_cuda.py --dataset-root BiSSM/artifacts/pendulum_dataset --output-dir BiSSM/artifacts/pendulum_run --device cpu
python BiSSM/evaluate_ct_bissm.py --checkpoint BiSSM/artifacts/pendulum_run/best.pt --env-id Pendulum-v1 --device cpu
```

Smoke test:

```bash
source /home/filliones/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
python BiSSM/run_ct_bissm_smoke_test.py
```

Notes:

- The code supports MuJoCo-style physics scaling, but this environment currently does not have `mujoco` installed.
- The TPU launcher expects `torch_xla` to be installed in your `pytorch_env`.
- Dataset storage is local `.npz` + `manifest.json` so the full pipeline runs even without `minari`.

SAC collector workflow:

```bash
source /home/filliones/Downloads/Documents/Work/Research/CVPR/pytorch_env/bin/activate
python BiSSM/train_sac_collector.py --env-id HalfCheetah-v5 --output-dir BiSSM/collectors/halfcheetah_sac --total-timesteps 1000000
python BiSSM/generate_ct_bissm_data.py --env-id HalfCheetah-v5 --output-dir BiSSM/data/halfcheetah_expert --policy-name sb3 --checkpoint-path BiSSM/collectors/halfcheetah_sac/expert_model.zip --qualities expert --policy-noise-scale 0.0 --episodes-per-regime 50 --max-steps 1000 --jitter 0.2
python BiSSM/generate_ct_bissm_data.py --env-id HalfCheetah-v5 --output-dir BiSSM/data/halfcheetah_medium --policy-name sb3 --checkpoint-path BiSSM/collectors/halfcheetah_sac/medium_model.zip --qualities medium --policy-noise-scale 0.0 --episodes-per-regime 50 --max-steps 1000 --jitter 0.2
```
