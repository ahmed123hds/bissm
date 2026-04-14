from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .envs import PhysicsRegime, apply_physics_regime, build_time_deltas, create_env, default_regimes
from .policies import build_policy
from .storage import append_episode, initialize_manifest, load_manifest, save_manifest
from .utils import ensure_dir


def rollout_episode(
    env_id: str,
    regime: PhysicsRegime,
    policy_name: str,
    quality: str,
    max_steps: int,
    base_dt: float,
    jitter: float,
    seed: int,
    checkpoint_path: str | None = None,
    policy_device: str = "cpu",
    policy_noise_scale: float | None = None,
) -> tuple[dict, dict]:
    env = create_env(env_id)
    apply_physics_regime(env, regime)
    policy = build_policy(
        env=env,
        policy_name=policy_name,
        quality=quality,
        checkpoint_path=checkpoint_path,
        device=policy_device,
        policy_noise_scale=policy_noise_scale,
    )
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=int(seed))
    observations = [np.asarray(observation, dtype=np.float32)]
    actions = []
    rewards = []
    dones = []
    policy.reset()
    deltas, timestamps = build_time_deltas(max_steps, base_dt, jitter, rng)

    for step in range(max_steps):
        action = policy.act(np.asarray(observation, dtype=np.float32))
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        rewards.append(float(reward))
        dones.append(float(done))
        observations.append(np.asarray(next_observation, dtype=np.float32))
        observation = next_observation
        if done:
            break

    env.close()
    horizon = len(actions)
    episode_data = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "delta_t": np.asarray(deltas[:horizon], dtype=np.float32),
        "timestamps": np.asarray(timestamps[: horizon + 1], dtype=np.float32),
    }
    metadata = {
        "split": regime.split,
        "task_name": env_id,
        "physics_id": regime.name,
        "physics_params": regime.params,
        "quality": quality,
        "jitter": float(jitter),
        "base_dt": float(base_dt),
        "num_steps": int(horizon),
        "episode_return": float(np.sum(episode_data["rewards"])),
        "policy_noise_scale": None if policy_noise_scale is None else float(policy_noise_scale),
    }
    return episode_data, metadata


def collect_offline_dataset(
    dataset_root: str | Path,
    env_id: str,
    policy_name: str = "auto",
    qualities: Sequence[str] = ("random", "medium", "expert"),
    episodes_per_regime: int = 25,
    max_steps: int = 200,
    base_dt: float = 0.05,
    jitter: float = 0.0,
    seed: int = 0,
    checkpoint_path: str | None = None,
    policy_device: str = "cpu",
    policy_noise_scale: float | None = None,
    regimes: Iterable[PhysicsRegime] | None = None,
) -> dict:
    dataset_root = ensure_dir(dataset_root)
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(dataset_root)
    else:
        bootstrap_env = create_env(env_id)
        manifest = initialize_manifest(dataset_root.name, bootstrap_env)
        bootstrap_env.close()

    regimes = list(regimes) if regimes is not None else default_regimes(env_id)
    episode_seed = int(seed)
    for regime in regimes:
        for quality in qualities:
            for _ in range(episodes_per_regime):
                episode_data, metadata = rollout_episode(
                    env_id=env_id,
                    regime=regime,
                    policy_name=policy_name,
                    quality=quality,
                    max_steps=max_steps,
                    base_dt=base_dt,
                    jitter=jitter,
                    seed=episode_seed,
                    checkpoint_path=checkpoint_path,
                    policy_device=policy_device,
                    policy_noise_scale=policy_noise_scale,
                )
                append_episode(dataset_root, manifest, episode_data=episode_data, metadata=metadata)
                episode_seed += 1
    save_manifest(dataset_root, manifest)
    return manifest
