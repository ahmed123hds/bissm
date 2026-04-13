from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from .utils import ensure_dir, load_json, save_json


def _serialize_space(space: gym.Space) -> Dict[str, Any]:
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "box",
            "shape": list(space.shape),
            "low": np.asarray(space.low, dtype=np.float32).tolist(),
            "high": np.asarray(space.high, dtype=np.float32).tolist(),
        }
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "discrete", "n": int(space.n)}
    raise TypeError(f"Unsupported space type {type(space)}")


def initialize_manifest(dataset_name: str, env: gym.Env) -> Dict[str, Any]:
    observation_shape = list(env.observation_space.shape)
    return {
        "dataset_name": dataset_name,
        "env_id": env.spec.id if env.spec is not None else type(env.unwrapped).__name__,
        "observation_shape": observation_shape,
        "action_space": _serialize_space(env.action_space),
        "episodes": [],
    }


def save_manifest(dataset_root: str | Path, manifest: Dict[str, Any]) -> None:
    save_json(manifest, Path(dataset_root) / "manifest.json")


def load_manifest(dataset_root: str | Path) -> Dict[str, Any]:
    return load_json(Path(dataset_root) / "manifest.json")


def load_episode(dataset_root: str | Path, episode_record: Dict[str, Any]) -> Dict[str, np.ndarray]:
    episode_path = Path(dataset_root) / episode_record["path"]
    episode = np.load(episode_path)
    return {key: episode[key] for key in episode.files}


def append_episode(
    dataset_root: str | Path,
    manifest: Dict[str, Any],
    episode_data: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    episodes_dir = ensure_dir(dataset_root / "episodes")
    episode_id = len(manifest["episodes"])
    relative_path = Path("episodes") / f"episode_{episode_id:06d}.npz"
    np.savez_compressed(episodes_dir / relative_path.name, **episode_data)
    record = {"episode_id": episode_id, "path": str(relative_path), **metadata}
    manifest["episodes"].append(record)
    return record

