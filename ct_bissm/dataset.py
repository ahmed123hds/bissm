from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .storage import load_episode, load_manifest
from .utils import discounted_cumsum, save_json, window_starts


@dataclass
class DatasetStatistics:
    state_mean: np.ndarray
    state_std: np.ndarray
    reward_scale: float
    return_scale: float
    base_dt: float

    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "reward_scale": float(self.reward_scale),
            "return_scale": float(self.return_scale),
            "base_dt": float(self.base_dt),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetStatistics":
        return cls(
            state_mean=np.asarray(data["state_mean"], dtype=np.float32),
            state_std=np.asarray(data["state_std"], dtype=np.float32),
            reward_scale=float(data["reward_scale"]),
            return_scale=float(data["return_scale"]),
            base_dt=float(data["base_dt"]),
        )


def _stats_path(dataset_root: Path) -> Path:
    return dataset_root / "stats_train.json"


def compute_dataset_statistics(dataset_root: str | Path, split: str = "train") -> DatasetStatistics:
    dataset_root = Path(dataset_root)
    manifest = load_manifest(dataset_root)
    state_sum = None
    state_sq_sum = None
    count = 0
    rewards = []
    returns = []
    base_dts = []
    for episode_record in manifest["episodes"]:
        if episode_record["split"] != split:
            continue
        episode = load_episode(dataset_root, episode_record)
        states = episode["observations"][:-1].astype(np.float32)
        if state_sum is None:
            state_sum = np.zeros(states.shape[-1], dtype=np.float64)
            state_sq_sum = np.zeros(states.shape[-1], dtype=np.float64)
        state_sum += states.sum(axis=0)
        state_sq_sum += np.square(states).sum(axis=0)
        count += states.shape[0]
        rewards.extend(episode["rewards"].astype(np.float32).tolist())
        returns.append(float(np.sum(episode["rewards"])))
        base_dts.append(float(episode_record.get("base_dt", 1.0)))

    if count == 0:
        raise RuntimeError(f"No episodes found for split='{split}' in {dataset_root}.")

    mean = (state_sum / count).astype(np.float32)
    var = np.maximum((state_sq_sum / count) - np.square(mean), 1e-6)
    std = np.sqrt(var).astype(np.float32)
    reward_scale = max(1.0, float(np.std(np.asarray(rewards, dtype=np.float32))))
    return_scale = max(1.0, float(np.percentile(np.abs(np.asarray(returns, dtype=np.float32)), 95)))
    base_dt = float(np.median(np.asarray(base_dts, dtype=np.float32)))
    stats = DatasetStatistics(
        state_mean=mean,
        state_std=std,
        reward_scale=reward_scale,
        return_scale=return_scale,
        base_dt=base_dt,
    )
    save_json(stats.to_dict(), _stats_path(dataset_root))
    return stats


def load_or_compute_dataset_statistics(dataset_root: str | Path) -> DatasetStatistics:
    dataset_root = Path(dataset_root)
    stats_path = _stats_path(dataset_root)
    if stats_path.exists():
        from .utils import load_json

        return DatasetStatistics.from_dict(load_json(stats_path))
    return compute_dataset_statistics(dataset_root)


class OfflineWindowPairDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        window_size: int = 50,
        stride: int = 10,
        gamma: float = 1.0,
        return_buckets: int = 8,
        seed: int = 0,
        cache_size: int = 8,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest = load_manifest(self.dataset_root)
        self.stats = load_or_compute_dataset_statistics(self.dataset_root)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.gamma = gamma
        self.return_buckets = return_buckets
        self.seed = seed
        self.cache_size = cache_size
        self._cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()
        action_space = self.manifest["action_space"]
        if action_space["type"] != "box":
            raise ValueError("CT-BiSSM trainer currently expects continuous Box action spaces.")
        self.action_low = np.asarray(action_space["low"], dtype=np.float32)
        self.action_high = np.asarray(action_space["high"], dtype=np.float32)
        self.action_dim = int(np.prod(action_space["shape"]))
        self.state_dim = int(np.prod(self.manifest["observation_shape"]))
        self.episodes = [record for record in self.manifest["episodes"] if record["split"] == split]
        self.window_index = self._build_window_index()
        if not self.window_index:
            raise RuntimeError(f"No windows available for split='{split}' in {dataset_root}.")
        self._build_pair_groups()

    def _get_episode(self, list_index: int) -> Dict[str, np.ndarray]:
        if list_index in self._cache:
            episode = self._cache.pop(list_index)
            self._cache[list_index] = episode
            return episode
        episode = load_episode(self.dataset_root, self.episodes[list_index])
        self._cache[list_index] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def _build_window_index(self) -> List[dict]:
        windows: List[dict] = []
        for episode_list_index, record in enumerate(self.episodes):
            episode = self._get_episode(episode_list_index)
            rewards = episode["rewards"]
            next_obs = episode["observations"][1:]
            num_steps = len(rewards)
            for start in window_starts(num_steps, self.window_size, self.stride):
                stop = start + self.window_size
                windows.append(
                    {
                        "episode_list_index": episode_list_index,
                        "start": start,
                        "stop": stop,
                        "task_name": record["task_name"],
                        "physics_id": record["physics_id"],
                        "return_sum": float(rewards[start:stop].sum()),
                        "mean_reward": float(rewards[start:stop].mean()),
                        "mean_next_feature": next_obs[start:stop].mean(axis=0).astype(np.float32),
                    }
                )
        returns = np.asarray([entry["return_sum"] for entry in windows], dtype=np.float32)
        if len(np.unique(returns)) == 1:
            bucket_edges = np.asarray([returns[0]], dtype=np.float32)
        else:
            bucket_edges = np.quantile(returns, np.linspace(0.0, 1.0, self.return_buckets + 1)[1:-1])
        for entry in windows:
            entry["return_bucket"] = int(np.searchsorted(bucket_edges, entry["return_sum"], side="right"))
        return windows

    def _build_pair_groups(self) -> None:
        self.task_bucket_groups = defaultdict(list)
        self.task_groups = defaultdict(list)
        for idx, entry in enumerate(self.window_index):
            self.task_bucket_groups[(entry["task_name"], entry["return_bucket"])].append(idx)
            self.task_groups[entry["task_name"]].append(idx)

    def _sample_partner_index(self, idx: int) -> int:
        anchor = self.window_index[idx]
        rng = np.random.default_rng(self.seed + idx)
        same_bucket = [
            cand
            for cand in self.task_bucket_groups[(anchor["task_name"], anchor["return_bucket"])]
            if cand != idx and self.window_index[cand]["physics_id"] != anchor["physics_id"]
        ]
        if same_bucket:
            return int(same_bucket[rng.integers(len(same_bucket))])
        cross_physics = [
            cand
            for cand in self.task_groups[anchor["task_name"]]
            if cand != idx and self.window_index[cand]["physics_id"] != anchor["physics_id"]
        ]
        if cross_physics:
            return int(cross_physics[rng.integers(len(cross_physics))])
        same_task = [cand for cand in self.task_groups[anchor["task_name"]] if cand != idx]
        if same_task:
            return int(same_task[rng.integers(len(same_task))])
        return idx

    def _window_to_tensors(self, entry: dict) -> dict:
        record = self.episodes[entry["episode_list_index"]]
        episode = self._get_episode(entry["episode_list_index"])
        start = entry["start"]
        stop = entry["stop"]
        observations = episode["observations"]
        actions = episode["actions"].astype(np.float32)
        rewards = episode["rewards"].astype(np.float32)
        dones = episode["dones"].astype(np.float32)
        delta_t = episode["delta_t"].astype(np.float32)
        states = observations[start:stop].astype(np.float32)
        next_states = observations[start + 1 : stop + 1].astype(np.float32)
        action_slice = actions[start:stop]
        reward_slice = rewards[start:stop]
        done_slice = dones[start:stop]
        delta_slice = np.empty_like(reward_slice, dtype=np.float32)
        delta_slice[0] = delta_t[start - 1] if start > 0 else float(record.get("base_dt", self.stats.base_dt))
        if len(delta_slice) > 1:
            delta_slice[1:] = delta_t[start : stop - 1]
        prev_actions = np.zeros_like(action_slice, dtype=np.float32)
        prev_rewards = np.zeros((len(reward_slice), 1), dtype=np.float32)
        if start > 0:
            prev_actions[0] = actions[start - 1]
            prev_rewards[0, 0] = rewards[start - 1]
        if len(action_slice) > 1:
            prev_actions[1:] = action_slice[:-1]
            prev_rewards[1:, 0] = reward_slice[:-1]
        episode_returns_to_go = discounted_cumsum(rewards, gamma=self.gamma)
        returns_to_go = episode_returns_to_go[start:stop].reshape(-1, 1)
        state_mean = self.stats.state_mean
        state_std = self.stats.state_std
        base_dt = max(1e-6, float(record.get("base_dt", self.stats.base_dt)))
        scaled_dt = (delta_slice / base_dt).reshape(-1, 1).astype(np.float32)
        log_delta_t = np.log1p(scaled_dt).astype(np.float32)
        tensors = {
            "states": torch.from_numpy(((states - state_mean) / state_std).astype(np.float32)),
            "actions": torch.from_numpy(action_slice.astype(np.float32)),
            "prev_actions": torch.from_numpy(prev_actions.astype(np.float32)),
            "prev_rewards": torch.from_numpy((prev_rewards / self.stats.reward_scale).astype(np.float32)),
            "returns_to_go": torch.from_numpy((returns_to_go / self.stats.return_scale).astype(np.float32)),
            "delta_t": torch.from_numpy(scaled_dt.astype(np.float32)),
            "log_delta_t": torch.from_numpy(log_delta_t.astype(np.float32)),
            "dones": torch.from_numpy(done_slice.reshape(-1, 1).astype(np.float32)),
            "mean_next_feature": torch.from_numpy(
                (((next_states.mean(axis=0) - state_mean) / state_std)).astype(np.float32)
            ),
            "mean_reward": torch.tensor(entry["mean_reward"] / self.stats.reward_scale, dtype=torch.float32),
        }
        return tensors

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> dict:
        partner_idx = self._sample_partner_index(idx)
        anchor_entry = self.window_index[idx]
        partner_entry = self.window_index[partner_idx]
        return {
            "anchor": self._window_to_tensors(anchor_entry),
            "partner": self._window_to_tensors(partner_entry),
        }
