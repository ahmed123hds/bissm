from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .dataset import DatasetStatistics
from .envs import apply_physics_regime, build_time_deltas, create_env, default_regimes
from .trainer import load_checkpoint_model
from .utils import save_json, set_seed


def _context_batch(
    states: deque,
    prev_actions: deque,
    prev_rewards: deque,
    returns_to_go: deque,
    delta_t: deque,
    dones: deque,
    stats: DatasetStatistics,
) -> dict:
    states_np = np.stack(states, axis=0).astype(np.float32)
    prev_actions_np = np.stack(prev_actions, axis=0).astype(np.float32)
    prev_rewards_np = np.asarray(prev_rewards, dtype=np.float32).reshape(-1, 1)
    rtg_np = np.asarray(returns_to_go, dtype=np.float32).reshape(-1, 1)
    delta_np = np.asarray(delta_t, dtype=np.float32).reshape(-1, 1)
    done_np = np.asarray(dones, dtype=np.float32).reshape(-1, 1)
    scaled_dt = delta_np / max(1e-6, stats.base_dt)
    return {
        "states": torch.from_numpy(((states_np - stats.state_mean) / stats.state_std)).unsqueeze(0),
        "prev_actions": torch.from_numpy(prev_actions_np).unsqueeze(0),
        "prev_rewards": torch.from_numpy(prev_rewards_np / stats.reward_scale).unsqueeze(0),
        "returns_to_go": torch.from_numpy(rtg_np / stats.return_scale).unsqueeze(0),
        "delta_t": torch.from_numpy(scaled_dt).unsqueeze(0),
        "log_delta_t": torch.from_numpy(np.log1p(scaled_dt)).unsqueeze(0),
        "dones": torch.from_numpy(done_np).unsqueeze(0),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    env_id: str,
    output_path: str | Path | None = None,
    split: str = "test",
    episodes_per_regime: int = 5,
    target_return: float = 0.0,
    max_steps: int = 200,
    jitter: float = 0.0,
    seed: int = 0,
    deterministic: bool = True,
    device: str = "auto",
) -> dict:
    set_seed(seed)
    model, checkpoint, device_ctx = load_checkpoint_model(checkpoint_path, device=device)
    stats = DatasetStatistics.from_dict(checkpoint["dataset_statistics"])
    context_length = int(checkpoint.get("window_size", 50))
    action_dim = int(checkpoint["action_dim"])
    regimes = [regime for regime in default_regimes(env_id) if regime.split == split]
    if not regimes:
        raise RuntimeError(f"No regimes found for split='{split}' and env_id='{env_id}'.")

    results = {"env_id": env_id, "split": split, "regimes": []}
    episode_seed = seed
    for regime in regimes:
        returns = []
        for _ in range(episodes_per_regime):
            env = create_env(env_id)
            apply_physics_regime(env, regime)
            observation, _ = env.reset(seed=episode_seed)
            deltas, _ = build_time_deltas(max_steps, stats.base_dt, jitter, np.random.default_rng(episode_seed))
            states = deque(maxlen=context_length)
            prev_actions = deque(maxlen=context_length)
            prev_rewards = deque(maxlen=context_length)
            returns_to_go = deque(maxlen=context_length)
            delta_hist = deque(maxlen=context_length)
            done_hist = deque(maxlen=context_length)
            episode_return = 0.0
            running_target = target_return
            states.append(np.asarray(observation, dtype=np.float32))
            prev_actions.append(np.zeros(action_dim, dtype=np.float32))
            prev_rewards.append(0.0)
            returns_to_go.append(running_target)
            delta_hist.append(stats.base_dt)
            done_hist.append(0.0)
            for step in range(max_steps):
                batch = _context_batch(
                    states=states,
                    prev_actions=prev_actions,
                    prev_rewards=prev_rewards,
                    returns_to_go=returns_to_go,
                    delta_t=delta_hist,
                    dones=done_hist,
                    stats=stats,
                )
                batch = device_ctx.move_batch(batch)
                with torch.no_grad():
                    action = model.predict_action(batch, deterministic=deterministic)
                action_np = action.squeeze(0).detach().cpu().numpy()
                next_observation, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated
                episode_return += float(reward)
                running_target -= float(reward)
                states.append(np.asarray(next_observation, dtype=np.float32))
                prev_actions.append(action_np.astype(np.float32))
                prev_rewards.append(float(reward))
                returns_to_go.append(float(running_target))
                delta_hist.append(float(deltas[min(step, len(deltas) - 1)]))
                done_hist.append(float(done))
                observation = next_observation
                if done:
                    break
            env.close()
            returns.append(episode_return)
            episode_seed += 1
        results["regimes"].append(
            {
                "physics_id": regime.name,
                "physics_params": regime.params,
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "returns": [float(value) for value in returns],
            }
        )

    if output_path is not None:
        save_json(results, output_path)
    return results

