from __future__ import annotations

from pathlib import Path
from typing import Optional
import importlib
import warnings

import gymnasium as gym
import numpy as np
import torch


def _safe_torch_load(checkpoint_path: str | Path, map_location: str):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


class BehaviorPolicy:
    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(BehaviorPolicy):
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = self.action_space.sample()
        return np.asarray(action, dtype=np.float32).reshape(-1)


class NoisyPolicy(BehaviorPolicy):
    def __init__(self, base: BehaviorPolicy, action_space: gym.Space, noise_scale: float) -> None:
        self.base = base
        self.action_space = action_space
        self.noise_scale = noise_scale

    def reset(self) -> None:
        self.base.reset()

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = self.base.act(observation)
        if self.noise_scale <= 0.0:
            return action.astype(np.float32)
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        scale = (high - low) / 2.0
        noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=action.shape).astype(np.float32)
        return np.clip(action + noise * scale, low, high).astype(np.float32)


class PendulumHeuristicPolicy(BehaviorPolicy):
    def __init__(self, torque_limit: float = 2.0, kp: float = 2.5, kd: float = 0.75) -> None:
        self.torque_limit = torque_limit
        self.kp = kp
        self.kd = kd

    def act(self, observation: np.ndarray) -> np.ndarray:
        cos_theta, sin_theta, theta_dot = observation
        theta = np.arctan2(sin_theta, cos_theta)
        torque = -(self.kp * theta + self.kd * theta_dot)
        return np.asarray([np.clip(torque, -self.torque_limit, self.torque_limit)], dtype=np.float32)


class MountainCarContinuousHeuristicPolicy(BehaviorPolicy):
    def act(self, observation: np.ndarray) -> np.ndarray:
        position, velocity = observation
        action = 1.0 if velocity >= 0.0 else -1.0
        action += 0.3 * np.sign(position + 0.25)
        return np.asarray([np.clip(action, -1.0, 1.0)], dtype=np.float32)


class TorchCheckpointPolicy(BehaviorPolicy):
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        module = _safe_torch_load(checkpoint_path, map_location=device)
        if isinstance(module, torch.jit.ScriptModule):
            self.model = module
        elif isinstance(module, torch.nn.Module):
            self.model = module
        else:
            raise TypeError("Checkpoint policy must be a torch.nn.Module or TorchScript module.")
        self.model.to(device)
        self.model.eval()
        self.device = device

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.model(obs)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


class SB3PolicyAdapter(BehaviorPolicy):
    def __init__(self, checkpoint_path: str | Path) -> None:
        stable_baselines3 = importlib.import_module("stable_baselines3")
        algo_names = ["SAC", "PPO", "TD3", "DDPG"]
        loaded = None
        errors = []
        for name in algo_names:
            algo_cls = getattr(stable_baselines3, name, None)
            if algo_cls is None:
                continue
            try:
                loaded = algo_cls.load(str(checkpoint_path))
                break
            except Exception as exc:  # pragma: no cover - optional dependency path
                errors.append(f"{name}: {exc}")
        if loaded is None:
            raise RuntimeError(f"Could not load SB3 policy from {checkpoint_path}. Errors: {errors}")
        self.model = loaded

    def act(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)


def build_policy(
    env: gym.Env,
    policy_name: str,
    quality: str = "medium",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    policy_noise_scale: Optional[float] = None,
) -> BehaviorPolicy:
    policy_name = policy_name.lower()
    quality = quality.lower()
    if quality == "random" or policy_name == "random":
        return RandomPolicy(env.action_space)

    if policy_name == "auto":
        if env.spec and env.spec.id == "Pendulum-v1":
            base = PendulumHeuristicPolicy(torque_limit=float(env.action_space.high[0]))
        elif env.spec and env.spec.id == "MountainCarContinuous-v0":
            base = MountainCarContinuousHeuristicPolicy()
        else:
            warnings.warn(
                "policy_name='auto' falls back to a random policy for this environment. "
                "For MuJoCo medium/expert datasets, pass --policy-name sb3 or --policy-name torch "
                "with a checkpoint path.",
                stacklevel=2,
            )
            base = RandomPolicy(env.action_space)
    elif policy_name == "pendulum_heuristic":
        base = PendulumHeuristicPolicy(torque_limit=float(env.action_space.high[0]))
    elif policy_name == "mountaincar_heuristic":
        base = MountainCarContinuousHeuristicPolicy()
    elif policy_name == "torch":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for policy_name='torch'")
        base = TorchCheckpointPolicy(checkpoint_path=checkpoint_path, device=device)
    elif policy_name == "sb3":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for policy_name='sb3'")
        base = SB3PolicyAdapter(checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown policy_name '{policy_name}'.")

    default_noise_by_quality = {
        "expert": 0.0,
        "medium": 0.0,
        "random": 1.00,
    }
    noise_scale = default_noise_by_quality.get(quality, 0.0) if policy_noise_scale is None else policy_noise_scale
    if noise_scale <= 0.0:
        return base
    return NoisyPolicy(base=base, action_space=env.action_space, noise_scale=noise_scale)
