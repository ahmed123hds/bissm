from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class PhysicsRegime:
    name: str
    split: str
    params: Dict[str, float]


def default_regimes(env_id: str) -> List[PhysicsRegime]:
    if env_id == "Pendulum-v1":
        return [
            PhysicsRegime("pend_train_0", "train", {"mass": 0.85, "length": 0.95, "gravity": 9.81}),
            PhysicsRegime("pend_train_1", "train", {"mass": 0.95, "length": 1.05, "gravity": 9.20}),
            PhysicsRegime("pend_train_2", "train", {"mass": 1.00, "length": 1.00, "gravity": 9.81}),
            PhysicsRegime("pend_train_3", "train", {"mass": 1.10, "length": 0.90, "gravity": 10.10}),
            PhysicsRegime("pend_train_4", "train", {"mass": 1.15, "length": 1.10, "gravity": 9.50}),
            PhysicsRegime("pend_train_5", "train", {"mass": 0.90, "length": 1.15, "gravity": 10.30}),
            PhysicsRegime("pend_val_0", "val", {"mass": 1.20, "length": 1.00, "gravity": 9.20}),
            PhysicsRegime("pend_val_1", "val", {"mass": 0.80, "length": 0.85, "gravity": 9.60}),
            PhysicsRegime("pend_test_0", "test", {"mass": 1.30, "length": 1.20, "gravity": 10.50}),
            PhysicsRegime("pend_test_1", "test", {"mass": 0.75, "length": 1.25, "gravity": 8.90}),
        ]
    if env_id == "MountainCarContinuous-v0":
        return [
            PhysicsRegime("mcc_train_0", "train", {"power": 0.0011, "gravity": 0.0023}),
            PhysicsRegime("mcc_train_1", "train", {"power": 0.0013, "gravity": 0.0025}),
            PhysicsRegime("mcc_train_2", "train", {"power": 0.0015, "gravity": 0.0025}),
            PhysicsRegime("mcc_train_3", "train", {"power": 0.0017, "gravity": 0.0027}),
            PhysicsRegime("mcc_train_4", "train", {"power": 0.0014, "gravity": 0.0022}),
            PhysicsRegime("mcc_train_5", "train", {"power": 0.0016, "gravity": 0.0024}),
            PhysicsRegime("mcc_val_0", "val", {"power": 0.0012, "gravity": 0.0028}),
            PhysicsRegime("mcc_val_1", "val", {"power": 0.0018, "gravity": 0.0021}),
            PhysicsRegime("mcc_test_0", "test", {"power": 0.0010, "gravity": 0.0030}),
            PhysicsRegime("mcc_test_1", "test", {"power": 0.0019, "gravity": 0.0020}),
        ]
    return [
        PhysicsRegime("train_0", "train", {"body_mass_scale": 0.9, "dof_damping_scale": 0.9, "geom_friction_scale": 0.8}),
        PhysicsRegime("train_1", "train", {"body_mass_scale": 1.0, "dof_damping_scale": 1.0, "geom_friction_scale": 1.0}),
        PhysicsRegime("train_2", "train", {"body_mass_scale": 1.1, "dof_damping_scale": 0.95, "geom_friction_scale": 1.2}),
        PhysicsRegime("train_3", "train", {"body_mass_scale": 0.85, "dof_damping_scale": 1.1, "geom_friction_scale": 1.0}),
        PhysicsRegime("train_4", "train", {"body_mass_scale": 1.15, "dof_damping_scale": 1.15, "geom_friction_scale": 0.9}),
        PhysicsRegime("train_5", "train", {"body_mass_scale": 0.95, "dof_damping_scale": 0.85, "geom_friction_scale": 1.3}),
        PhysicsRegime("val_0", "val", {"body_mass_scale": 1.2, "dof_damping_scale": 1.0, "geom_friction_scale": 1.1}),
        PhysicsRegime("val_1", "val", {"body_mass_scale": 0.8, "dof_damping_scale": 1.2, "geom_friction_scale": 0.85}),
        PhysicsRegime("test_0", "test", {"body_mass_scale": 1.3, "dof_damping_scale": 1.1, "geom_friction_scale": 1.25}),
        PhysicsRegime("test_1", "test", {"body_mass_scale": 0.75, "dof_damping_scale": 0.8, "geom_friction_scale": 1.4}),
    ]


def create_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    return gym.make(env_id, render_mode=render_mode)


def _apply_mujoco_scales(env: gym.Env, params: Dict[str, float]) -> None:
    model = getattr(env.unwrapped, "model", None)
    if model is None:
        return
    if "body_mass_scale" in params:
        model.body_mass[:] = model.body_mass[:] * params["body_mass_scale"]
    if "dof_damping_scale" in params:
        model.dof_damping[:] = model.dof_damping[:] * params["dof_damping_scale"]
    if "geom_friction_scale" in params:
        model.geom_friction[:] = model.geom_friction[:] * params["geom_friction_scale"]
    if "actuator_gear_scale" in params and hasattr(model, "actuator_gear"):
        model.actuator_gear[:] = model.actuator_gear[:] * params["actuator_gear_scale"]


def apply_physics_regime(env: gym.Env, regime: PhysicsRegime) -> None:
    unwrapped = env.unwrapped
    env_id = env.spec.id if env.spec is not None else type(unwrapped).__name__
    if env_id == "Pendulum-v1":
        if "mass" in regime.params:
            unwrapped.m = float(regime.params["mass"])
        if "length" in regime.params:
            unwrapped.l = float(regime.params["length"])
        if "gravity" in regime.params:
            unwrapped.g = float(regime.params["gravity"])
        return
    if env_id == "MountainCarContinuous-v0":
        if "power" in regime.params:
            unwrapped.power = float(regime.params["power"])
        if "gravity" in regime.params:
            unwrapped.gravity = float(regime.params["gravity"])
        return
    if env_id == "CartPole-v1":
        if "masscart" in regime.params:
            unwrapped.masscart = float(regime.params["masscart"])
        if "masspole" in regime.params:
            unwrapped.masspole = float(regime.params["masspole"])
        if "length" in regime.params:
            unwrapped.length = float(regime.params["length"])
        if "force_mag" in regime.params:
            unwrapped.force_mag = float(regime.params["force_mag"])
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length
        return
    _apply_mujoco_scales(env, regime.params)


def build_time_deltas(
    horizon: int,
    base_dt: float,
    jitter: float,
    rng: np.random.Generator,
    drop_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.full(horizon, base_dt, dtype=np.float32)
    if jitter > 0.0:
        deltas = deltas * (1.0 + rng.uniform(-jitter, jitter, size=horizon).astype(np.float32))
    deltas = np.clip(deltas, max(1e-4, base_dt * 0.05), None)
    if drop_probability > 0.0:
        drops = rng.random(horizon) < drop_probability
        extra = rng.integers(1, 4, size=horizon).astype(np.float32)
        deltas = deltas + drops.astype(np.float32) * extra * base_dt
    timestamps = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(deltas, dtype=np.float32)],
        axis=0,
    )
    return deltas.astype(np.float32), timestamps.astype(np.float32)

