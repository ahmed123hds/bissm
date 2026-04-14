from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import shutil

from .utils import ensure_dir, save_json


@dataclass
class SACTrainConfig:
    env_id: str
    output_dir: str
    total_timesteps: int = 1_000_000
    medium_timestep: int | None = None
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 50_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    seed: int = 0
    device: str = "auto"
    net_arch: tuple[int, ...] = field(default_factory=lambda: (256, 256))
    verbose: int = 1

    def resolved_medium_timestep(self) -> int:
        if self.medium_timestep is not None:
            return int(self.medium_timestep)
        return max(1, int(0.3 * self.total_timesteps))


def _require_sb3() -> Any:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "stable_baselines3 is required for SAC collector training. "
            "Install it in your environment before running train_sac_collector.py."
        ) from exc
    return {
        "SAC": SAC,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
    }


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        ensure_dir(target.parent)
        shutil.copy2(source, target)


def train_sac_behavior_policy(config: SACTrainConfig) -> dict[str, Any]:
    sb3 = _require_sb3()
    SAC = sb3["SAC"]
    BaseCallback = sb3["BaseCallback"]
    CallbackList = sb3["CallbackList"]
    CheckpointCallback = sb3["CheckpointCallback"]
    EvalCallback = sb3["EvalCallback"]
    Monitor = sb3["Monitor"]
    DummyVecEnv = sb3["DummyVecEnv"]
    import gymnasium as gym

    output_dir = ensure_dir(config.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    best_dir = ensure_dir(output_dir / "best_model")
    tb_dir = ensure_dir(output_dir / "tensorboard")

    def make_env(seed: int):
        def _factory():
            env = gym.make(config.env_id)
            env = Monitor(env)
            env.reset(seed=seed)
            return env

        return _factory

    train_env = DummyVecEnv([make_env(config.seed)])
    eval_env = DummyVecEnv([make_env(config.seed + 10_000)])

    class MilestoneCallback(BaseCallback):
        def __init__(self, medium_timestep: int, save_dir: Path, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self.medium_timestep = int(medium_timestep)
            self.save_dir = save_dir
            self.saved = False

        def _on_step(self) -> bool:
            if (not self.saved) and self.num_timesteps >= self.medium_timestep:
                medium_path = self.save_dir / "medium_model"
                self.model.save(str(medium_path))
                self.saved = True
                if self.verbose:
                    print(f"Saved medium checkpoint at step {self.num_timesteps} to {medium_path}.zip", flush=True)
            return True

    callback = CallbackList(
        [
            MilestoneCallback(config.resolved_medium_timestep(), checkpoints_dir, verbose=config.verbose),
            CheckpointCallback(
                save_freq=max(1, int(config.checkpoint_freq)),
                save_path=str(checkpoints_dir),
                name_prefix="sac_checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(output_dir / "eval_logs"),
                eval_freq=max(1, int(config.eval_freq)),
                n_eval_episodes=int(config.n_eval_episodes),
                deterministic=True,
                render=False,
            ),
        ]
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        ent_coef=config.ent_coef,
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
        tensorboard_log=str(tb_dir),
        policy_kwargs={"net_arch": list(config.net_arch)},
    )

    model.learn(total_timesteps=int(config.total_timesteps), callback=callback, progress_bar=False)

    final_model_path = output_dir / "final_model"
    expert_model_path = output_dir / "expert_model"
    medium_model_path = checkpoints_dir / "medium_model"
    best_model_path = best_dir / "best_model"
    model.save(str(final_model_path))
    model.save(str(expert_model_path))

    _copy_if_exists(Path(f"{medium_model_path}.zip"), output_dir / "medium_model.zip")
    _copy_if_exists(Path(f"{best_model_path}.zip"), output_dir / "best_model.zip")
    _copy_if_exists(Path(f"{expert_model_path}.zip"), checkpoints_dir / "expert_model.zip")

    train_env.close()
    eval_env.close()

    manifest = {
        "config": asdict(config),
        "paths": {
            "medium": str((output_dir / "medium_model.zip").resolve()),
            "expert": str((output_dir / "expert_model.zip").resolve()),
            "best_eval": str((output_dir / "best_model.zip").resolve()),
            "final": str((output_dir / "final_model.zip").resolve()),
        },
    }
    save_json(manifest, output_dir / "sac_training_manifest.json")
    return manifest

