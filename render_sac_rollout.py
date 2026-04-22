from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from BiSSM.ct_bissm.envs import PhysicsRegime, apply_physics_regime, default_regimes
except ModuleNotFoundError:
    from ct_bissm.envs import PhysicsRegime, apply_physics_regime, default_regimes


def _load_sb3_model(checkpoint_path: str):
    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable_baselines3 is required to render SB3 collector checkpoints.") from exc
    return SAC.load(checkpoint_path)


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    margin = 12
    line_height = 18
    box_height = margin * 2 + line_height * len(lines)
    draw.rounded_rectangle(
        [(8, 8), (image.width - 8, 8 + box_height)],
        radius=10,
        fill=(0, 0, 0, 170),
    )
    y = 8 + margin
    for line in lines:
        draw.text((16, y), line, fill=(255, 255, 255))
        y += line_height
    return np.asarray(image)


def _select_regime(env_id: str, split: str, index: int) -> PhysicsRegime | None:
    regimes = [regime for regime in default_regimes(env_id) if regime.split == split]
    if not regimes:
        return None
    index = max(0, min(index, len(regimes) - 1))
    return regimes[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an SB3 SAC policy to a GIF for CT-BiSSM experiments.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--output-gif", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--camera-name", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--physics-split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--physics-index", type=int, default=0)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--mujoco-gl", type=str, default="auto", choices=["auto", "egl", "osmesa", "glfw"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mujoco_gl == "auto":
        if not os.environ.get("DISPLAY"):
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    else:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
        if args.mujoco_gl in {"egl", "osmesa"}:
            os.environ["PYOPENGL_PLATFORM"] = args.mujoco_gl

    import gymnasium as gym

    model = _load_sb3_model(args.checkpoint_path)
    render_kwargs = {
        "render_mode": "rgb_array",
        "width": args.width,
        "height": args.height,
    }
    if args.camera_name is not None:
        render_kwargs["camera_name"] = args.camera_name

    output_path = Path(args.output_gif)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames: list[np.ndarray] = []
    episode_returns: list[float] = []
    regime = None
    if args.physics_split is not None:
        regime = _select_regime(args.env_id, args.physics_split, args.physics_index)

    for episode_idx in range(args.episodes):
        env = gym.make(args.env_id, **render_kwargs)
        if regime is not None:
            apply_physics_regime(env, regime)
        observation, _ = env.reset(seed=args.seed + episode_idx)
        episode_return = 0.0
        for step in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if step % max(1, args.frame_skip) == 0:
                frame = env.render()
                if frame is not None:
                    frame = np.asarray(frame, dtype=np.uint8)
                    if not args.no_overlay:
                        lines = [
                            f"{args.env_id} | episode {episode_idx + 1}/{args.episodes}",
                            f"step {step + 1} | return {episode_return:.1f}",
                            f"checkpoint: {Path(args.checkpoint_path).name}",
                        ]
                        if regime is not None:
                            lines.append(f"physics: {regime.name} ({regime.split})")
                        frame = _overlay_text(frame, lines)
                    all_frames.append(frame)
            if terminated or truncated:
                break
        env.close()
        episode_returns.append(episode_return)

    if not all_frames:
        raise RuntimeError("No frames were captured. Check render settings or env compatibility.")

    duration = (1.0 / max(1, args.fps)) / max(args.speedup, 1e-6)
    imageio.mimsave(output_path, all_frames, duration=duration, loop=0)
    print(
        {
            "output_gif": str(output_path.resolve()),
            "episodes": args.episodes,
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "frames": len(all_frames),
            "fps": args.fps,
            "frame_skip": args.frame_skip,
            "physics_regime": None if regime is None else {"name": regime.name, "split": regime.split},
        }
    )


if __name__ == "__main__":
    main()
