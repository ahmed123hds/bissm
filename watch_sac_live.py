from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from BiSSM.ct_bissm.envs import apply_physics_regime, default_regimes
except ModuleNotFoundError:
    from ct_bissm.envs import apply_physics_regime, default_regimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch an SB3 SAC checkpoint live in a MuJoCo viewer.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--physics-split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--physics-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import gymnasium as gym
    from stable_baselines3 import SAC

    model = SAC.load(args.checkpoint_path)
    regime = None
    if args.physics_split is not None:
        regimes = [item for item in default_regimes(args.env_id) if item.split == args.physics_split]
        if regimes:
            regime = regimes[max(0, min(args.physics_index, len(regimes) - 1))]

    for episode_idx in range(args.episodes):
        env = gym.make(args.env_id, render_mode="human")
        if regime is not None:
            apply_physics_regime(env, regime)
        observation, _ = env.reset(seed=args.seed + episode_idx)
        episode_return = 0.0
        for step in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            env.render()
            if args.sleep > 0.0:
                time.sleep(args.sleep)
            if terminated or truncated:
                break
        print(
            f"episode={episode_idx + 1}/{args.episodes} "
            f"return={episode_return:.2f} "
            f"steps={step + 1}",
            flush=True,
        )
        env.close()


if __name__ == "__main__":
    main()
