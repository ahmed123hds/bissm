from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from BiSSM.ct_bissm.sac_training import SACTrainConfig, train_sac_behavior_policy
except ModuleNotFoundError:
    from ct_bissm.sac_training import SACTrainConfig, train_sac_behavior_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a strong SB3 SAC collector for CT-BiSSM datasets.")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v5")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--medium-timestep", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--net-arch", nargs="+", type=int, default=[256, 256])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SACTrainConfig(
        env_id=args.env_id,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        medium_timestep=args.medium_timestep,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        net_arch=tuple(args.net_arch),
    )
    manifest = train_sac_behavior_policy(config)
    print(manifest)


if __name__ == "__main__":
    main()
