from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BiSSM.ct_bissm.eval import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CT-BiSSM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--episodes-per-regime", type=int, default=5)
    parser.add_argument("--target-return", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "tpu"])
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        output_path=args.output_path,
        split=args.split,
        episodes_per_regime=args.episodes_per_regime,
        target_return=args.target_return,
        max_steps=args.max_steps,
        jitter=args.jitter,
        seed=args.seed,
        deterministic=not args.stochastic,
        device=args.device,
    )
    print(results)


if __name__ == "__main__":
    main()
