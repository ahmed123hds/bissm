from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BiSSM.ct_bissm.generation import collect_offline_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline datasets for CT-BiSSM.")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--policy-name", type=str, default="auto")
    parser.add_argument("--qualities", nargs="+", default=["random", "medium", "expert"])
    parser.add_argument("--episodes-per-regime", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--base-dt", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--policy-device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = collect_offline_dataset(
        dataset_root=args.output_dir,
        env_id=args.env_id,
        policy_name=args.policy_name,
        qualities=args.qualities,
        episodes_per_regime=args.episodes_per_regime,
        max_steps=args.max_steps,
        base_dt=args.base_dt,
        jitter=args.jitter,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        policy_device=args.policy_device,
    )
    print(f"Wrote {len(manifest['episodes'])} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
