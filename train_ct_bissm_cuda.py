from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BiSSM.ct_bissm.trainer import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CT-BiSSM on CPU/CUDA.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="ct_bissm", choices=["ct_bissm", "fixed_ssm", "time_transformer"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--total-updates", type=int, default=20000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--lambda-bis", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--projection-mode", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_updates=args.total_updates,
        grad_clip=args.grad_clip,
        window_size=args.window_size,
        stride=args.stride,
        lambda_bis=args.lambda_bis,
        gamma=args.gamma,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        projection_mode=args.projection_mode,
        eval_interval=args.eval_interval,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    checkpoint_path = train_model(config)
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
