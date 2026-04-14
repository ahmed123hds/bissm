from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from BiSSM.ct_bissm.eval import evaluate_checkpoint
    from BiSSM.ct_bissm.generation import collect_offline_dataset
    from BiSSM.ct_bissm.trainer import TrainConfig, train_model
except ModuleNotFoundError:
    from ct_bissm.eval import evaluate_checkpoint
    from ct_bissm.generation import collect_offline_dataset
    from ct_bissm.trainer import TrainConfig, train_model


def main() -> None:
    root = Path("BiSSM/artifacts/smoke_pendulum")
    dataset_root = root / "dataset"
    run_root = root / "run"
    collect_offline_dataset(
        dataset_root=dataset_root,
        env_id="Pendulum-v1",
        policy_name="auto",
        qualities=("medium", "expert"),
        episodes_per_regime=2,
        max_steps=120,
        base_dt=0.05,
        jitter=0.2,
        seed=7,
    )
    checkpoint = train_model(
        TrainConfig(
            dataset_root=str(dataset_root),
            output_dir=str(run_root),
            model_name="ct_bissm",
            batch_size=4,
            total_updates=30,
            warmup_steps=5,
            window_size=20,
            stride=10,
            eval_interval=10,
            d_model=64,
            depth=2,
            dropout=0.1,
            lambda_bis=0.1,
            device="cpu",
            seed=7,
        )
    )
    results = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        env_id="Pendulum-v1",
        split="test",
        episodes_per_regime=1,
        max_steps=120,
        jitter=0.2,
        output_path=run_root / "smoke_eval.json",
        device="cpu",
    )
    print(results)


if __name__ == "__main__":
    main()
