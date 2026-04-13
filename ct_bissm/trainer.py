from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .dataset import OfflineWindowPairDataset
from .device import DeviceContext, get_device_context
from .losses import bisimulation_regression_loss, bisimulation_target, tanh_gaussian_nll
from .model import build_model
from .utils import ensure_dir, linear_warmup_multiplier, save_json, set_seed


def _safe_torch_load(checkpoint_path: str | Path) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str
    model_name: str = "ct_bissm"
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    total_updates: int = 20000
    grad_clip: float = 1.0
    window_size: int = 50
    stride: int = 10
    lambda_bis: float = 0.1
    gamma: float = 1.0
    d_model: int = 256
    depth: int = 4
    dropout: float = 0.1
    projection_mode: str = "last"
    eval_interval: int = 1000
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"


def _sequence_losses(model: torch.nn.Module, batch: Dict[str, torch.Tensor], projection_mode: str) -> Dict[str, torch.Tensor]:
    anchor_out = model(batch["anchor"])
    partner_out = model(batch["partner"])
    anchor_nll = tanh_gaussian_nll(
        actions=batch["anchor"]["actions"],
        mean=anchor_out["mean"],
        log_std=anchor_out["log_std"],
        action_low=model.action_low,
        action_high=model.action_high,
    ).mean()
    partner_nll = tanh_gaussian_nll(
        actions=batch["partner"]["actions"],
        mean=partner_out["mean"],
        log_std=partner_out["log_std"],
        action_low=model.action_low,
        action_high=model.action_high,
    ).mean()
    act_loss = 0.5 * (anchor_nll + partner_nll)
    anchor_proj = model.project_hidden(anchor_out["hidden"], mode=projection_mode)
    partner_proj = model.project_hidden(partner_out["hidden"], mode=projection_mode)
    target = bisimulation_target(
        mean_reward_a=batch["anchor"]["mean_reward"],
        mean_reward_b=batch["partner"]["mean_reward"],
        next_feature_a=batch["anchor"]["mean_next_feature"],
        next_feature_b=batch["partner"]["mean_next_feature"],
    )
    bis_loss = bisimulation_regression_loss(anchor_proj, partner_proj, target)
    return {"act_loss": act_loss, "bis_loss": bis_loss}


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device_ctx: DeviceContext,
    projection_mode: str,
    lambda_bis: float,
    max_batches: int = 50,
) -> Dict[str, float]:
    model.eval()
    totals = {"act_loss": 0.0, "bis_loss": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = device_ctx.move_batch(batch)
            losses = _sequence_losses(model, batch, projection_mode=projection_mode)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
            if batches >= max_batches:
                break
    if batches == 0:
        return {"act_loss": float("nan"), "bis_loss": float("nan"), "total_loss": float("nan")}
    act = totals["act_loss"] / batches
    bis = totals["bis_loss"] / batches
    return {"act_loss": act, "bis_loss": bis, "total_loss": act + lambda_bis * bis}


def _save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    train_dataset: OfflineWindowPairDataset,
    step: int,
    best_val: float,
) -> None:
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "model_name": config.model_name,
            "model_hparams": {
                "d_model": config.d_model,
                "depth": config.depth,
                "dropout": config.dropout,
            },
            "dataset_statistics": train_dataset.stats.to_dict(),
            "action_low": train_dataset.action_low.tolist(),
            "action_high": train_dataset.action_high.tolist(),
            "state_dim": train_dataset.state_dim,
            "action_dim": train_dataset.action_dim,
            "window_size": config.window_size,
            "config": config.__dict__,
        },
        checkpoint_path,
    )


def train_model(config: TrainConfig) -> Path:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    device_ctx = get_device_context(config.device)
    train_dataset = OfflineWindowPairDataset(
        dataset_root=config.dataset_root,
        split="train",
        window_size=config.window_size,
        stride=config.stride,
        gamma=config.gamma,
        seed=config.seed,
    )
    val_dataset = None
    try:
        val_dataset = OfflineWindowPairDataset(
            dataset_root=config.dataset_root,
            split="val",
            window_size=config.window_size,
            stride=config.stride,
            gamma=config.gamma,
            seed=config.seed + 17,
        )
    except RuntimeError:
        val_dataset = None

    model = build_model(
        model_name=config.model_name,
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        action_low=train_dataset.action_low,
        action_high=train_dataset.action_high,
        d_model=config.d_model,
        depth=config.depth,
        dropout=config.dropout,
    )
    model.to(device_ctx.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: linear_warmup_multiplier(step + 1, config.warmup_steps),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )

    best_val = float("inf")
    history = []
    checkpoint_path = output_dir / "best.pt"
    for step, batch in zip(range(1, config.total_updates + 1), cycle(train_loader)):
        model.train()
        batch = device_ctx.move_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        losses = _sequence_losses(model, batch, projection_mode=config.projection_mode)
        total_loss = losses["act_loss"] + config.lambda_bis * losses["bis_loss"]
        total_loss.backward()
        clip_grad_norm_(model.parameters(), config.grad_clip)
        device_ctx.optimizer_step(optimizer)
        scheduler.step()
        device_ctx.mark_step()

        if step % 100 == 0 or step == 1:
            message = (
                f"step={step:06d} "
                f"act={float(losses['act_loss'].detach().cpu()):.4f} "
                f"bis={float(losses['bis_loss'].detach().cpu()):.4f} "
                f"total={float(total_loss.detach().cpu()):.4f}"
            )
            print(message, flush=True)

        if val_loader is not None and (step % config.eval_interval == 0 or step == config.total_updates):
            metrics = _evaluate(
                model=model,
                loader=val_loader,
                device_ctx=device_ctx,
                projection_mode=config.projection_mode,
                lambda_bis=config.lambda_bis,
            )
            metrics["step"] = step
            metrics["train_total_loss"] = float(total_loss.detach().cpu())
            history.append(metrics)
            print(
                "val "
                f"step={step:06d} act={metrics['act_loss']:.4f} bis={metrics['bis_loss']:.4f} total={metrics['total_loss']:.4f}",
                flush=True,
            )
            if metrics["total_loss"] < best_val:
                best_val = metrics["total_loss"]
                _save_checkpoint(checkpoint_path, model, optimizer, config, train_dataset, step, best_val)

    if not checkpoint_path.exists():
        _save_checkpoint(checkpoint_path, model, optimizer, config, train_dataset, config.total_updates, best_val)

    save_json([dict(item) for item in history], output_dir / "history.json")
    save_json(config.__dict__, output_dir / "train_config.json")
    return checkpoint_path


def load_checkpoint_model(checkpoint_path: str | Path, device: str = "auto") -> tuple[torch.nn.Module, dict, DeviceContext]:
    device_ctx = get_device_context(device)
    checkpoint = _safe_torch_load(checkpoint_path)
    model = build_model(
        model_name=checkpoint["model_name"],
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        action_low=np.asarray(checkpoint["action_low"], dtype=np.float32),
        action_high=np.asarray(checkpoint["action_high"], dtype=np.float32),
        d_model=int(checkpoint["model_hparams"]["d_model"]),
        depth=int(checkpoint["model_hparams"]["depth"]),
        dropout=float(checkpoint["model_hparams"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device_ctx.device)
    model.eval()
    return model, checkpoint, device_ctx
