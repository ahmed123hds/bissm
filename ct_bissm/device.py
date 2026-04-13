from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


@dataclass
class DeviceContext:
    device: torch.device
    is_xla: bool = False
    xm: Any = None

    def move_batch(self, batch: Any) -> Any:
        return _move(batch, self.device)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.is_xla:
            self.xm.optimizer_step(optimizer, barrier=False)
        else:
            optimizer.step()

    def mark_step(self) -> None:
        if self.is_xla:
            self.xm.mark_step()


def get_device_context(requested: str = "auto") -> DeviceContext:
    requested = requested.lower()
    if requested == "tpu":
        import torch_xla.core.xla_model as xm

        return DeviceContext(device=xm.xla_device(), is_xla=True, xm=xm)
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return DeviceContext(device=torch.device("cuda"))
    if requested == "cpu":
        return DeviceContext(device=torch.device("cpu"))
    if torch.cuda.is_available():
        return DeviceContext(device=torch.device("cuda"))
    return DeviceContext(device=torch.device("cpu"))

