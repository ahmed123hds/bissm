from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, d_model: int = 256) -> None:
        super().__init__()
        self.state_encoder = MLP([state_dim, 256, 256])
        self.action_encoder = MLP([action_dim, 128, 128])
        self.reward_encoder = nn.Linear(1, 64)
        self.rtg_encoder = nn.Linear(1, 64)
        self.time_encoder = nn.Linear(1, 64)
        self.done_encoder = nn.Linear(1, 16)
        self.input_projection = nn.Linear(256 + 128 + 64 + 64 + 64 + 16, d_model)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            [
                self.state_encoder(batch["states"]),
                self.action_encoder(batch["prev_actions"]),
                self.reward_encoder(batch["prev_rewards"]),
                self.rtg_encoder(batch["returns_to_go"]),
                self.time_encoder(batch["log_delta_t"]),
                self.done_encoder(batch["dones"]),
            ],
            dim=-1,
        )
        return self.input_projection(x)


class CTSelectiveSSMBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.input_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        use_actual_dt: bool = True,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, hidden_dim = x.shape
        if state is None:
            state = x.new_zeros(batch_size, hidden_dim)
        decay = F.softplus(self.log_decay).view(1, hidden_dim) + 1e-4
        outputs = []
        for step in range(horizon):
            dt = delta_t[:, step, :]
            if not use_actual_dt:
                dt = torch.ones_like(dt)
            dt = dt.clamp(min=1e-4)
            lamb = torch.exp(-decay * dt)
            update = self.input_proj(x[:, step, :])
            gate = torch.sigmoid(self.gate_proj(x[:, step, :]))
            state = lamb * state + (1.0 - lamb) * (gate * update)
            output = self.norm(state + self.dropout(self.output_proj(state)))
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class CTSSMBackbone(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CTSelectiveSSMBlock(d_model=d_model, dropout=dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor, use_actual_dt: bool = True) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden, _ = layer(hidden, delta_t=delta_t, use_actual_dt=use_actual_dt)
        return self.final_norm(hidden)


class TimeAwareTransformerBackbone(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 4, dropout: float = 0.1, n_heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor, use_actual_dt: bool = True) -> torch.Tensor:
        del delta_t, use_actual_dt
        horizon = x.shape[1]
        mask = torch.triu(torch.full((horizon, horizon), float("-inf"), device=x.device), diagonal=1)
        return self.norm(self.encoder(x, mask=mask))


class GaussianActionHead(nn.Module):
    def __init__(self, d_model: int, action_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.rtg_encoder = nn.Linear(1, 64)
        self.head = MLP([d_model + 64, 256, 256, 2 * action_dim], dropout=dropout)

    def forward(self, hidden: torch.Tensor, returns_to_go: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rtg_embedding = self.rtg_encoder(returns_to_go)
        params = self.head(torch.cat([hidden, rtg_embedding], dim=-1))
        mean, log_std = params.chunk(2, dim=-1)
        return mean, torch.clamp(log_std, min=-5.0, max=2.0)


class BisimProjectionHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.head = MLP([d_model, 128, 64])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BaseSequencePolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        d_model: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_encoder = TokenEncoder(state_dim=state_dim, action_dim=action_dim, d_model=d_model)
        self.action_head = GaussianActionHead(d_model=d_model, action_dim=action_dim, dropout=dropout)
        self.projection_head = BisimProjectionHead(d_model=d_model)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32).view(1, 1, -1))
        self.use_actual_dt = True

    def sequence_forward(self, tokens: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = self.token_encoder(batch)
        hidden = self.sequence_forward(tokens, batch["delta_t"])
        mean, log_std = self.action_head(hidden, batch["returns_to_go"])
        return {"hidden": hidden, "mean": mean, "log_std": log_std}

    def project_hidden(self, hidden: torch.Tensor, mode: str = "last") -> torch.Tensor:
        if mode == "mean":
            features = hidden.mean(dim=1)
        else:
            features = hidden[:, -1, :]
        return self.projection_head(features)

    def predict_action(self, batch: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        outputs = self.forward(batch)
        mean = outputs["mean"][:, -1, :]
        log_std = outputs["log_std"][:, -1, :]
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + torch.randn_like(mean) * torch.exp(log_std)
        action_unit = torch.tanh(pre_tanh)
        scale = (self.action_high[:, -1, :] - self.action_low[:, -1, :]) / 2.0
        bias = (self.action_high[:, -1, :] + self.action_low[:, -1, :]) / 2.0
        return action_unit * scale + bias


class CTBiSSMPolicy(BaseSequencePolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", 256)
        depth = kwargs.get("depth", 4)
        dropout = kwargs.get("dropout", 0.1)
        self.backbone = CTSSMBackbone(d_model=d_model, depth=depth, dropout=dropout)
        self.use_actual_dt = True

    def sequence_forward(self, tokens: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens, delta_t=delta_t, use_actual_dt=True)


class FixedStepSSMPolicy(BaseSequencePolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", 256)
        depth = kwargs.get("depth", 4)
        dropout = kwargs.get("dropout", 0.1)
        self.backbone = CTSSMBackbone(d_model=d_model, depth=depth, dropout=dropout)
        self.use_actual_dt = False

    def sequence_forward(self, tokens: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens, delta_t=delta_t, use_actual_dt=False)


class TimeAwareTransformerPolicy(BaseSequencePolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", 256)
        depth = kwargs.get("depth", 4)
        dropout = kwargs.get("dropout", 0.1)
        self.backbone = TimeAwareTransformerBackbone(d_model=d_model, depth=depth, dropout=dropout)

    def sequence_forward(self, tokens: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens, delta_t=delta_t, use_actual_dt=True)


def build_model(
    model_name: str,
    state_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    d_model: int = 256,
    depth: int = 4,
    dropout: float = 0.1,
) -> BaseSequencePolicy:
    model_name = model_name.lower()
    common = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_low": action_low,
        "action_high": action_high,
        "d_model": d_model,
        "depth": depth,
        "dropout": dropout,
    }
    if model_name == "ct_bissm":
        return CTBiSSMPolicy(**common)
    if model_name == "fixed_ssm":
        return FixedStepSSMPolicy(**common)
    if model_name in {"time_transformer", "dt_time"}:
        return TimeAwareTransformerPolicy(**common)
    raise ValueError(f"Unknown model_name '{model_name}'.")

