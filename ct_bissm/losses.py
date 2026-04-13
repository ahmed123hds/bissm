from __future__ import annotations

import math

import torch


def _atanh_clamped(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = torch.clamp(x, min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh_gaussian_nll(
    actions: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    unit_actions = (actions - bias) / torch.clamp(scale, min=1e-6)
    pre_tanh = _atanh_clamped(unit_actions)
    std = torch.exp(log_std)
    normal_log_prob = -0.5 * (
        ((pre_tanh - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi)
    )
    normal_log_prob = normal_log_prob.sum(dim=-1)
    tanh_correction = torch.log(1.0 - unit_actions.pow(2) + 1e-6).sum(dim=-1)
    scale_correction = torch.log(torch.clamp(scale, min=1e-6)).sum(dim=-1)
    return -(normal_log_prob - tanh_correction - scale_correction)


def bisimulation_target(
    mean_reward_a: torch.Tensor,
    mean_reward_b: torch.Tensor,
    next_feature_a: torch.Tensor,
    next_feature_b: torch.Tensor,
    reward_weight: float = 0.5,
    feature_weight: float = 0.5,
) -> torch.Tensor:
    reward_term = torch.abs(mean_reward_a - mean_reward_b)
    feature_term = torch.norm(next_feature_a - next_feature_b, dim=-1)
    return reward_weight * reward_term + feature_weight * feature_term


def bisimulation_regression_loss(
    projection_a: torch.Tensor,
    projection_b: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    latent_distance = torch.norm(projection_a - projection_b, dim=-1)
    return torch.mean((latent_distance - target) ** 2)

