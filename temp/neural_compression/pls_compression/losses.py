from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .schema import DEFAULT_MODEL_VARIANT


@dataclass
class LossConfig:
    fluid_weight: float = 35.0
    mass_weight: float = 0.0
    gradient_weight: float = 15.0
    false_negative_weight: float = 5.0
    mean_weight: float = 25.0
    fluid_threshold: float = 0.05
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        self.fluid_weight = float(self.fluid_weight)
        self.mass_weight = float(self.mass_weight)
        self.gradient_weight = float(self.gradient_weight)
        self.false_negative_weight = float(self.false_negative_weight)
        self.mean_weight = float(self.mean_weight)
        self.fluid_threshold = float(self.fluid_threshold)
        self.epsilon = float(self.epsilon)
        values = (
            self.fluid_weight,
            self.mass_weight,
            self.gradient_weight,
            self.false_negative_weight,
            self.mean_weight,
            self.fluid_threshold,
            self.epsilon,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("loss parameters must be finite")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def _safe_weighted_mean(values: Tensor, weights: Tensor, epsilon: float) -> Tensor:
    expanded_weights = weights.expand_as(values)
    numerator = torch.sum(values * expanded_weights)
    denominator = torch.sum(expanded_weights)
    safe_denominator = denominator.clamp_min(epsilon)
    return torch.where(denominator > epsilon, numerator / safe_denominator, torch.zeros_like(numerator))


def _validate_prediction_target(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must have shape [batch, channels, height, width]")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} must match target shape {tuple(target.shape)}")
    if prediction.shape[1] not in (1, 3):
        raise ValueError("prediction and target must have one or three channels")
    return prediction.float(), target.float()


def hybrid_loss(
    prediction: Tensor,
    target: Tensor,
    config: LossConfig | None = None,
    fluid_weight: float | None = None,
    mass_loss_weight: float | None = None,
    gradient_weight: float | None = None,
    f_weight: float | None = None,
    m_weight: float | None = None,
    grad_weight: float | None = None,
    **kwargs: Any,
) -> Tensor:
    settings = config or LossConfig()
    if fluid_weight is None:
        fluid_weight = f_weight if f_weight is not None else kwargs.get("fluid_weight", settings.fluid_weight)
    if mass_loss_weight is None:
        mass_loss_weight = m_weight if m_weight is not None else kwargs.get("mass_weight", settings.mass_weight)
    if gradient_weight is None:
        gradient_weight = grad_weight if grad_weight is not None else kwargs.get("gradient_weight", settings.gradient_weight)
    prediction, target = _validate_prediction_target(prediction, target)
    epsilon = settings.epsilon
    error = (prediction - target) ** 2
    density_target = target[:, 0:1]
    density_prediction = prediction[:, 0:1]
    fluid_mask = (density_target > settings.fluid_threshold).to(prediction.dtype)
    background_mask = 1.0 - fluid_mask
    fluid_error = _safe_weighted_mean(error, fluid_mask, epsilon)
    background_error = _safe_weighted_mean(error, background_mask, epsilon)
    false_negative_mask = fluid_mask * (density_prediction <= settings.fluid_threshold).to(prediction.dtype)
    false_negative_error = _safe_weighted_mean((density_prediction - density_target) ** 2, false_negative_mask, epsilon)
    fluid_mean_denominator = fluid_mask.sum()
    predicted_fluid_mean = torch.where(
        fluid_mean_denominator > epsilon,
        torch.sum(density_prediction * fluid_mask) / fluid_mean_denominator.clamp_min(epsilon),
        torch.zeros((), device=prediction.device, dtype=prediction.dtype),
    )
    target_fluid_mean = torch.where(
        fluid_mean_denominator > epsilon,
        torch.sum(density_target * fluid_mask) / fluid_mean_denominator.clamp_min(epsilon),
        torch.zeros((), device=prediction.device, dtype=prediction.dtype),
    )
    mean_error = (predicted_fluid_mean - target_fluid_mean) ** 2
    gradient_error = prediction.new_zeros(())
    if prediction.shape[-2] > 1:
        prediction_dx = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        prediction_dy = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        target_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
        mask_x = fluid_mask.expand(-1, prediction.shape[1], -1, -1)[:, :, 1:, :]
        mask_y = fluid_mask.expand(-1, prediction.shape[1], -1, -1)[:, :, :, 1:]
        gradient_error = _safe_weighted_mean((prediction_dx - target_dx) ** 2, mask_x, epsilon)
        gradient_error = gradient_error + _safe_weighted_mean((prediction_dy - target_dy) ** 2, mask_y, epsilon)
    mass_error = prediction.new_zeros(())
    if mass_loss_weight:
        predicted_mass = prediction[:, 0:1].mean(dim=(1, 2, 3))
        target_mass = target[:, 0:1].mean(dim=(1, 2, 3))
        mass_error = (predicted_mass - target_mass).square().mean()
    total = (
        float(fluid_weight) * fluid_error
        + background_error
        + float(fluid_weight) * float(settings.false_negative_weight) * false_negative_error
        + float(settings.mean_weight) * mean_error
        + float(gradient_weight) * gradient_error
        + float(mass_loss_weight) * mass_error
    )
    return total


class HybridLoss(nn.Module):
    def __init__(self, config: LossConfig | None = None):
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return hybrid_loss(prediction, target, self.config)


def reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    prediction, target = _validate_prediction_target(prediction, target)
    return (prediction - target).square().mean()


compute_loss = hybrid_loss
weighted_mse_loss = hybrid_loss
