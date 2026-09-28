from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


DEFAULT_SSIM_WINDOW_SIZE = 11
DEFAULT_SSIM_SIGMA = 1.5
DEFAULT_DATA_RANGE = 1.0
DEFAULT_SSIM_PADDING = "same"
DEFAULT_FLUID_THRESHOLD = 0.05
DEFAULT_SSIM_EPSILON = 1e-6
SSIM_PADDING_MODES = ("same", "valid")
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _validate(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} must match target shape {tuple(target.shape)}")
    if prediction.ndim < 2:
        raise ValueError("metrics require a channel dimension")
    return prediction.float(), target.float()


def mse(prediction: Tensor, target: Tensor) -> Tensor:
    prediction, target = _validate(prediction, target)
    return (prediction - target).square().mean()


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    prediction, target = _validate(prediction, target)
    return (prediction - target).abs().mean()


def zero_baseline_mse(target: Tensor) -> Tensor:
    if not isinstance(target, Tensor):
        raise TypeError("target must be a torch tensor")
    return target.float().square().mean()


def identity_baseline_mse(target: Tensor, previous_density: Tensor, previous_velocity: Tensor | None = None) -> Tensor:
    if not isinstance(target, Tensor) or not isinstance(previous_density, Tensor):
        raise TypeError("target and previous_density must be torch tensors")
    if target.shape[1] == 1:
        prediction = previous_density
    else:
        if previous_velocity is None:
            raise ValueError("three-channel metrics require previous_velocity")
        prediction = torch.cat((previous_density, previous_velocity), dim=1)
    if prediction.shape != target.shape:
        raise ValueError("identity baseline inputs do not match target shape")
    return mse(prediction, target)


def _masked_mse(prediction: Tensor, target: Tensor, threshold: float = 0.05, epsilon: float = 1e-6) -> tuple[Tensor, Tensor]:
    density_target = target[:, 0:1]
    density_prediction = prediction[:, 0:1]
    mask = (density_target > threshold).to(prediction.dtype)
    error = (prediction - target).square()
    expanded = mask.expand_as(error)
    fluid_denominator = expanded.sum().clamp_min(epsilon)
    background = (1.0 - mask).expand_as(error)
    background_denominator = background.sum().clamp_min(epsilon)
    fluid = (error * expanded).sum() / fluid_denominator
    background_error = (error * background).sum() / background_denominator
    return fluid, background_error


def _validate_ssim(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch tensors")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(prediction.shape)} must match target shape {tuple(target.shape)}")
    if prediction.ndim < 3:
        raise ValueError("structural similarity requires a channel dimension and two spatial dimensions")
    if not (prediction.is_floating_point() and target.is_floating_point()):
        prediction, target = prediction.float(), target.float()
    if prediction.dtype == torch.float64 or target.dtype == torch.float64:
        return prediction.double(), target.double()
    return prediction.float(), target.float()


def _odd_window(window_size: int, height: int, width: int) -> int:
    limit = min(height, width)
    if limit < 3:
        return 0
    size = min(int(window_size), limit)
    if size % 2 == 0:
        size -= 1
    return size if size >= 3 else 0


@lru_cache(maxsize=32)
def _gaussian_kernel(window_size: int, sigma: float, device: Any, dtype: Any) -> Tensor:
    positions = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = positions - (window_size - 1) / 2.0
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _ssim_filter(value: Tensor, window_size: int, sigma: float, padding: str) -> Tensor:
    kernel = _gaussian_kernel(window_size, sigma, value.device, value.dtype)
    weight = (kernel[:, None] * kernel[None, :]).to(value.dtype)
    weight = weight.expand(value.shape[1], 1, window_size, window_size).contiguous()
    return F.conv2d(value, weight, padding=padding, groups=value.shape[1])


def _ssim_map(
    prediction: Tensor,
    target: Tensor,
    window_size: int,
    sigma: float,
    c1: float,
    c2: float,
    padding: str,
) -> Tensor:
    mean_prediction = _ssim_filter(prediction, window_size, sigma, padding)
    mean_target = _ssim_filter(target, window_size, sigma, padding)
    mean_product = mean_prediction * mean_target
    variance_prediction = _ssim_filter(prediction * prediction, window_size, sigma, padding) - mean_product
    variance_target = _ssim_filter(target * target, window_size, sigma, padding) - mean_product
    covariance = _ssim_filter(prediction * target, window_size, sigma, padding) - mean_product
    luminance = (2.0 * mean_product + c1) / (mean_prediction**2 + mean_target**2 + c1)
    structure = (2.0 * covariance + c2) / (variance_prediction + variance_target + c2)
    return luminance * structure


def _global_ssim_map(
    prediction: Tensor,
    target: Tensor,
    c1: float,
    c2: float,
) -> Tensor:
    mean_prediction = prediction.mean(dim=(-2, -1), keepdim=True)
    mean_target = target.mean(dim=(-2, -1), keepdim=True)
    variance_prediction = prediction.var(dim=(-2, -1), unbiased=False, keepdim=True)
    variance_target = target.var(dim=(-2, -1), unbiased=False, keepdim=True)
    covariance = ((prediction - mean_prediction) * (target - mean_target)).mean(dim=(-2, -1), keepdim=True)
    luminance = (2.0 * mean_prediction * mean_target + c1) / (
        mean_prediction**2 + mean_target**2 + c1
    )
    structure = (2.0 * covariance + c2) / (variance_prediction + variance_target + c2)
    return luminance * structure


def _ssim_maps(
    prediction: Tensor,
    target: Tensor,
    window_size: int = DEFAULT_SSIM_WINDOW_SIZE,
    sigma: float = DEFAULT_SSIM_SIGMA,
    data_range: float = DEFAULT_DATA_RANGE,
    padding: str = DEFAULT_SSIM_PADDING,
) -> tuple[Tensor, Tensor]:
    """Per-channel maps ``[..., channels, out_height, out_width]``.

    The density mask is returned alongside so every SSIM summary shares one
    convolution pass.
    """
    prediction, target = _validate_ssim(prediction, target)
    data_range = float(data_range)
    if not data_range > 0:
        raise ValueError("data_range must be positive")
    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if int(window_size) != window_size or window_size < 3 or int(window_size) % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")
    window_size = int(window_size)
    if padding not in SSIM_PADDING_MODES:
        raise ValueError(f"padding must be one of {SSIM_PADDING_MODES}, got {padding!r}")
    height, width = prediction.shape[-2], prediction.shape[-1]
    leading = prediction.shape[:-3]
    channels = prediction.shape[-3]
    flat_prediction = prediction.reshape(-1, channels, height, width)
    flat_target = target.reshape(-1, channels, height, width)
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    effective_window = _odd_window(window_size, height, width)
    if effective_window == 0:
        per_channel = _global_ssim_map(flat_prediction, flat_target, c1, c2).expand(-1, -1, height, width)
    else:
        conv_padding = effective_window // 2 if padding == "same" else 0
        per_channel = _ssim_map(flat_prediction, flat_target, effective_window, sigma, c1, c2, conv_padding)
    out_height, out_width = per_channel.shape[-2], per_channel.shape[-1]
    maps = per_channel.reshape(*leading, channels, out_height, out_width)
    mask = (target[..., 0:1, :, :] > DEFAULT_FLUID_THRESHOLD).to(maps.dtype)
    return maps, mask


def ssim_map(
    prediction: Tensor,
    target: Tensor,
    window_size: int = DEFAULT_SSIM_WINDOW_SIZE,
    sigma: float = DEFAULT_SSIM_SIGMA,
    data_range: float = DEFAULT_DATA_RANGE,
    padding: str = DEFAULT_SSIM_PADDING,
) -> Tensor:
    """Per-pixel structural similarity with shape ``[..., out_height, out_width]``.

    Channels are averaged, matching the channel-agnostic default of reference
    implementations. Select the density channel by slicing ``[..., 0:1, :, :]``
    to score it alone. ``padding="same"`` keeps the map aligned with the input;
    ``padding="valid"`` crops the border and matches reference implementations
    that only score fully covered windows. The default is ``"same"``.
    """
    maps, _ = _ssim_maps(prediction, target, window_size, sigma, data_range, padding)
    return maps.mean(dim=-3)


def ssim(
    prediction: Tensor,
    target: Tensor,
    window_size: int = DEFAULT_SSIM_WINDOW_SIZE,
    sigma: float = DEFAULT_SSIM_SIGMA,
    data_range: float = DEFAULT_DATA_RANGE,
    padding: str = DEFAULT_SSIM_PADDING,
) -> Tensor:
    """Mean structural similarity over every channel, image, and pixel."""
    return ssim_map(prediction, target, window_size, sigma, data_range, padding).mean()


def ssim_fluid(
    prediction: Tensor,
    target: Tensor,
    window_size: int = DEFAULT_SSIM_WINDOW_SIZE,
    sigma: float = DEFAULT_SSIM_SIGMA,
    data_range: float = DEFAULT_DATA_RANGE,
    padding: str = DEFAULT_SSIM_PADDING,
) -> Tensor:
    """Structural similarity averaged over the fluid region only.

    Fluid occupies a small fraction of a 400x400 field, so an unmasked mean is
    dominated by background agreement and scores a total fluid dropout at about
    0.99. This averages the same per-pixel map over target density above
    ``DEFAULT_FLUID_THRESHOLD``, the region the loss already treats as fluid.
    Returns zero when no target pixel is fluid, matching the masked means in
    ``losses._safe_weighted_mean``.
    """
    maps, mask = _ssim_maps(prediction, target, window_size, sigma, data_range, padding)
    weights = mask
    numerator = (maps * weights).sum()
    denominator = weights.expand_as(maps).sum()
    return torch.where(
        denominator > DEFAULT_SSIM_EPSILON,
        numerator / denominator.clamp_min(DEFAULT_SSIM_EPSILON),
        torch.zeros_like(numerator),
    )


def structural_similarity(
    prediction: Tensor,
    target: Tensor,
    window_size: int = DEFAULT_SSIM_WINDOW_SIZE,
    sigma: float = DEFAULT_SSIM_SIGMA,
    data_range: float = DEFAULT_DATA_RANGE,
) -> Tensor:
    return ssim(prediction, target, window_size, sigma, data_range)


def compute_metrics(
    prediction: Tensor,
    target: Tensor,
    previous_density: Tensor | None = None,
    previous_velocity: Tensor | None = None,
) -> dict[str, float]:
    prediction, target = _validate(prediction, target)
    fluid, background = _masked_mse(prediction, target)
    result: dict[str, float] = {
        "mse": float(mse(prediction, target).detach().cpu()),
        "mae": float(mae(prediction, target).detach().cpu()),
        "zero_mse": float(zero_baseline_mse(target).detach().cpu()),
        "zero_baseline_mse": float(zero_baseline_mse(target).detach().cpu()),
        "fluid_mse": float(fluid.detach().cpu()),
        "background_mse": float(background.detach().cpu()),
    }
    if prediction.ndim >= 3:
        maps, weights = _ssim_maps(prediction, target)
        denominator = weights.expand_as(maps).sum()
        result["ssim"] = float(maps.mean().detach().cpu())
        result["ssim_density"] = float(maps[..., 0:1, :, :].mean().detach().cpu())
        result["ssim_fluid"] = float(
            torch.where(
                denominator > DEFAULT_SSIM_EPSILON,
                (maps * weights).sum() / denominator.clamp_min(DEFAULT_SSIM_EPSILON),
                torch.zeros((), device=maps.device, dtype=maps.dtype),
            )
            .detach()
            .cpu()
        )
    if previous_density is not None:
        if target.shape[1] == 1:
            identity = float(mse(previous_density, target).detach().cpu())
        elif previous_velocity is not None:
            identity = float(identity_baseline_mse(target, previous_density, previous_velocity).detach().cpu())
        else:
            identity = None
        if identity is not None:
            result["identity_mse"] = identity
            result["identity_baseline_mse"] = identity
    return result


def baseline_metrics(
    prediction: Tensor,
    target: Tensor,
    previous_density: Tensor | None = None,
    previous_velocity: Tensor | None = None,
) -> dict[str, float]:
    return compute_metrics(prediction, target, previous_density, previous_velocity)


@dataclass
class MetricAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, values: dict[str, float], weight: int = 1) -> None:
        if weight <= 0:
            return
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * weight
        self.count += weight

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}


calculate_metrics = compute_metrics
