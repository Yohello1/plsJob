from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .dataset import SPHDataset, discover_sessions
from .models import CompressionModel, load_checkpoint, load_model_checkpoint
from .schema import ModelConfig, NormalizationMetadata
from .training import select_device


@dataclass
class EvaluationStep:
    prediction: Tensor
    target: Tensor
    context_density: Tensor
    context_velocity: Tensor
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_shape": list(self.prediction.shape),
            "target_shape": list(self.target.shape),
            "metrics": dict(self.metrics),
        }


@dataclass
class EvaluationResult:
    model_path: Path
    data_dir: Path
    variant: str
    normalization: NormalizationMetadata
    steps: list[EvaluationStep]
    plot: Path | None = None

    @property
    def predictions(self) -> list[Tensor]:
        return [step.prediction for step in self.steps]

    @property
    def targets(self) -> list[Tensor]:
        return [step.target for step in self.steps]

    def to_dict(self) -> dict[str, Any]:
        metrics = [step.metrics for step in self.steps]
        return {
            "model": str(self.model_path),
            "data_dir": str(self.data_dir),
            "variant": self.variant,
            "normalization": self.normalization.to_dict(),
            "steps": metrics,
            "mse": metrics[0]["mse"] if metrics else None,
            "zero_mse": metrics[0]["zero_mse"] if metrics else None,
            "identity_mse": metrics[0].get("identity_mse") if metrics else None,
            "zero_baseline_mse": metrics[0].get("zero_mse") if metrics else None,
            "identity_baseline_mse": metrics[0].get("identity_mse") if metrics else None,
            "mean_mse": sum(value["mse"] for value in metrics) / len(metrics) if metrics else None,
            "plot": str(self.plot) if self.plot is not None else None,
        }


def _normalization_from_payload(payload: dict[str, Any], model: CompressionModel) -> NormalizationMetadata:
    value = payload.get("normalization")
    if value is None:
        value = payload.get("normalization_metadata")
    if value is not None:
        return NormalizationMetadata.from_dict(value)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    return base_model.normalization or NormalizationMetadata()


def load_evaluation_components(
    model_path: str | Path,
    data_dir: str | Path,
    steps: int = 2,
    skip: int = 1,
    device: str | torch.device = "auto",
) -> tuple[CompressionModel, SPHDataset, NormalizationMetadata]:
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(skip, bool) or int(skip) != skip or skip <= 0:
        raise ValueError("skip must be a positive integer")
    selected_device = select_device(device)
    payload = load_checkpoint(model_path, map_location="cpu")
    config_value = payload.get("model_config")
    if config_value is None:
        raise ValueError("evaluation requires a checkpoint containing model_config")
    config = config_value if isinstance(config_value, ModelConfig) else ModelConfig.from_dict(config_value)
    model = load_model_checkpoint(model_path, device=selected_device)
    normalization = _normalization_from_payload(payload, model)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.normalization = normalization
    sessions = discover_sessions(data_dir)
    if not sessions:
        raise FileNotFoundError(f"no simulation sessions found in {data_dir}")
    dataset = SPHDataset(
        sessions,
        skip=int(skip),
        n_steps=int(steps),
        skip_initial=1,
        augment=False,
        schema=config.schema,
        normalization=normalization,
    )
    if len(dataset) == 0:
        dataset.close()
        raise ValueError("evaluation dataset has no complete future-step samples")
    return model, dataset, normalization


def rollout(
    model: CompressionModel,
    dataset: SPHDataset,
    index: int = 0,
    steps: int = 2,
) -> EvaluationResult:
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if index < 0:
        index += len(dataset)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"dataset index {index} is outside length {len(dataset)}")
    previous_density, previous_velocity, future_density, future_velocity, obstacle_mask = dataset[index]
    if future_density.shape[0] < steps:
        raise ValueError(f"dataset sample has {future_density.shape[0]} future steps, requested {steps}")
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    output_channels = base_model.output_channels
    device = next(model.parameters()).device
    context_density = previous_density.unsqueeze(0).to(device)
    context_velocity = previous_velocity.unsqueeze(0).to(device)
    mask = obstacle_mask.unsqueeze(0).to(device)
    evaluations: list[EvaluationStep] = []
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            target_density = future_density[step].unsqueeze(0).to(device)
            target_velocity = future_velocity[step].unsqueeze(0).to(device)
            prediction = model(
                context_density,
                context_velocity,
                target_density,
                target_velocity,
                mask,
            )
            if not isinstance(prediction, Tensor):
                raise TypeError("model evaluation must return a tensor")
            prediction = prediction.float()
            target = target_density.float()
            if output_channels == 3:
                target = torch.cat((target, target_velocity.float()), dim=1)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}"
                )
            from .metrics import compute_metrics

            metrics = compute_metrics(
                prediction,
                target,
                context_density,
                context_velocity if output_channels == 3 else None,
            )
            metrics["step"] = float(step + 1)
            evaluations.append(
                EvaluationStep(
                    prediction=prediction.detach().cpu(),
                    target=target.detach().cpu(),
                    context_density=context_density.detach().cpu(),
                    context_velocity=context_velocity.detach().cpu(),
                    metrics=metrics,
                )
            )
            context_density = prediction[:, 0:1]
            context_velocity = prediction[:, 1:3] if output_channels == 3 else target_velocity
    return EvaluationResult(
        model_path=Path(""),
        data_dir=Path(""),
        variant=base_model.model_variant,
        normalization=base_model.normalization or NormalizationMetadata(),
        steps=evaluations,
    )


def evaluate_checkpoint(
    model_path: str | Path,
    data_dir: str | Path,
    index: int = 0,
    steps: int = 2,
    skip: int = 1,
    device: str | torch.device = "auto",
) -> EvaluationResult:
    model, dataset, normalization = load_evaluation_components(model_path, data_dir, steps, skip, device)
    try:
        result = rollout(model, dataset, index, steps)
    finally:
        dataset.close()
    result.model_path = Path(model_path).expanduser().resolve()
    result.data_dir = Path(data_dir).expanduser().resolve()
    result.normalization = normalization
    return result


def plot_evaluation(result: EvaluationResult, output: str | Path) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = len(result.steps)
    figure, axes = plt.subplots(rows, 3, figsize=(12, max(3, rows * 3)), squeeze=False)
    for row, step in enumerate(result.steps):
        target_density = step.target[0, 0].numpy()
        prediction_density = step.prediction[0, 0].numpy()
        difference = target_density - prediction_density
        images = (
            (target_density, "Target density", "viridis"),
            (prediction_density, "Prediction density", "viridis"),
            (difference, "Target - prediction", "RdBu_r"),
        )
        for column, (values, title, color_map) in enumerate(images):
            axes[row, column].imshow(values, cmap=color_map, origin="lower")
            axes[row, column].set_title(f"{title}, step {row + 1}")
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def print_evaluation(result: EvaluationResult) -> None:
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
