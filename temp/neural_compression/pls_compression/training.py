from __future__ import annotations

import csv
import inspect
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .dataset import SPHDataset, compute_global_stats, discover_sessions
from .losses import LossConfig, hybrid_loss
from .metrics import MetricAccumulator, compute_metrics
from .models import CompressionModel, build_model, load_model_checkpoint, save_checkpoint
from .schema import (
    DEFAULT_LATENT_DIM,
    DEFAULT_MODEL_VARIANT,
    HEIGHT,
    WIDTH,
    DataSchema,
    ModelConfig,
    NormalizationMetadata,
    canonical_model_variant,
    discover_session_directories,
    ensure_compatible_sessions,
)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value.strip()):
        raise ValueError(f"{name} must be an integer")
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass
class TrainingConfig:
    data_dir: str | Path | None = None
    output_dir: str | Path = "attempts"
    model_config: ModelConfig | None = None
    epochs: int = 1
    batch_size: int = 1
    effective_batch_size: int | None = None
    skip_frames: int = 10
    n_steps: int = 1
    skip_initial: int = 1
    device: str | torch.device = "auto"
    max_batches: int | None = None
    smoke: bool = False
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    noise_std: float = 0.0
    gradient_clip_norm: float | None = None
    num_workers: int = 0
    seed: int = 0
    validation_fraction: float = 0.1
    loss_config: LossConfig = field(default_factory=LossConfig)
    normalization: NormalizationMetadata | None = None
    use_amp: bool = False
    bf16: bool = False
    fused: bool = False
    use_8bit_adam: bool = False
    model_filename: str = "best_model.pth"
    min_delta: float = 0.0
    save_every: int = 1
    width: int | None = None
    height: int | None = None
    model_variant: str | None = None
    latent_dim: int | None = None
    base_channels: int | None = None
    bottleneck_channels: int | None = None
    context_channels: int | None = None
    projection_dim: int | None = None
    num_downsamples: int | None = None

    def __post_init__(self) -> None:
        if self.model_config is None:
            values: dict[str, Any] = {
                "width": self.width if self.width is not None else WIDTH,
                "height": self.height if self.height is not None else HEIGHT,
                "model_variant": self.model_variant if self.model_variant is not None else DEFAULT_MODEL_VARIANT,
                "latent_dim": self.latent_dim if self.latent_dim is not None else DEFAULT_LATENT_DIM,
            }
            optional = {
                "base_channels": self.base_channels,
                "bottleneck_channels": self.bottleneck_channels,
                "context_channels": self.context_channels,
                "projection_dim": self.projection_dim,
                "num_downsamples": self.num_downsamples,
            }
            values.update({key: value for key, value in optional.items() if value is not None})
            self.model_config = ModelConfig(**values)
        else:
            if isinstance(self.model_config, Mapping):
                self.model_config = ModelConfig.from_dict(self.model_config)
            changes: dict[str, Any] = {}
            if self.width is not None:
                changes["width"] = self.width
            if self.height is not None:
                changes["height"] = self.height
            if self.model_variant is not None:
                changes["model_variant"] = self.model_variant
            if self.latent_dim is not None:
                changes["latent_dim"] = self.latent_dim
            if self.base_channels is not None:
                changes["base_channels"] = self.base_channels
            if self.bottleneck_channels is not None:
                changes["bottleneck_channels"] = self.bottleneck_channels
            if self.context_channels is not None:
                changes["context_channels"] = self.context_channels
            if self.projection_dim is not None:
                changes["projection_dim"] = self.projection_dim
            if self.num_downsamples is not None:
                changes["num_downsamples"] = self.num_downsamples
            if changes:
                self.model_config = self.model_config.replace(**changes)
        if isinstance(self.loss_config, Mapping):
            self.loss_config = LossConfig(**dict(self.loss_config))
        if isinstance(self.normalization, Mapping):
            self.normalization = NormalizationMetadata.from_dict(self.normalization)
        elif isinstance(self.normalization, (tuple, list)):
            self.normalization = NormalizationMetadata(*self.normalization)
        self.epochs = _integer(self.epochs, "epochs", 1)
        self.batch_size = _integer(self.batch_size, "batch_size", 0)
        if self.effective_batch_size is not None:
            self.effective_batch_size = _integer(self.effective_batch_size, "effective_batch_size", 1)
        self.skip_frames = _integer(self.skip_frames, "skip_frames", 1)
        self.n_steps = _integer(self.n_steps, "n_steps", 1)
        self.skip_initial = _integer(self.skip_initial, "skip_initial", 1)
        self.seed = _integer(self.seed, "seed", 0)
        self.min_delta = float(self.min_delta)
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be non-negative and finite")
        self.save_every = _integer(self.save_every, "save_every", 1)
        self.noise_std = float(self.noise_std)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        if not math.isfinite(float(self.noise_std)) or self.noise_std < 0:
            raise ValueError("noise_std must be non-negative and finite")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
            if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive and finite")
        if self.max_batches is not None:
            self.max_batches = _integer(self.max_batches, "max_batches", 1)
        if int(self.num_workers) != self.num_workers or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        self.num_workers = int(self.num_workers)
        self.validation_fraction = float(self.validation_fraction)
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.smoke and self.max_batches is None:
            self.max_batches = 1
        if self.output_dir is None:
            self.output_dir = "attempts"
        self.output_dir = Path(self.output_dir).expanduser()
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir).expanduser()


@dataclass
class TrainingResult:
    model: CompressionModel
    checkpoint: Path
    best_validation_loss: float
    history: list[dict[str, float]]
    normalization: NormalizationMetadata
    train_sessions: list[Path]
    validation_sessions: list[Path]
    device: torch.device

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_variant": getattr(_base_model(self.model), "model_variant", None),
            "checkpoint": str(self.checkpoint),
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "normalization": self.normalization.to_dict(),
            "train_sessions": [str(path) for path in self.train_sessions],
            "validation_sessions": [str(path) for path in self.validation_sessions],
            "device": str(self.device),
        }


def select_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    else:
        text = str(requested)
        if text == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                device = torch.device(text)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid device {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not hasattr(mps_backend, "is_available"):
            raise RuntimeError("MPS was requested but is not supported by this PyTorch build")
        if not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    return select_device(requested)


def split_sessions(
    session_dirs: Sequence[str | Path],
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    sessions = sorted({Path(path).expanduser().resolve() for path in session_dirs})
    if not sessions:
        raise ValueError("cannot split an empty session list")
    if len(sessions) == 1:
        return sessions, sessions
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * validation_fraction))))
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])
    return training, validation


def _validate_precision(config: TrainingConfig, device: torch.device) -> torch.dtype:
    if config.bf16 and device.type != "cuda":
        raise ValueError("bf16 is a CUDA-only option and cannot be used on the selected CPU device")
    if config.use_amp and device.type != "cuda":
        raise ValueError("AMP is a CUDA-only option and cannot be used on the selected CPU device")
    if config.fused and device.type != "cuda":
        raise ValueError("fused AdamW is a CUDA-only option and cannot be used on the selected CPU device")
    if config.use_8bit_adam and device.type != "cuda":
        raise ValueError("8-bit AdamW is a CUDA-only option and cannot be used on the selected CPU device")
    if config.bf16:
        if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 was requested but the selected CUDA device does not support it")
        return torch.bfloat16
    if config.use_amp:
        return torch.float16
    return torch.float32


def _make_scaler(device: torch.device, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _make_optimizer(model: CompressionModel, config: TrainingConfig, device: torch.device):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("use_8bit_adam was requested but bitsandbytes is unavailable") from exc
        return bnb.optim.AdamW8bit(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    kwargs: dict[str, Any] = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    if config.fused:
        try:
            supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError("fused AdamW was requested but cannot inspect this optimizer") from exc
        if not supports_fused:
            raise RuntimeError("fused AdamW was requested but this PyTorch build does not support it")
        kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def _move_batch(batch: Sequence[Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    values = [value.to(device=device, dtype=torch.float32) for value in batch]
    if len(values) != 5:
        raise ValueError("dataset batches must contain five tensors")
    return values[0], values[1], values[2], values[3], values[4]


def _rollout(
    model: CompressionModel,
    batch: Sequence[Tensor],
    loss_function: LossConfig,
    training: bool,
    noise_std: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    previous_density, previous_velocity, future_density, future_velocity, obstacle_mask = batch
    output_channels = _base_model(model).output_channels
    context_density = previous_density
    context_velocity = previous_velocity
    total_loss: Tensor | None = None
    first_prediction: Tensor | None = None
    first_target: Tensor | None = None
    last_prediction: Tensor | None = None
    last_target: Tensor | None = None
    for step in range(future_density.shape[1]):
        target_density = future_density[:, step]
        target_velocity = future_velocity[:, step]
        prediction = model(
            context_density,
            context_velocity,
            target_density,
            target_velocity,
            obstacle_mask,
            noise_std=noise_std,
        )
        if output_channels == 3:
            target = torch.cat((target_density, target_velocity), dim=1)
        else:
            target = target_density
        step_loss = hybrid_loss(prediction, target, loss_function)
        total_loss = step_loss if total_loss is None else total_loss + step_loss
        if first_prediction is None:
            first_prediction = prediction
            first_target = target
        last_prediction = prediction
        last_target = target
        context_density = prediction[:, 0:1]
        if output_channels == 3:
            context_velocity = prediction[:, 1:3]
        else:
            context_velocity = target_velocity
    if total_loss is None or first_prediction is None or first_target is None or last_prediction is None or last_target is None:
        raise ValueError("future step count must be positive")
    return total_loss / future_density.shape[1], first_prediction, first_target, last_prediction, last_target, previous_density, previous_velocity


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    model: nn.Module,
    gradient_clip_norm: float | None,
) -> None:
    if scaler is not None:
        scaler.unscale_(optimizer)
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def _validate_model_and_loader(
    model: CompressionModel,
    loader: DataLoader,
    device: torch.device,
    loss_function: LossConfig,
    max_batches: int | None,
    amp_enabled: bool,
    precision_dtype: torch.dtype,
    noise_std: float,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    loss_sum = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            values = _move_batch(batch, device)
            with _autocast_context(device, precision_dtype, amp_enabled):
                sequence_loss, first_prediction, first_target, _, _, previous_density, previous_velocity = _rollout(
                    model,
                    values,
                    loss_function,
                    training=False,
                    noise_std=noise_std,
                )
            batch_size = first_target.shape[0]
            metrics = compute_metrics(
                first_prediction,
                first_target,
                previous_density,
                previous_velocity if first_target.shape[1] == 3 else None,
            )
            accumulator.update(metrics, batch_size)
            loss_sum += float(sequence_loss.detach().cpu())
            batch_count += 1
    result = accumulator.compute()
    result["loss"] = loss_sum / max(1, batch_count)
    result["batches"] = float(batch_count)
    return result


def _write_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path = output_dir / "losses.csv"
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def _base_model(model: Any) -> Any:
    return model.module if isinstance(model, nn.DataParallel) else model


def train_model(
    data_dir: str | Path | Sequence[str | Path] | None = None,
    output_dir: str | Path = "attempts",
    config: TrainingConfig | Mapping[str, Any] | None = None,
    sessions: Sequence[str | Path] | None = None,
    model: CompressionModel | None = None,
    resume_path: str | Path | None = None,
    **kwargs: Any,
) -> TrainingResult:
    if config is None:
        if model is not None and hasattr(model, "config"):
            kwargs.setdefault("model_config", model.config)
        config = TrainingConfig(**kwargs)
    elif isinstance(config, Mapping):
        config = TrainingConfig(**dict(config))
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig or mapping")
    if data_dir is not None:
        if isinstance(data_dir, (str, Path)):
            config.data_dir = data_dir
        elif sessions is None:
            sessions = data_dir
    if output_dir is not None and str(output_dir) != "attempts":
        config.output_dir = Path(output_dir).expanduser()
    if sessions is not None:
        session_values = [sessions] if isinstance(sessions, (str, Path)) else list(sessions)
        session_paths: list[Path] = []
        for value in session_values:
            session_paths.extend(discover_session_directories(value))
        session_paths = sorted({path.resolve() for path in session_paths})
    else:
        if config.data_dir is None:
            raise ValueError("data_dir or sessions must be provided")
        if not isinstance(config.data_dir, (str, Path)):
            raise TypeError("data_dir must be a path when sessions is not supplied")
        session_paths = discover_session_directories(config.data_dir)
    if not session_paths:
        raise FileNotFoundError("no simulation sessions were found")
    model_config = config.model_config
    if model_config is None:
        raise ValueError("training requires a model configuration")
    if model is not None:
        supplied_model = _base_model(model)
        supplied_config = getattr(supplied_model, "config", None)
        if supplied_config is not None and supplied_config.model_variant != model_config.model_variant:
            raise ValueError(
                f"model variant {supplied_config.model_variant} does not match training variant {model_config.model_variant}"
            )
    if isinstance(config.loss_config, Mapping):
        config.loss_config = LossConfig(**dict(config.loss_config))
    schema = model_config.schema
    ensure_compatible_sessions(session_paths, schema)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(config.seed)
    device = select_device(config.device)
    precision_dtype = _validate_precision(config, device)
    train_sessions, validation_sessions = split_sessions(session_paths, config.validation_fraction, config.seed)
    normalization = config.normalization
    if normalization is None:
        density_max, velocity_max = compute_global_stats(session_paths, schema)
        normalization = NormalizationMetadata.from_maxima(density_max, velocity_max)
    if isinstance(normalization, Mapping):
        normalization = NormalizationMetadata.from_dict(normalization)
    elif isinstance(normalization, (tuple, list)):
        normalization = NormalizationMetadata(*normalization)
    train_dataset = SPHDataset(
        train_sessions,
        skip=config.skip_frames,
        n_steps=config.n_steps,
        skip_initial=config.skip_initial,
        augment=True,
        schema=schema,
        normalization=normalization,
    )
    validation_dataset = SPHDataset(
        validation_sessions,
        skip=config.skip_frames,
        n_steps=config.n_steps,
        skip_initial=config.skip_initial,
        augment=False,
        schema=schema,
        normalization=normalization,
    )
    if len(train_dataset) == 0:
        raise ValueError("training dataset has no complete future-step samples")
    if len(validation_dataset) == 0:
        train_dataset.close()
        validation_dataset.close()
        raise ValueError("validation dataset has no complete future-step samples")
    batch_size = config.batch_size
    if batch_size == 0:
        batch_size = 1
    effective_batch_size = config.effective_batch_size or batch_size
    if effective_batch_size < batch_size:
        raise ValueError("effective_batch_size must be at least batch_size")
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    active_model = model if model is not None else build_model(model_config.model_variant, model_config)
    active_model.to(device)
    base_active_model = _base_model(active_model)
    base_active_model.data_schema = schema
    base_active_model.normalization = normalization
    if resume_path is not None:
        resumed = load_model_checkpoint(resume_path, device=device, model=active_model)
        active_model = resumed
        base_active_model = _base_model(active_model)
        base_active_model.data_schema = schema
        base_active_model.normalization = normalization
    optimizer = _make_optimizer(active_model, config, device)
    amp_enabled = bool(config.use_amp or config.bf16)
    scaler = _make_scaler(device, amp_enabled)
    output_path = config.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / config.model_filename
    history: list[dict[str, float]] = []
    best_validation_loss = math.inf
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(config.epochs):
            active_model.train()
            train_loss_sum = 0.0
            train_batches = 0
            pending = 0
            for batch_index, batch in enumerate(train_loader):
                if config.max_batches is not None and batch_index >= config.max_batches:
                    break
                values = _move_batch(batch, device)
                with _autocast_context(device, precision_dtype, amp_enabled):
                    sequence_loss, _, _, _, _, _, _ = _rollout(
                        active_model,
                        values,
                        config.loss_config,
                        training=True,
                        noise_std=config.noise_std,
                    )
                    scaled_loss = sequence_loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                pending += 1
                train_loss_sum += float(sequence_loss.detach().cpu())
                train_batches += 1
                if pending >= accumulation_steps:
                    _optimizer_step(optimizer, scaler, active_model, config.gradient_clip_norm)
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
            if pending:
                _optimizer_step(optimizer, scaler, active_model, config.gradient_clip_norm)
                optimizer.zero_grad(set_to_none=True)
            if train_batches == 0:
                raise ValueError("training loader produced no batches")
            validation = _validate_model_and_loader(
                active_model,
                validation_loader,
                device,
                config.loss_config,
                config.max_batches,
                amp_enabled,
                precision_dtype,
                0.0,
            )
            train_loss = train_loss_sum / train_batches
            row = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": float(validation.get("loss", math.inf)),
                "val_mse": float(validation.get("mse", math.inf)),
                "val_zero_mse": float(validation.get("zero_mse", math.inf)),
                "val_identity_mse": float(validation.get("identity_mse", math.inf)),
                "val_fluid_mse": float(validation.get("fluid_mse", math.inf)),
                "val_background_mse": float(validation.get("background_mse", math.inf)),
                "val_ssim": float(validation.get("ssim", math.inf)),
                "val_ssim_density": float(validation.get("ssim_density", math.inf)),
                "val_ssim_fluid": float(validation.get("ssim_fluid", math.inf)),
            }
            history.append(row)
            _write_history(output_path, history)
            current_validation_loss = row["val_loss"]
            improved = math.isfinite(current_validation_loss) and current_validation_loss < best_validation_loss
            # Measure the margin against the previous best before updating it.
            improvement = best_validation_loss - current_validation_loss if improved else 0.0
            if improved:
                # The reported best always tracks the true minimum, even when the
                # write is throttled, so losses.csv and the returned result never
                # disagree with each other.
                best_validation_loss = current_validation_loss
            write_due = (epoch + 1) % config.save_every == 0
            # Writing the full state dict costs one file per improvement, which is
            # 162 MiB for the default model. Throttle on the improvement margin or
            # the epoch interval, but always keep the first write so a run that
            # never improves again still leaves a usable checkpoint.
            should_write = improved and (
                not checkpoint_path.is_file()
                or (improvement >= config.min_delta and write_due)
            )
            if should_write:
                save_checkpoint(
                    checkpoint_path,
                    active_model,
                    normalization=normalization,
                    schema=schema,
                    metadata={
                        "epoch": epoch + 1,
                        "validation_loss": current_validation_loss,
                        "train_loss": train_loss,
                        "device": str(device),
                    },
                )
            active_model.train()
    finally:
        train_dataset.close()
        validation_dataset.close()
    if not math.isfinite(best_validation_loss):
        raise RuntimeError("training did not produce a finite validation checkpoint")
    run_metadata = {
        "model_config": _base_model(active_model).config.to_dict(),
        "normalization": normalization.to_dict(),
        "schema": schema.to_dict(),
        "train_sessions": [str(path) for path in train_sessions],
        "validation_sessions": [str(path) for path in validation_sessions],
        "device": str(device),
        "smoke": config.smoke,
        "max_batches": config.max_batches,
    }
    (output_path / "run_config.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return TrainingResult(
        model=active_model,
        checkpoint=checkpoint_path,
        best_validation_loss=best_validation_loss,
        history=history,
        normalization=normalization,
        train_sessions=train_sessions,
        validation_sessions=validation_sessions,
        device=device,
    )


def get_global_stats(data_dirs: Iterable[str | Path], schema: DataSchema | None = None) -> tuple[float, float]:
    values = [data_dirs] if isinstance(data_dirs, (str, Path)) else data_dirs
    sessions: list[Path] = []
    for value in values:
        sessions.extend(discover_session_directories(value))
    return compute_global_stats(sessions, schema or DataSchema())


def find_max_batch_size(model: CompressionModel, device: str | torch.device, n_steps: int = 1, is_bf16: bool = False) -> int:
    selected = select_device(device)
    if selected.type != "cuda":
        raise RuntimeError("automatic batch-size probing is only available on CUDA")
    dtype = torch.bfloat16 if is_bf16 else torch.float16
    if is_bf16 and (not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bf16 batch-size probing is unsupported by this CUDA device")
    model.to(selected)
    found = 1
    candidates = (1, 2, 4, 8)
    for batch_size in candidates:
        try:
            values = (
                torch.zeros(batch_size, 1, model.config.height, model.config.width, device=selected),
                torch.zeros(batch_size, 2, model.config.height, model.config.width, device=selected),
                torch.zeros(batch_size, n_steps, 1, model.config.height, model.config.width, device=selected),
                torch.zeros(batch_size, n_steps, 2, model.config.height, model.config.width, device=selected),
                torch.zeros(batch_size, 1, model.config.height, model.config.width, device=selected),
            )
            model.train()
            with _autocast_context(selected, dtype, True):
                loss, _, _, _, _, _, _ = _rollout(model, values, LossConfig(), True)
            loss.backward()
            model.zero_grad(set_to_none=True)
            found = batch_size
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                break
            raise
    return found


def train(
    requested_epochs: int | None = None,
    data_dir: str | Path = "data",
    output_dir: str | Path = "attempts",
    model_filename: str = "best_model.pth",
    fluid_weight: float = 50.0,
    mass_loss_weight: float = 0.0,
    args: Any = None,
) -> TrainingResult:
    def argument(name: str, default: Any = None) -> Any:
        return getattr(args, name, default) if args is not None else default

    width = argument("width", None)
    height = argument("height", None)
    variant = argument("model_variant", argument("variant", DEFAULT_MODEL_VARIANT))
    model_config = ModelConfig(
        width=width if width is not None else WIDTH,
        height=height if height is not None else HEIGHT,
        model_variant=variant,
        latent_dim=argument("latent_dim", DEFAULT_LATENT_DIM),
        base_channels=argument("base_channels", 32),
        bottleneck_channels=argument("bottleneck_channels", 64),
        context_channels=argument("context_channels", 32),
        num_downsamples=argument("num_downsamples", 3),
    )
    config_values: dict[str, Any] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "model_config": model_config,
        "epochs": requested_epochs if requested_epochs is not None else argument("epochs", 10),
        "batch_size": argument("batch_size", 1),
        "effective_batch_size": argument("effective_batch_size", None),
        "skip_frames": argument("skip_frames", 10),
        "n_steps": argument("n_steps", 1),
        "skip_initial": argument("skip_initial", 1),
        "device": argument("device", "auto"),
        "max_batches": argument("max_batches", None),
        "smoke": argument("smoke", False),
        "learning_rate": argument("learning_rate", 5e-4),
        "noise_std": argument("noise_std", 0.0),
        "gradient_clip_norm": argument("gradient_clip_norm", None),
        "num_workers": argument("num_workers", 0),
        "seed": argument("seed", 0),
        "use_amp": argument("use_amp", False),
        "bf16": argument("bf16", False),
        "fused": argument("fused", False),
        "use_8bit_adam": argument("use_8bit_adam", False),
        "model_filename": model_filename,
        "loss_config": LossConfig(
            fluid_weight=fluid_weight,
            mass_weight=mass_loss_weight,
            gradient_weight=argument("gradient_weight", 15.0),
            false_negative_weight=argument("false_negative_weight", 5.0),
            mean_weight=argument("mean_weight", 25.0),
        ),
    }
    session_dirs = argument("session_dirs", None)
    if session_dirs:
        if isinstance(session_dirs, str):
            session_dirs = [value.strip() for value in session_dirs.split(",") if value.strip()]
        config_values["data_dir"] = None
    config = TrainingConfig(**config_values)
    return train_model(sessions=session_dirs, config=config)
