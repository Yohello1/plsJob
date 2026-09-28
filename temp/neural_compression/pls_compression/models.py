from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .schema import (
    DEFAULT_LATENT_DIM,
    DEFAULT_MODEL_VARIANT,
    HEIGHT,
    LATENT_DIM,
    WIDTH,
    DataSchema,
    ModelConfig,
    NormalizationMetadata,
    canonical_model_variant,
)


def get_coord_grid(*args: Any, height: int | None = None, width: int | None = None, batch_size: int | None = None, device: Any = None, dtype: torch.dtype = torch.float32) -> Tensor:
    values = list(args)
    if len(values) >= 3:
        first, second, third = values[:3]
        if isinstance(third, (str, torch.device)) or torch.is_tensor(third):
            height = int(first) if height is None else height
            width = int(second) if width is None else width
            device = third if device is None else device
        else:
            batch_size = int(first) if batch_size is None else batch_size
            height = int(second) if height is None else height
            width = int(third) if width is None else width
    elif len(values) == 2:
        height = int(values[0]) if height is None else height
        width = int(values[1]) if width is None else width
    elif len(values) == 1:
        height = int(values[0]) if height is None else height
    if height is None:
        height = HEIGHT
    if width is None:
        width = WIDTH
    if batch_size is None:
        batch_size = 1
    if height <= 0 or width <= 0 or batch_size <= 0:
        raise ValueError("coordinate grid dimensions must be positive")
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    result = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    return result.contiguous()


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    return nn.SiLU(inplace=False)


class ResBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = _activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_channels = config.input_channels
        channel_plan: list[int] = []
        for index in range(config.num_downsamples):
            if index == config.num_downsamples - 1:
                output_channels = config.bottleneck_channels
            else:
                output_channels = min(config.base_channels * (2**index), config.bottleneck_channels)
            channel_plan.append(output_channels)
            layers.append(nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1))
            layers.append(ResBlock(output_channels, config.activation))
            input_channels = output_channels
        self.channel_plan = channel_plan
        self.conv = nn.Sequential(*layers)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(config.bottleneck_channels * config.bottleneck_height * config.bottleneck_width, config.projection_dim)
        self.latent = nn.Linear(config.projection_dim, config.latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.latent(self.fc(self.flat(self.conv(x))))


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        bottleneck_height, bottleneck_width = config.bottleneck_size
        self.fc = nn.Linear(config.latent_dim, config.projection_dim)
        self.expand = nn.Linear(config.projection_dim, config.bottleneck_channels * bottleneck_height * bottleneck_width)
        self.context_stages = nn.ModuleList()
        context_input = config.context_input_channels
        context_channels = config.context_channels
        for index in range(config.num_downsamples):
            output_channels = max(1, min(context_channels * (2**index), config.bottleneck_channels * 2))
            stride = 1 if index == 0 else 2
            self.context_stages.append(
                nn.Sequential(
                    nn.Conv2d(context_input, output_channels, 3, stride=stride, padding=1),
                    _activation(config.activation),
                )
            )
            context_input = output_channels
        self.context_stage_channels = [stage[0].out_channels for stage in self.context_stages]
        self.up_stages = nn.ModuleList()
        current_channels = config.bottleneck_channels
        for index, context_channels in enumerate(self.context_stage_channels):
            final = index == config.num_downsamples - 1
            output_channels = config.output_channels if final else config.bottleneck_channels
            stage: list[nn.Module] = [
                nn.Conv2d(current_channels + context_channels, output_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            ]
            if not final:
                stage.append(ResBlock(output_channels, config.activation))
            self.up_stages.append(nn.Sequential(*stage))
            current_channels = output_channels
        self.residual_blocks = nn.Sequential(
            ResBlock(config.bottleneck_channels, config.activation),
            ResBlock(config.bottleneck_channels, config.activation),
        ) if config.num_downsamples > 1 else nn.Identity()

    @property
    def up_low(self):
        return self.up_stages[0]

    @property
    def up_mid(self):
        return self.up_stages[1] if len(self.up_stages) > 1 else self.up_stages[0]

    @property
    def up_high(self):
        return self.up_stages[-1]

    def _context_features(
        self,
        previous_density: Tensor,
        previous_velocity: Tensor | None,
        obstacle_mask: Tensor,
        coords: Tensor,
    ) -> list[Tensor]:
        if self.config.model_variant == DEFAULT_MODEL_VARIANT:
            context = torch.cat((previous_density, obstacle_mask, coords), dim=1)
        else:
            if previous_velocity is None:
                previous_velocity = torch.zeros_like(previous_density).expand(-1, 2, -1, -1)
            context = torch.cat((previous_density, previous_velocity, obstacle_mask, coords), dim=1)
        features: list[Tensor] = []
        for stage in self.context_stages:
            context = stage(context)
            features.append(context)
        return features

    def forward(
        self,
        z: Tensor,
        previous_density: Tensor,
        previous_velocity: Tensor | None = None,
        obstacle_mask: Tensor | None = None,
        coords: Tensor | None = None,
    ) -> Tensor:
        if coords is None and obstacle_mask is not None and previous_velocity is not None:
            if previous_velocity.ndim == 4 and previous_velocity.shape[1] == 1 and obstacle_mask.ndim == 4 and obstacle_mask.shape[1] == 2:
                coords = obstacle_mask
                obstacle_mask = previous_velocity
                previous_velocity = None
        if obstacle_mask is None or coords is None:
            raise ValueError("decoder requires obstacle_mask and coords")
        bottleneck_height, bottleneck_width = self.config.bottleneck_size
        x = self.expand(self.fc(z)).reshape(z.shape[0], self.config.bottleneck_channels, bottleneck_height, bottleneck_width)
        context_features = self._context_features(previous_density, previous_velocity, obstacle_mask, coords)
        for index, stage in enumerate(self.up_stages):
            context = context_features[index]
            if context.shape[-2:] != x.shape[-2:]:
                context = F.interpolate(context, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = stage(torch.cat((x, context), dim=1))
            if index == self.config.num_downsamples - 2:
                x = self.residual_blocks(x)
        if x.shape[-2:] != (self.config.height, self.config.width):
            x = F.interpolate(x, size=(self.config.height, self.config.width), mode="bilinear", align_corners=False)
        density = torch.sigmoid(x[:, 0:1])
        if self.config.output_channels == 1:
            return density
        return torch.cat((density, torch.tanh(x[:, 1:3])), dim=1)


class CompressionModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig | Mapping[str, Any] | int | None = None,
        latent_dim: int | None = None,
        width: int = WIDTH,
        height: int = HEIGHT,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        base_channels: int | None = None,
        bottleneck_channels: int | None = None,
        context_channels: int | None = None,
        projection_dim: int | None = None,
        num_downsamples: int | None = None,
        activation: str = "silu",
    ):
        super().__init__()
        if isinstance(config, int) and not isinstance(config, bool):
            if latent_dim is not None:
                raise TypeError("latent_dim was specified twice")
            latent_dim = config
            config = None
        if config is None:
            values: dict[str, Any] = {
                "width": width,
                "height": height,
                "model_variant": model_variant,
                "latent_dim": latent_dim if latent_dim is not None else DEFAULT_LATENT_DIM,
                "activation": activation,
            }
            if base_channels is not None:
                values["base_channels"] = base_channels
            if bottleneck_channels is not None:
                values["bottleneck_channels"] = bottleneck_channels
            if context_channels is not None:
                values["context_channels"] = context_channels
            if projection_dim is not None:
                values["projection_dim"] = projection_dim
            if num_downsamples is not None:
                values["num_downsamples"] = num_downsamples
            config = ModelConfig(**values)
        elif isinstance(config, Mapping):
            config = ModelConfig.from_dict(config)
        elif not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig or mapping")
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.normalization: NormalizationMetadata | None = None
        self.data_schema: DataSchema | None = None

    @property
    def model_config(self) -> ModelConfig:
        return self.config

    @property
    def model_variant(self) -> str:
        return self.config.model_variant

    @property
    def output_channels(self) -> int:
        return self.config.output_channels

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _prepare_input(self, value: Tensor, name: str, channels: int) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, channels, height, width], got {tuple(value.shape)}")
        if value.shape[1] != channels:
            raise ValueError(f"{name} must have {channels} channels, got {value.shape[1]}")
        if tuple(value.shape[-2:]) != (self.config.height, self.config.width):
            raise ValueError(
                f"{name} spatial shape {tuple(value.shape[-2:])} does not match model shape "
                f"{(self.config.height, self.config.width)}"
            )
        return value.to(device=self.device, dtype=self.dtype)

    def forward(
        self,
        p_d: Tensor,
        p_v: Tensor | None = None,
        c_d: Tensor | None = None,
        c_v: Tensor | None = None,
        mask: Tensor | None = None,
        noise_std: float = 0.0,
    ) -> Tensor:
        if c_d is None or mask is None:
            raise ValueError("forward requires previous density, target density, and obstacle mask")
        if p_v is None:
            raise ValueError("forward requires previous velocity")
        p_d = self._prepare_input(p_d, "p_d", 1)
        c_d = self._prepare_input(c_d, "c_d", 1)
        if c_v is None:
            c_v = torch.zeros((c_d.shape[0], 2, c_d.shape[2], c_d.shape[3]), device=c_d.device, dtype=c_d.dtype)
        p_v = self._prepare_input(p_v, "p_v", 2)
        c_v = self._prepare_input(c_v, "c_v", 2)
        mask = self._prepare_input(mask, "mask", 1)
        if any(value.shape[0] != p_d.shape[0] for value in (p_v, c_d, c_v, mask)):
            raise ValueError("all model inputs must have the same batch size")
        if self.training and noise_std and float(noise_std) > 0:
            p_d = p_d + torch.randn_like(p_d) * float(noise_std)
            p_v = p_v + torch.randn_like(p_v) * float(noise_std)
        coords = get_coord_grid(
            batch_size=p_d.shape[0],
            height=self.config.height,
            width=self.config.width,
            device=p_d.device,
            dtype=p_d.dtype,
        )
        encoder_input = torch.cat((p_d, p_v, c_d, c_v, mask, coords), dim=1)
        latent = self.encoder(encoder_input)
        return self.decoder(latent, p_d, p_v, mask, coords)

    def get_depth(self) -> int:
        return sum(isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) for module in self.modules())

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()


class DensityModel(CompressionModel):
    def __init__(self, config: ModelConfig | Mapping[str, Any] | int | None = None, **kwargs: Any):
        if isinstance(config, int) and not isinstance(config, bool):
            kwargs.setdefault("latent_dim", config)
            config = None
        if config is None:
            kwargs["model_variant"] = DEFAULT_MODEL_VARIANT
        elif isinstance(config, Mapping):
            config = ModelConfig.from_dict(config).replace(model_variant=DEFAULT_MODEL_VARIANT)
        elif config.model_variant != DEFAULT_MODEL_VARIANT:
            config = config.replace(model_variant=DEFAULT_MODEL_VARIANT)
        super().__init__(config=config, **kwargs)


class DensityVelocityModel(CompressionModel):
    def __init__(self, config: ModelConfig | Mapping[str, Any] | int | None = None, **kwargs: Any):
        if isinstance(config, int) and not isinstance(config, bool):
            kwargs.setdefault("latent_dim", config)
            config = None
        if config is None:
            kwargs["model_variant"] = "density_velocity"
        elif isinstance(config, Mapping):
            config = ModelConfig.from_dict(config).replace(model_variant="density_velocity")
        elif config.model_variant != "density_velocity":
            config = config.replace(model_variant="density_velocity")
        super().__init__(config=config, **kwargs)


FullModel = CompressionModel


def build_model(
    model_variant: str = DEFAULT_MODEL_VARIANT,
    config: ModelConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CompressionModel:
    variant = canonical_model_variant(model_variant)
    if config is None:
        kwargs["model_variant"] = variant
        config = ModelConfig(**kwargs)
    elif isinstance(config, Mapping):
        config = ModelConfig.from_dict(config).replace(model_variant=variant)
    elif config.model_variant != variant:
        config = config.replace(model_variant=variant)
    if variant == DEFAULT_MODEL_VARIANT:
        return DensityModel(config)
    return DensityVelocityModel(config)


def save_checkpoint(
    path: str | Path,
    model: CompressionModel,
    normalization: NormalizationMetadata | Mapping[str, Any] | None = None,
    schema: DataSchema | Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    if normalization is None:
        normalization = base_model.normalization or NormalizationMetadata()
    if schema is None:
        schema = base_model.data_schema or base_model.config.schema
    if isinstance(normalization, Mapping):
        normalization_value = NormalizationMetadata.from_dict(normalization)
    elif isinstance(normalization, (tuple, list)):
        normalization_value = NormalizationMetadata(*normalization)
    else:
        normalization_value = normalization
    if isinstance(schema, Mapping):
        schema_value = DataSchema.from_dict(schema)
    else:
        schema_value = schema
    payload: dict[str, Any] = {
        "format_version": 1,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": base_model.config.to_dict(),
        "normalization": normalization_value.to_dict() if normalization_value is not None else None,
        "normalization_metadata": normalization_value.to_dict() if normalization_value is not None else None,
        "schema": schema_value.to_dict() if schema_value is not None else None,
        "metadata": dict(metadata) if metadata is not None else {},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {source} does not contain a mapping")
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"checkpoint {source} has an invalid state_dict")
        return payload
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return {
            "format_version": 0,
            "state_dict": payload,
            "model_config": None,
            "normalization": None,
            "normalization_metadata": None,
            "schema": None,
            "metadata": {},
        }
    raise ValueError(f"checkpoint {source} does not contain model state")


def _checkpoint_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        device = value
    elif str(value) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint loading was requested but is not available")
    if device.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not hasattr(backend, "is_available") or not backend.is_available():
            raise RuntimeError("MPS checkpoint loading was requested but is not available")
    return device


def load_model_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    model: CompressionModel | None = None,
) -> CompressionModel:
    payload = load_checkpoint(path, map_location="cpu")
    config_value = payload.get("model_config")
    if config_value is None:
        if model is None:
            raise ValueError("legacy state-dict checkpoints require an explicit model")
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        config = base_model.config
    else:
        config = config_value if isinstance(config_value, ModelConfig) else ModelConfig.from_dict(config_value)
    result = model if model is not None else build_model(config.model_variant, config)
    base_result = result.module if isinstance(result, nn.DataParallel) else result
    if base_result.config != config:
        raise ValueError(
            f"checkpoint model configuration does not match the supplied model: "
            f"{config.to_dict()} != {base_result.config.to_dict()}"
        )
    state_dict = payload["state_dict"]
    if any(str(key).startswith("module.") for key in state_dict):
        stripped: dict[str, Any] = {}
        for key, value in state_dict.items():
            name = str(key)
            while name.startswith("module."):
                name = name[7:]
            stripped[name] = value
        state_dict = stripped
        if isinstance(result, nn.DataParallel):
            state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    elif isinstance(result, nn.DataParallel) and not any(str(key).startswith("module.") for key in state_dict):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    result.load_state_dict(state_dict)
    normalization_value = payload.get("normalization")
    if normalization_value is None:
        normalization_value = payload.get("normalization_metadata")
    if normalization_value is not None:
        base_result.normalization = NormalizationMetadata.from_dict(normalization_value)
    schema_value = payload.get("schema")
    if schema_value is not None:
        loaded_schema = schema_value if isinstance(schema_value, DataSchema) else DataSchema.from_dict(schema_value)
        if loaded_schema != config.schema:
            raise ValueError(f"checkpoint schema does not match model configuration: {loaded_schema.to_dict()} != {config.schema.to_dict()}")
        base_result.data_schema = loaded_schema
    result.to(_checkpoint_device(device))
    return result



def load_model(path: str | Path, device: str | torch.device = "cpu") -> CompressionModel:
    return load_model_checkpoint(path, device)


BUFFER_WIDTH = WIDTH
BUFFER_HEIGHT = HEIGHT
