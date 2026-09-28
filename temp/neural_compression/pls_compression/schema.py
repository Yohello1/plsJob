from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


WIDTH = 400
HEIGHT = 400
FIELDS = ("density", "velocity_x", "velocity_y", "obstacle_mask")
DTYPE = np.float32
DTYPE_NAME = "float32"
FIELD_COUNT = len(FIELDS)
DEFAULT_MODEL_VARIANT = "density"
DENSITY_MODEL_VARIANT = DEFAULT_MODEL_VARIANT
DENSITY_VELOCITY_MODEL_VARIANT = "density_velocity"
MODEL_VARIANTS = (DEFAULT_MODEL_VARIANT, DENSITY_VELOCITY_MODEL_VARIANT)
MODEL_VARIANT_LABELS = {
    DEFAULT_MODEL_VARIANT: "density-only",
    DENSITY_VELOCITY_MODEL_VARIANT: "density-velocity",
}
DEFAULT_LATENT_DIM = 1024
DEFAULT_DOWNSAMPLES = 3
DEFAULT_DENSITY_SCALE = 1.0
DEFAULT_VELOCITY_SCALE = 1.0

_BUFFER_WIDTH = WIDTH
_BUFFER_HEIGHT = HEIGHT
_LATENT_DIM = DEFAULT_LATENT_DIM

_VARIANT_ALIASES = {
    "density": DEFAULT_MODEL_VARIANT,
    "density_only": DEFAULT_MODEL_VARIANT,
    "densityonly": DEFAULT_MODEL_VARIANT,
    "density-only": DEFAULT_MODEL_VARIANT,
    "d": DEFAULT_MODEL_VARIANT,
    "density_velocity": DENSITY_VELOCITY_MODEL_VARIANT,
    "densityvelocity": DENSITY_VELOCITY_MODEL_VARIANT,
    "density-velocity": DENSITY_VELOCITY_MODEL_VARIANT,
    "density+velocity": DENSITY_VELOCITY_MODEL_VARIANT,
    "density_velocity_model": DENSITY_VELOCITY_MODEL_VARIANT,
    "dv": DENSITY_VELOCITY_MODEL_VARIANT,
    "velocity": DENSITY_VELOCITY_MODEL_VARIANT,
    "velocity_density": DENSITY_VELOCITY_MODEL_VARIANT,
    "d+v": DENSITY_VELOCITY_MODEL_VARIANT,
    "d_v": DENSITY_VELOCITY_MODEL_VARIANT,
}

_FIELD_ALIASES = {
    "density": "density",
    "d": "density",
    "densityfield": "density",
    "velocityx": "velocity_x",
    "velocity_x": "velocity_x",
    "v_x": "velocity_x",
    "vx": "velocity_x",
    "velocityy": "velocity_y",
    "velocity_y": "velocity_y",
    "v_y": "velocity_y",
    "vy": "velocity_y",
    "obstaclemask": "obstacle_mask",
    "obstacle_mask": "obstacle_mask",
    "mask": "obstacle_mask",
    "fluidmask": "obstacle_mask",
    "fluid_mask": "obstacle_mask",
}

_METADATA_FILENAMES = (
    "metadata.json",
    "session.json",
    "session_metadata.json",
    "sim_data.json",
    "config.json",
    "manifest.json",
    "metadata.yaml",
    "metadata.yml",
    "metadata.txt",
    "sim_data.meta",
)


def canonical_model_variant(value: Any) -> str:
    if isinstance(value, str):
        key = re.sub(r"[^a-z0-9_+-]", "", value.strip().lower())
        if key in _VARIANT_ALIASES:
            return _VARIANT_ALIASES[key]
        compact = key.replace("-", "_").replace("+", "_")
        if compact in _VARIANT_ALIASES:
            return _VARIANT_ALIASES[compact]
    raise ValueError(f"unsupported model_variant {value!r}; expected one of {MODEL_VARIANTS}")


def model_variant_label(value: Any) -> str:
    return MODEL_VARIANT_LABELS[canonical_model_variant(value)]


def canonical_field(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"field names must be strings, got {value!r}")
    key = re.sub(r"[^a-z0-9]", "", value.strip().lower())
    if key in _FIELD_ALIASES:
        return _FIELD_ALIASES[key]
    raise ValueError(f"unsupported field {value!r}; expected one of {FIELDS}")


def canonical_dtype(value: Any) -> str:
    if value is None:
        return DTYPE_NAME
    text = str(value).strip().lower()
    if text in {"float", "single", "f4", "<f4", ">f4", "np.float32", "numpy.float32", "torch.float32", "float32"}:
        return DTYPE_NAME
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported dtype {value!r}; expected float32") from exc
    if dtype != np.dtype(DTYPE):
        raise ValueError(f"unsupported dtype {dtype}; expected float32")
    return DTYPE_NAME


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)) or int(value) != value:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    elif not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class DataSchema:
    width: int = WIDTH
    height: int = HEIGHT
    fields: tuple[str, ...] = FIELDS
    dtype: str = DTYPE_NAME
    model_variant: str = DEFAULT_MODEL_VARIANT

    def __post_init__(self) -> None:
        width = _as_positive_int(self.width, "width")
        height = _as_positive_int(self.height, "height")
        fields_value = re.split(r"[,|]", self.fields) if isinstance(self.fields, str) else self.fields
        if fields_value is None:
            fields_value = FIELDS
        fields = tuple(canonical_field(field) for field in fields_value)
        if fields != FIELDS:
            raise ValueError(f"fields must be exactly {FIELDS}, got {fields}")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        object.__setattr__(self, "model_variant", canonical_model_variant(self.model_variant))

    @property
    def frame_values(self) -> int:
        return FIELD_COUNT * self.height * self.width

    @property
    def frame_bytes(self) -> int:
        return self.frame_values * np.dtype(self.dtype).itemsize

    @property
    def output_channels(self) -> int:
        return 1 if self.model_variant == DEFAULT_MODEL_VARIANT else 3

    @property
    def model_input_channels(self) -> int:
        return 9

    def validate(self) -> "DataSchema":
        self.__post_init__()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fields": list(self.fields),
            "dtype": self.dtype,
            "model_variant": self.model_variant,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataSchema":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("schema metadata must be a mapping")
        nested = value.get("schema")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({key: val for key, val in value.items() if key != "schema"})
            value = merged
        fields_value = value.get("fields", FIELDS)
        if isinstance(fields_value, str):
            fields_value = tuple(part.strip() for part in re.split(r"[,|]", fields_value) if part.strip())
        elif isinstance(fields_value, (int, float, np.integer, np.floating)):
            if int(fields_value) != FIELD_COUNT:
                raise ValueError(f"fields must contain exactly {FIELD_COUNT} entries")
            fields_value = FIELDS
        elif fields_value is None:
            fields_value = FIELDS
        else:
            fields_value = tuple(fields_value)
        return cls(
            width=value.get("width", WIDTH),
            height=value.get("height", HEIGHT),
            fields=fields_value,
            dtype=value.get("dtype", DTYPE_NAME),
            model_variant=value.get("model_variant", value.get("variant", DEFAULT_MODEL_VARIANT)),
        )

    def with_dimensions(self, width: int, height: int) -> "DataSchema":
        return DataSchema(width, height, self.fields, self.dtype, self.model_variant)


@dataclass(frozen=True)
class NormalizationMetadata:
    density_scale: float = DEFAULT_DENSITY_SCALE
    velocity_scale: float = DEFAULT_VELOCITY_SCALE
    density_max: float | None = None
    velocity_max: float | None = None

    def __post_init__(self) -> None:
        density_scale = float(self.density_scale)
        velocity_scale = float(self.velocity_scale)
        if not math.isfinite(density_scale) or density_scale <= 0:
            raise ValueError("density_scale must be finite and positive")
        if not math.isfinite(velocity_scale) or velocity_scale <= 0:
            raise ValueError("velocity_scale must be finite and positive")
        object.__setattr__(self, "density_scale", density_scale)
        object.__setattr__(self, "velocity_scale", velocity_scale)
        if self.density_max is not None:
            density_max = float(self.density_max)
            if not math.isfinite(density_max) or density_max < 0:
                raise ValueError("density_max must be finite and non-negative")
            object.__setattr__(self, "density_max", density_max)
        if self.velocity_max is not None:
            velocity_max = float(self.velocity_max)
            if not math.isfinite(velocity_max) or velocity_max < 0:
                raise ValueError("velocity_max must be finite and non-negative")
            object.__setattr__(self, "velocity_max", velocity_max)

    @classmethod
    def from_maxima(cls, density_max: float, velocity_max: float) -> "NormalizationMetadata":
        density_max = float(density_max)
        velocity_max = float(velocity_max)
        if not math.isfinite(density_max) or not math.isfinite(velocity_max):
            raise ValueError("normalization maxima must be finite")
        density_max = max(0.0, density_max)
        velocity_max = max(0.0, velocity_max)
        density_scale = 1.0 / density_max if density_max > 0 else DEFAULT_DENSITY_SCALE
        velocity_scale = 1.0 / velocity_max if velocity_max > 0 else DEFAULT_VELOCITY_SCALE
        return cls(density_scale, velocity_scale, density_max, velocity_max)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "NormalizationMetadata":
        if value is None:
            return cls()
        if isinstance(value, NormalizationMetadata):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("normalization metadata must be a mapping")
        return cls(
            density_scale=value.get("density_scale", value.get("density_norm", DEFAULT_DENSITY_SCALE)),
            velocity_scale=value.get("velocity_scale", value.get("velocity_norm", DEFAULT_VELOCITY_SCALE)),
            density_max=value.get("density_max"),
            velocity_max=value.get("velocity_max"),
        )

    def apply(self, density: Any, velocity: Any) -> tuple[Any, Any]:
        return density * self.density_scale, velocity * self.velocity_scale


@dataclass(frozen=True)
class ModelConfig:
    width: int = WIDTH
    height: int = HEIGHT
    model_variant: str = DEFAULT_MODEL_VARIANT
    latent_dim: int = DEFAULT_LATENT_DIM
    base_channels: int = 32
    bottleneck_channels: int = 64
    context_channels: int = 32
    projection_dim: int = 128
    num_downsamples: int = DEFAULT_DOWNSAMPLES
    activation: str = "silu"

    def __post_init__(self) -> None:
        width = _as_positive_int(self.width, "width")
        height = _as_positive_int(self.height, "height")
        latent_dim = _as_positive_int(self.latent_dim, "latent_dim")
        base_channels = _as_positive_int(self.base_channels, "base_channels")
        bottleneck_channels = _as_positive_int(self.bottleneck_channels, "bottleneck_channels")
        context_channels = _as_positive_int(self.context_channels, "context_channels")
        projection_dim = _as_positive_int(self.projection_dim, "projection_dim")
        num_downsamples = _as_positive_int(self.num_downsamples, "num_downsamples")
        if num_downsamples > 6:
            raise ValueError("num_downsamples must not exceed 6")
        activation = str(self.activation).strip().lower()
        if activation not in {"silu", "relu", "gelu", "tanh"}:
            raise ValueError(f"unsupported activation {self.activation!r}")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "latent_dim", latent_dim)
        object.__setattr__(self, "base_channels", base_channels)
        object.__setattr__(self, "bottleneck_channels", bottleneck_channels)
        object.__setattr__(self, "context_channels", context_channels)
        object.__setattr__(self, "projection_dim", projection_dim)
        object.__setattr__(self, "num_downsamples", num_downsamples)
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "model_variant", canonical_model_variant(self.model_variant))

    @property
    def downsample_factor(self) -> int:
        return 2**self.num_downsamples

    @property
    def bottleneck_width(self) -> int:
        return ceil_div(self.width, self.downsample_factor)

    @property
    def bottleneck_height(self) -> int:
        return ceil_div(self.height, self.downsample_factor)

    @property
    def bottleneck_size(self) -> tuple[int, int]:
        return self.bottleneck_height, self.bottleneck_width

    @property
    def output_channels(self) -> int:
        return 1 if self.model_variant == DEFAULT_MODEL_VARIANT else 3

    @property
    def input_channels(self) -> int:
        return 9

    @property
    def context_input_channels(self) -> int:
        return 4 if self.model_variant == DEFAULT_MODEL_VARIANT else 6

    @property
    def schema(self) -> DataSchema:
        return DataSchema(self.width, self.height, FIELDS, DTYPE_NAME, self.model_variant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "model_variant": self.model_variant,
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "context_channels": self.context_channels,
            "projection_dim": self.projection_dim,
            "num_downsamples": self.num_downsamples,
            "activation": self.activation,
            "bottleneck_width": self.bottleneck_width,
            "bottleneck_height": self.bottleneck_height,
            "output_channels": self.output_channels,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("model configuration must be a mapping")
        nested = value.get("schema")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({key: val for key, val in value.items() if key != "schema"})
            value = merged
        return cls(
            width=value.get("width", WIDTH),
            height=value.get("height", HEIGHT),
            model_variant=value.get("model_variant", value.get("variant", DEFAULT_MODEL_VARIANT)),
            latent_dim=value.get("latent_dim", value.get("latent", DEFAULT_LATENT_DIM)),
            base_channels=value.get("base_channels", 32),
            bottleneck_channels=value.get("bottleneck_channels", 64),
            context_channels=value.get("context_channels", 32),
            projection_dim=value.get("projection_dim", 128),
            num_downsamples=value.get("num_downsamples", DEFAULT_DOWNSAMPLES),
            activation=value.get("activation", "silu"),
        )

    def replace(self, **changes: Any) -> "ModelConfig":
        values = {
            "width": self.width,
            "height": self.height,
            "model_variant": self.model_variant,
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "context_channels": self.context_channels,
            "projection_dim": self.projection_dim,
            "num_downsamples": self.num_downsamples,
            "activation": self.activation,
        }
        values.update(changes)
        return ModelConfig(**values)


@dataclass(frozen=True)
class SessionMetadata:
    session_dir: Path
    width: int
    height: int
    fields: tuple[str, ...]
    dtype: str
    model_variant: str
    frame_count: int | None = None
    source: Path | None = None
    extra: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_dir", Path(self.session_dir).resolve())
        object.__setattr__(self, "width", _as_positive_int(self.width, "width"))
        object.__setattr__(self, "height", _as_positive_int(self.height, "height"))
        fields_value = re.split(r"[,|]", self.fields) if isinstance(self.fields, str) else self.fields
        if fields_value is None:
            fields_value = FIELDS
        fields = tuple(canonical_field(field) for field in fields_value)
        if fields != FIELDS:
            raise ValueError(f"metadata fields must be exactly {FIELDS}, got {fields}")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        object.__setattr__(self, "model_variant", canonical_model_variant(self.model_variant))
        if self.frame_count is not None:
            object.__setattr__(self, "frame_count", _as_positive_int(self.frame_count, "frame_count"))

    @property
    def schema(self) -> DataSchema:
        return DataSchema(self.width, self.height, self.fields, self.dtype, self.model_variant)

    def validate_against(self, expected: DataSchema) -> None:
        mismatches: list[str] = []
        if self.width != expected.width:
            mismatches.append(f"width {self.width} != {expected.width}")
        if self.height != expected.height:
            mismatches.append(f"height {self.height} != {expected.height}")
        if self.fields != expected.fields:
            mismatches.append(f"fields {self.fields} != {expected.fields}")
        if self.dtype != expected.dtype:
            mismatches.append(f"dtype {self.dtype} != {expected.dtype}")
        if self.model_variant != expected.model_variant:
            mismatches.append(f"model_variant {self.model_variant!r} != {expected.model_variant!r}")
        if mismatches:
            location = f" from {self.source}" if self.source is not None else ""
            raise ValueError(f"incompatible session metadata for {self.session_dir}{location}: " + "; ".join(mismatches))

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fields": list(self.fields),
            "dtype": self.dtype,
            "model_variant": self.model_variant,
            "frame_count": self.frame_count,
            "source": str(self.source) if self.source is not None else None,
            "extra": dict(self.extra) if self.extra is not None else {},
        }


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        normalized = _normalise_key(key)
        full = normalized if not prefix else f"{prefix}_{normalized}"
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, full))
        else:
            flattened.setdefault(full, item)
            flattened.setdefault(normalized, item)
    return flattened


def _find_value(values: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        key = _normalise_key(name)
        if key in values:
            return values[key]
    for key, value in values.items():
        if any(_normalise_key(name) in key for name in names):
            return value
    return None


def _parse_resolution(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    if isinstance(value, str):
        parts = re.split(r"[,xX:]", value)
        if len(parts) >= 2:
            return parts[0], parts[1]
    return None


def _read_metadata_file(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read session metadata {path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"([^:=]+)\s*[:=]\s*(.*)$", line)
            if match:
                value[match.group(1).strip()] = match.group(2).strip()
        if not value:
            raise ValueError(f"session metadata {path} is not valid JSON or key/value data")
    if not isinstance(value, Mapping):
        raise ValueError(f"session metadata {path} must contain an object")
    return value


def session_metadata_from_mapping(
    session_dir: str | Path,
    value: Mapping[str, Any],
    source: str | Path | None = None,
    defaults: DataSchema | None = None,
) -> SessionMetadata:
    flattened = _flatten_mapping(value)
    width = _find_value(flattened, ("width", "buffer_width", "grid_width", "resolution_width", "n_w", "w"))
    height = _find_value(flattened, ("height", "buffer_height", "grid_height", "resolution_height", "n_h", "h"))
    resolution = _find_value(flattened, ("resolution", "grid_resolution", "shape", "dimensions"))
    if (width is None or height is None) and resolution is not None:
        parsed = _parse_resolution(resolution)
        if parsed is not None:
            width = width if width is not None else parsed[0]
            height = height if height is not None else parsed[1]
    if width is None:
        width = defaults.width if defaults is not None else WIDTH
    if height is None:
        height = defaults.height if defaults is not None else HEIGHT
    fields_value = _find_value(flattened, ("fields", "field_order", "field_names", "channels"))
    numeric_channels = isinstance(fields_value, (int, float, np.integer, np.floating)) and int(fields_value) == 4
    fields_text = str(fields_value).strip() if isinstance(fields_value, str) else None
    if fields_value is None or numeric_channels or fields_text == "4":
        fields = defaults.fields if defaults is not None else FIELDS
    elif isinstance(fields_value, str):
        fields = tuple(part.strip() for part in re.split(r"[,|]", fields_value) if part.strip())
    else:
        try:
            fields = tuple(fields_value)
        except TypeError as exc:
            raise ValueError("metadata fields must be a sequence") from exc
    dtype = _find_value(flattened, ("dtype", "data_type", "scalar_type", "element_type", "format"))
    if dtype is None:
        dtype = defaults.dtype if defaults is not None else DTYPE_NAME
    variant = _find_value(flattened, ("model_variant", "variant", "architecture", "mode", "model"))
    if variant is None:
        variant = defaults.model_variant if defaults is not None else DEFAULT_MODEL_VARIANT
    frame_count = _find_value(flattened, ("frame_count", "frames", "num_frames", "n_frames"))
    return SessionMetadata(
        session_dir=session_dir,
        width=width,
        height=height,
        fields=fields,
        dtype=dtype,
        model_variant=variant,
        frame_count=frame_count,
        source=Path(source) if source is not None else None,
        extra=dict(value),
    )


def load_session_metadata(
    session_dir: str | Path,
    defaults: DataSchema | None = None,
) -> SessionMetadata | None:
    directory = Path(session_dir).resolve()
    for filename in _METADATA_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            value = _read_metadata_file(candidate)
            return session_metadata_from_mapping(directory, value, candidate, defaults)
    return None


def discover_session_directories(data_path: str | Path) -> list[Path]:
    root = Path(data_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data path does not exist: {root}")
    if root.is_file():
        raise ValueError(f"data path must be a directory: {root}")
    if (root / "sim_data.bin").is_file():
        return [root]
    sessions = sorted(
        child.resolve()
        for child in root.iterdir()
        if child.is_dir() and (child / "sim_data.bin").is_file()
    )
    return sessions


def ensure_compatible_sessions(
    session_dirs: list[str | Path],
    schema: DataSchema,
) -> list[SessionMetadata]:
    metadata: list[SessionMetadata] = []
    for session_dir in session_dirs:
        directory = Path(session_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"session directory does not exist: {directory}")
        if not (directory / "sim_data.bin").is_file():
            raise FileNotFoundError(f"session has no sim_data.bin: {directory}")
        item = load_session_metadata(directory, schema)
        if item is None:
            item = SessionMetadata(
                session_dir=directory,
                width=schema.width,
                height=schema.height,
                fields=schema.fields,
                dtype=schema.dtype,
                model_variant=schema.model_variant,
            )
        else:
            item.validate_against(schema)
        metadata.append(item)
    return metadata


BUFFER_WIDTH = WIDTH
BUFFER_HEIGHT = HEIGHT
LATENT_DIM = DEFAULT_LATENT_DIM
MODEL_VARIANT = DEFAULT_MODEL_VARIANT
DENSITY_NORM = DEFAULT_DENSITY_SCALE
VELOCITY_NORM = DEFAULT_VELOCITY_SCALE

SchemaConfig = DataSchema
DatasetSchema = DataSchema
NormalizationConfig = NormalizationMetadata
ModelSettings = ModelConfig
