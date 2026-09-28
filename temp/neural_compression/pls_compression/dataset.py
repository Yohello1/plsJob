from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    DEFAULT_MODEL_VARIANT,
    DTYPE_NAME,
    FIELDS,
    DataSchema,
    ModelConfig,
    NormalizationMetadata,
    SessionMetadata,
    canonical_dtype,
    canonical_model_variant,
    discover_session_directories,
    ensure_compatible_sessions,
    load_session_metadata,
)


@dataclass(frozen=True)
class SampleIndex:
    session_dir: Path
    frame_index: int

    def __iter__(self):
        yield self.session_dir
        yield self.frame_index


class SessionReader:
    def __init__(self, session_dir: str | Path, schema: DataSchema, metadata: SessionMetadata | None = None):
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.schema = schema
        self.metadata = metadata
        self.binary_path = self.session_dir / "sim_data.bin"
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"session has no sim_data.bin: {self.session_dir}")
        if metadata is not None:
            metadata.validate_against(schema)
        self.file_size = self.binary_path.stat().st_size
        self.frame_count = self.file_size // schema.frame_bytes
        if self.frame_count <= 0:
            raise ValueError(
                f"session {self.session_dir} contains no complete {schema.width}x{schema.height} float32 frames; "
                f"file has {self.file_size} bytes and one frame requires {schema.frame_bytes}"
            )
        if metadata is not None and metadata.frame_count is not None and metadata.frame_count != self.frame_count:
            raise ValueError(
                f"session {self.session_dir} metadata declares {metadata.frame_count} frames but contains {self.frame_count}"
            )
        self._mapping: np.memmap | None = None

    @property
    def trailing_bytes(self) -> int:
        return self.file_size % self.schema.frame_bytes

    def _ensure_mapping(self) -> np.memmap:
        if self._mapping is None:
            total_values = self.frame_count * self.schema.frame_values
            self._mapping = np.memmap(
                self.binary_path,
                dtype=np.dtype(self.schema.dtype),
                mode="r",
                shape=(total_values,),
            )
        return self._mapping

    def read_frame(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frame_index < 0:
            frame_index += self.frame_count
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(f"frame index {frame_index} is outside {self.session_dir} with {self.frame_count} frames")
        mapping = self._ensure_mapping()
        start = frame_index * self.schema.frame_values
        end = start + self.schema.frame_values
        values = np.array(mapping[start:end], dtype=np.float32, copy=True).reshape(
            len(FIELDS), self.schema.height, self.schema.width
        )
        density = torch.from_numpy(values[0:1].copy()).contiguous()
        velocity = torch.from_numpy(values[1:3].copy()).contiguous()
        obstacle_mask = torch.from_numpy(values[3:4].copy()).contiguous()
        return density, velocity, obstacle_mask

    def close(self) -> None:
        mapping = self._mapping
        self._mapping = None
        if mapping is not None:
            mmap = getattr(mapping, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "SessionReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mapping"] = None
        return state


def _normalization_from_args(
    normalization: NormalizationMetadata | dict[str, Any] | tuple[float, float] | None,
    density_norm: float | None,
    velocity_norm: float | None,
) -> NormalizationMetadata:
    if normalization is not None:
        if isinstance(normalization, NormalizationMetadata):
            result = normalization
        elif isinstance(normalization, dict):
            result = NormalizationMetadata.from_dict(normalization)
        else:
            try:
                density_value, velocity_value = normalization
            except (TypeError, ValueError) as exc:
                raise ValueError("normalization must be metadata, a mapping, or a pair") from exc
            result = NormalizationMetadata(density_value, velocity_value)
        if density_norm is not None or velocity_norm is not None:
            result = NormalizationMetadata(
                density_norm if density_norm is not None else result.density_scale,
                velocity_norm if velocity_norm is not None else result.velocity_scale,
                result.density_max,
                result.velocity_max,
            )
        return result
    return NormalizationMetadata(
        density_norm if density_norm is not None else 1.0,
        velocity_norm if velocity_norm is not None else 1.0,
    )


def _field_tuple(fields: Sequence[str] | str | None, default: Sequence[str] = FIELDS) -> tuple[str, ...]:
    if fields is None:
        return tuple(default)
    if isinstance(fields, str):
        return tuple(part.strip() for part in fields.split(",") if part.strip())
    return tuple(fields)


def _coerce_schema(
    schema: DataSchema | dict[str, Any] | None,
    width: int | None,
    height: int | None,
    model_variant: str | None,
    dtype: Any,
    fields: Sequence[str] | None,
) -> DataSchema:
    if isinstance(schema, DataSchema):
        result = schema
        if width is not None or height is not None or model_variant is not None or dtype is not None or fields is not None:
            result = DataSchema(
                width if width is not None else result.width,
                height if height is not None else result.height,
                _field_tuple(fields) if fields is not None else result.fields,
                canonical_dtype(dtype) if dtype is not None else result.dtype,
                canonical_model_variant(model_variant) if model_variant is not None else result.model_variant,
            )
        return result
    if isinstance(schema, dict):
        result = DataSchema.from_dict(schema)
    else:
        result = DataSchema(
            width=width if width is not None else DataSchema().width,
            height=height if height is not None else DataSchema().height,
            fields=_field_tuple(fields) if fields is not None else FIELDS,
            dtype=canonical_dtype(dtype) if dtype is not None else DTYPE_NAME,
            model_variant=canonical_model_variant(model_variant) if model_variant is not None else DEFAULT_MODEL_VARIANT,
        )
    if width is not None or height is not None or model_variant is not None or fields is not None or dtype is not None:
        result = DataSchema(
            width if width is not None else result.width,
            height if height is not None else result.height,
            _field_tuple(fields) if fields is not None else result.fields,
            canonical_dtype(dtype) if dtype is not None else result.dtype,
            canonical_model_variant(model_variant) if model_variant is not None else result.model_variant,
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, str):
        if not value.strip().isdigit():
            raise ValueError(f"{name} must be a positive integer")
        result = int(value.strip())
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if result != value:
            raise ValueError(f"{name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _discover(data_dirs: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(data_dirs, (str, Path)):
        return discover_session_directories(data_dirs)
    result: list[Path] = []
    seen: set[Path] = set()
    for value in data_dirs:
        sessions = discover_session_directories(value)
        for session in sessions:
            if session not in seen:
                result.append(session)
                seen.add(session)
    if not result:
        raise FileNotFoundError("no session directories containing sim_data.bin were found")
    return result


class SPHDataset(Dataset):
    def __init__(
        self,
        data_dirs: str | Path | Sequence[str | Path],
        skip: int = 10,
        n_steps: int = 1,
        skip_initial: int = 1,
        augment: bool = False,
        schema: DataSchema | dict[str, Any] | None = None,
        width: int | None = None,
        height: int | None = None,
        model_variant: str | None = None,
        dtype: Any = None,
        fields: Sequence[str] | None = None,
        normalization: NormalizationMetadata | dict[str, Any] | tuple[float, float] | None = None,
        density_norm: float | None = None,
        velocity_norm: float | None = None,
    ):
        skip = _positive_int(skip, "skip")
        n_steps = _positive_int(n_steps, "n_steps")
        skip_initial = _positive_int(skip_initial, "skip_initial")
        self.data_dirs = _discover(data_dirs)
        first_metadata = None
        for session_dir in self.data_dirs:
            first_metadata = load_session_metadata(session_dir)
            if first_metadata is not None:
                break
        if schema is None:
            schema_base = first_metadata.schema if first_metadata is not None else DataSchema()
        else:
            schema_base = schema
        self.schema = _coerce_schema(schema_base, width, height, model_variant, dtype, fields)
        self.skip = int(skip)
        self.n_steps = int(n_steps)
        self.skip_initial = int(skip_initial)
        self.augment = bool(augment)
        self.normalization = _normalization_from_args(normalization, density_norm, velocity_norm)
        self.frame_size = self.schema.frame_bytes
        self.field_size = self.schema.height * self.schema.width * np.dtype(self.schema.dtype).itemsize
        self.order = {"d": 0, "v_x": 1, "v_y": 2, "m": 3}
        metadata = ensure_compatible_sessions(self.data_dirs, self.schema)
        self.session_metadata = {item.session_dir: item for item in metadata}
        self.readers = {session: SessionReader(session, self.schema, self.session_metadata[session]) for session in self.data_dirs}
        self.samples: list[SampleIndex] = []
        for session in self.data_dirs:
            reader = self.readers[session]
            available = reader.frame_count - self.n_steps * self.skip
            for frame_index in range(max(0, available)):
                self.samples.append(SampleIndex(session, frame_index))
        if self.skip_initial > 1:
            self.samples = self.samples[:: self.skip_initial]

    @property
    def density_norm(self) -> float:
        return self.normalization.density_scale

    @property
    def velocity_norm(self) -> float:
        return self.normalization.velocity_scale

    @property
    def session_dirs(self) -> list[Path]:
        return list(self.data_dirs)

    @property
    def handles(self) -> dict[Path, SessionReader]:
        return self.readers

    def set_norms(self, density_norm: float, velocity_norm: float) -> "SPHDataset":
        self.normalization = NormalizationMetadata(density_norm, velocity_norm)
        return self

    def set_normalization(self, normalization: NormalizationMetadata | dict[str, Any] | tuple[float, float]) -> "SPHDataset":
        self.normalization = _normalization_from_args(normalization, None, None)
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def _reader_for(self, session_dir: str | Path) -> SessionReader:
        path = Path(session_dir).expanduser().resolve()
        if path in self.readers:
            return self.readers[path]
        metadata = self.session_metadata.get(path) or load_session_metadata(path, self.schema)
        if metadata is not None:
            metadata.validate_against(self.schema)
        reader = SessionReader(path, self.schema, metadata)
        self.readers[path] = reader
        return reader

    def load_frame_data(
        self,
        data_dir: str | Path,
        frame_idx: int,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        density, velocity, obstacle_mask = self._reader_for(data_dir).read_frame(frame_idx)
        if normalize:
            density = density * self.normalization.density_scale
            velocity = velocity * self.normalization.velocity_scale
        return density, velocity, obstacle_mask

    def _augment_frame(
        self,
        density: torch.Tensor,
        velocity: torch.Tensor,
        obstacle_mask: torch.Tensor,
        flip_horizontal: bool,
        flip_vertical: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if flip_horizontal:
            density = torch.flip(density, dims=(-1,))
            velocity = torch.flip(velocity, dims=(-1,))
            velocity = torch.cat((-velocity[0:1], velocity[1:2]), dim=0)
            obstacle_mask = torch.flip(obstacle_mask, dims=(-1,))
        if flip_vertical:
            density = torch.flip(density, dims=(-2,))
            velocity = torch.flip(velocity, dims=(-2,))
            velocity = torch.cat((velocity[0:1], -velocity[1:2]), dim=0)
            obstacle_mask = torch.flip(obstacle_mask, dims=(-2,))
        return density, velocity, obstacle_mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"dataset index {index} is outside length {len(self.samples)}")
        sample = self.samples[index]
        reader = self.readers[sample.session_dir]
        flip_horizontal = self.augment and random.random() >= 0.5
        flip_vertical = self.augment and random.random() >= 0.5
        densities: list[torch.Tensor] = []
        velocities: list[torch.Tensor] = []
        obstacle_mask: torch.Tensor | None = None
        for step in range(self.n_steps + 1):
            density, velocity, mask = reader.read_frame(sample.frame_index + step * self.skip)
            density = density * self.normalization.density_scale
            velocity = velocity * self.normalization.velocity_scale
            density, velocity, mask = self._augment_frame(
                density,
                velocity,
                mask,
                flip_horizontal,
                flip_vertical,
            )
            densities.append(density)
            velocities.append(velocity)
            if step == 0:
                obstacle_mask = mask
        if obstacle_mask is None:
            raise RuntimeError("sequence did not produce a context frame")
        future_density = torch.stack(densities[1:], dim=0)
        future_velocity = torch.stack(velocities[1:], dim=0)
        return densities[0], velocities[0], future_density, future_velocity, obstacle_mask

    def close(self) -> None:
        for reader in list(self.readers.values()):
            reader.close()
        self.readers.clear()

    def __enter__(self) -> "SPHDataset":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["readers"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.readers = {session: SessionReader(session, self.schema, self.session_metadata[session]) for session in self.data_dirs}


def compute_global_stats(
    session_dirs: Iterable[str | Path],
    schema: DataSchema | None = None,
) -> tuple[float, float]:
    schema = schema or DataSchema()
    max_density = 0.0
    max_velocity = 0.0
    for session_dir in session_dirs:
        path = Path(session_dir).expanduser().resolve()
        reader = SessionReader(path, schema)
        try:
            mapping = reader._ensure_mapping()
            for frame_index in range(reader.frame_count):
                start = frame_index * schema.frame_values
                end = start + schema.frame_values
                frame = np.asarray(mapping[start:end]).reshape(len(FIELDS), schema.height, schema.width)
                max_density = max(max_density, float(np.max(frame[0])))
                max_velocity = max(max_velocity, float(np.max(np.abs(frame[1:3]))))
        finally:
            reader.close()
    return max_density, max_velocity


def normalization_from_sessions(
    session_dirs: Iterable[str | Path],
    schema: DataSchema,
) -> NormalizationMetadata:
    density_max, velocity_max = compute_global_stats(session_dirs, schema)
    return NormalizationMetadata.from_maxima(density_max, velocity_max)


SessionDataset = SPHDataset


def discover_sessions(data_path: str | Path) -> list[Path]:
    return discover_session_directories(data_path)
