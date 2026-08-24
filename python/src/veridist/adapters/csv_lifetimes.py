"""Typed API landing for the strict CSV lifetime adapter.

Parsing and delivery are deliberately not implemented in this landing commit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.delivery import ChunkEnvelope
from veridist.engine.errors import EngineContractError
from veridist.engine.provenance import PublicSourceId

CSV_SCHEMA_VERSION = "1"


class CsvBinarySource(Protocol):
    """Testable binary-source seam; implementations are replayable by contract."""

    def open_binary(self, path: Path) -> object:
        """Open one binary stream."""

    def identity(self, path: Path) -> object:
        """Return an opaque best-effort identity snapshot."""


@dataclass(frozen=True, slots=True)
class CsvLifetimeSchema:
    """Exact names of the two CSV columns accepted by this adapter."""

    time_column: str
    event_observed_column: str

    def __post_init__(self) -> None:
        for value in (self.time_column, self.event_observed_column):
            if not isinstance(value, str) or not value:
                raise ValueError("CSV column names must be non-empty strings")
        if self.time_column == self.event_observed_column:
            raise ValueError("CSV column names must be distinct")


@dataclass(frozen=True, slots=True)
class CsvLifetimeLimits:
    """Logical retained-payload limits for sequential CSV delivery."""

    chunk_bytes: int
    max_inflight_bytes: int

    def __post_init__(self) -> None:
        for value in (self.chunk_bytes, self.max_inflight_bytes):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("CSV byte limits must be built-in integers")
            if value <= 0:
                raise ValueError("CSV byte limits must be positive")
        if self.max_inflight_bytes < self.chunk_bytes:
            raise ValueError("max_inflight_bytes must cover one chunk")


class CsvLifetimeAdapterError(EngineContractError):
    """Typed redacted adapter failure from the closed engine code taxonomy."""


@dataclass(frozen=True, slots=True)
class CsvLifetimeChunk:
    """Typed CSV chunk shape reserved for the later delivery implementation."""

    envelope: ChunkEnvelope
    observations: tuple[object, ...]
    retained_payload_bytes: int


def retained_object_graph_bytes(value: object) -> int:
    """Reserve the declared retained-object accounting seam."""

    del value
    return 0


@dataclass(frozen=True, slots=True)
class CsvLifetimeAdapter:
    """Family-agnostic CSV source declaration; parsing is not landed yet."""

    path: Path
    schema: CsvLifetimeSchema
    source_id: PublicSourceId
    limits: CsvLifetimeLimits
    opener: CsvBinarySource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if type(self.schema) is not CsvLifetimeSchema:
            raise TypeError("schema must be CsvLifetimeSchema")
        if type(self.source_id) is not PublicSourceId:
            raise TypeError("source_id must be PublicSourceId")
        if type(self.limits) is not CsvLifetimeLimits:
            raise TypeError("limits must be CsvLifetimeLimits")

    @property
    def metadata(self) -> DataSourceMetadata:
        """Expose source planning facts without a file path or private revision."""

        return DataSourceMetadata(
            source_id=self.source_id.value,
            schema_version=CSV_SCHEMA_VERSION,
            provenance_schema_version="1",
            replayability=Replayability.REPLAYABLE,
            redaction_reason="hash_unavailable",
        )

    def iter_chunks(self) -> Iterator[object]:
        """Yield parsed chunks once the parsing/delivery behavior is implemented."""

        if False:
            yield None


def fit_exponential_csv(
    path: Path,
    *,
    schema: CsvLifetimeSchema,
    source_id: PublicSourceId,
    limits: CsvLifetimeLimits,
) -> object:
    """Convenience wrapper reserved for the later one-pass execution behavior."""

    del path, schema, source_id, limits
    raise NotImplementedError("CSV exponential execution is not implemented")


__all__ = [
    "CSV_SCHEMA_VERSION",
    "CsvBinarySource",
    "CsvLifetimeAdapter",
    "CsvLifetimeAdapterError",
    "CsvLifetimeChunk",
    "CsvLifetimeLimits",
    "CsvLifetimeSchema",
    "fit_exponential_csv",
    "retained_object_graph_bytes",
]
