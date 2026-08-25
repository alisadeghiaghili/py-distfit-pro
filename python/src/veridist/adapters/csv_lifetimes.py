"""Strict sequential CSV lifetime ingestion with no dialect or type inference."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections.abc import Generator, Iterator
from dataclasses import dataclass, fields, is_dataclass, replace
from decimal import Decimal
from io import BufferedIOBase, TextIOWrapper
from math import isfinite
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.delivery import ChunkEnvelope
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.provenance import PublicSourceId

CSV_SCHEMA_VERSION = "1"
_TIME = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_CHUNK_ID = re.compile(r"chk_[0-9a-f]{32}")
_CSV_FAILURE_CODES = frozenset(
    {
        FailureCode.SOURCE_OPEN_FAILED,
        FailureCode.SOURCE_DECODE_FAILED,
        FailureCode.SOURCE_SCHEMA_INVALID,
        FailureCode.SOURCE_ROW_INVALID,
        FailureCode.CHUNK_TOO_LARGE,
        FailureCode.SOURCE_REVISION_MISMATCH,
        FailureCode.SOURCE_REVISION_UNAVAILABLE,
    }
)
_REASONS = frozenset(
    {
        "open_failed",
        "invalid_utf8",
        "header_missing",
        "header_duplicate",
        "header_columns_mismatch",
        "blank_record",
        "malformed_record",
        "invalid_time",
        "invalid_event_token",
        "record_too_large",
        "source_mutated",
        "identity_unavailable",
    }
)


class CsvBinarySource(Protocol):
    """A replayable, testable binary source seam."""

    def open_binary(self, path: Path) -> BinaryIO:
        """Open a fresh binary stream for one iterator acquisition."""

    def identity(self, path: Path) -> object:
        """Return an opaque best-effort identity snapshot."""


class _FilesystemCsvSource:
    """Private local-file source; its path/identity never enters public output."""

    def open_binary(self, path: Path) -> BinaryIO:
        return path.open("rb")

    def identity(self, path: Path) -> object:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


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

    def __init__(self, code: FailureCode, *, reason: str, record_offset: int | None = None) -> None:
        if code not in _CSV_FAILURE_CODES:
            raise ValueError("unsupported CSV adapter failure code")
        if reason not in _REASONS:
            raise ValueError("unsupported CSV adapter failure reason")
        context: dict[str, object] = {"reason": reason}
        if record_offset is not None:
            if isinstance(record_offset, bool) or not isinstance(record_offset, int):
                raise TypeError("record_offset must be an integer")
            if record_offset < 0:
                raise ValueError("record_offset must be non-negative")
            context["record_offset"] = record_offset
        super().__init__(code, context)


@dataclass(frozen=True, slots=True)
class CsvLifetimeChunk:
    """One bounded parsed chunk with no raw CSV cells or source path."""

    envelope: ChunkEnvelope
    observations: tuple[LifetimeObservation, ...]
    retained_payload_bytes: int

    def __post_init__(self) -> None:
        if type(self.envelope) is not ChunkEnvelope:
            raise TypeError("envelope must be ChunkEnvelope")
        if type(self.observations) is not tuple:
            raise TypeError("observations must be a tuple")
        if any(
            type(item) not in {ExactLifetime, RightCensoredLifetime} for item in self.observations
        ):
            raise TypeError("observations must be lifetime observations")
        if isinstance(self.retained_payload_bytes, bool) or not isinstance(
            self.retained_payload_bytes, int
        ):
            raise TypeError("retained_payload_bytes must be an integer")
        if self.retained_payload_bytes <= 0:
            raise ValueError("retained_payload_bytes must be positive")
        if self.envelope.row_count != len(self.observations):
            raise ValueError("envelope range must match observations")
        if self.envelope.byte_size != self.retained_payload_bytes:
            raise ValueError("envelope byte size must equal retained payload bytes")


def retained_object_graph_bytes(value: object) -> int:
    """Return closed owned-object accounting for a chunk's retained graph."""

    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, type) or isinstance(item, type(sys)):
            return 0
        total = sys.getsizeof(item)
        if type(item) is tuple:
            return total + sum(visit(child) for child in item)
        if is_dataclass(item) and not isinstance(item, type):
            return total + sum(visit(getattr(item, field.name)) for field in fields(item))
        return total

    return visit(value)


@dataclass(frozen=True, slots=True)
class CsvLifetimeAdapter:
    """Family-neutral strict CSV source with replayable sequential chunks."""

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
        if self.opener is not None:
            if not callable(getattr(self.opener, "open_binary", None)):
                raise TypeError("opener must define open_binary")
            if not callable(getattr(self.opener, "identity", None)):
                raise TypeError("opener must define identity")

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

    def iter_chunks(self) -> Iterator[CsvLifetimeChunk]:
        """Parse one fresh source pass into bounded, ordered lifetime chunks."""

        source = self.opener if self.opener is not None else _FilesystemCsvSource()
        try:
            binary = source.open_binary(self.path)
        except OSError:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_OPEN_FAILED, reason="open_failed"
            ) from None
        text: TextIOWrapper | None = None
        primary_error: BaseException | None = None
        try:
            try:
                before_identity = source.identity(self.path)
            except OSError:
                raise CsvLifetimeAdapterError(
                    FailureCode.SOURCE_REVISION_UNAVAILABLE,
                    reason="identity_unavailable",
                ) from None
            if before_identity is None:
                raise CsvLifetimeAdapterError(
                    FailureCode.SOURCE_REVISION_UNAVAILABLE,
                    reason="identity_unavailable",
                )
            if not isinstance(binary, BufferedIOBase) and not hasattr(binary, "read"):
                raise TypeError("source must return a binary readable stream")
            text = TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = cast(Iterator[list[str]], csv.reader(text, strict=True))
            header = self._read_header(reader)
            self._validate_header(header)
            semantic_failure = yield from self._read_chunks(reader)
            try:
                after_identity = source.identity(self.path)
            except OSError:
                raise CsvLifetimeAdapterError(
                    FailureCode.SOURCE_REVISION_UNAVAILABLE,
                    reason="identity_unavailable",
                ) from None
            if after_identity is None:
                raise CsvLifetimeAdapterError(
                    FailureCode.SOURCE_REVISION_UNAVAILABLE,
                    reason="identity_unavailable",
                )
            if after_identity != before_identity:
                raise CsvLifetimeAdapterError(
                    FailureCode.SOURCE_REVISION_MISMATCH,
                    reason="source_mutated",
                )
            if semantic_failure is not None:
                raise semantic_failure
        except UnicodeDecodeError as error:
            primary_error = error
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_DECODE_FAILED, reason="invalid_utf8"
            ) from None
        except csv.Error as error:
            primary_error = error
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID, reason="malformed_record"
            ) from None
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                if text is not None:
                    text.close()
                else:
                    binary.close()
            except Exception:
                if primary_error is None:
                    raise

    def _read_header(self, reader: Iterator[list[str]]) -> list[str]:
        try:
            return next(reader)
        except StopIteration:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_SCHEMA_INVALID, reason="header_missing"
            ) from None

    def _validate_header(self, header: list[str]) -> None:
        if len(header) != len(set(header)):
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_SCHEMA_INVALID, reason="header_duplicate"
            )
        if header != [self.schema.time_column, self.schema.event_observed_column]:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_SCHEMA_INVALID,
                reason="header_columns_mismatch",
            )

    def _read_chunks(
        self, reader: Iterator[list[str]]
    ) -> Generator[CsvLifetimeChunk, None, CsvLifetimeAdapterError | None]:
        records: list[LifetimeObservation] = []
        start = 0
        record_offset = 0
        sequence = 0
        failure: CsvLifetimeAdapterError | None = None
        for row in reader:
            try:
                observation = self._parse_row(row, record_offset)
            except CsvLifetimeAdapterError as error:
                if failure is None:
                    failure = error
                record_offset += 1
                continue
            if failure is not None:
                record_offset += 1
                continue
            single = self._chunk((observation,), record_offset, sequence)
            if single.retained_payload_bytes > self.limits.chunk_bytes:
                raise CsvLifetimeAdapterError(
                    FailureCode.CHUNK_TOO_LARGE,
                    reason="record_too_large",
                    record_offset=record_offset,
                )
            candidate = tuple((*records, observation))
            chunk = self._chunk(candidate, start, sequence)
            if chunk.retained_payload_bytes > self.limits.chunk_bytes:
                yield self._chunk(tuple(records), start, sequence)
                start = record_offset
                sequence += 1
                records = [observation]
            else:
                records.append(observation)
            record_offset += 1
        if failure is not None:
            return failure
        if records:
            yield self._chunk(tuple(records), start, sequence)
        return None

    def _parse_row(self, row: list[str], record_offset: int) -> LifetimeObservation:
        if not row:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="blank_record",
                record_offset=record_offset,
            )
        if len(row) != 2:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="malformed_record",
                record_offset=record_offset,
            )
        time = self._parse_time(row[0], record_offset)
        if row[1] == "1":
            return ExactLifetime(time)
        if row[1] == "0":
            return RightCensoredLifetime(time)
        raise CsvLifetimeAdapterError(
            FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_event_token",
            record_offset=record_offset,
        )

    @staticmethod
    def _parse_time(value: str, record_offset: int) -> Decimal:
        if _TIME.fullmatch(value) is None:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="invalid_time",
                record_offset=record_offset,
            )
        decimal = Decimal(value)
        converted = float(decimal)
        if not isfinite(converted) or (decimal > 0 and converted == 0.0):
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="invalid_time",
                record_offset=record_offset,
            )
        return decimal

    def _chunk(
        self,
        observations: tuple[LifetimeObservation, ...],
        start: int,
        sequence: int,
    ) -> CsvLifetimeChunk:
        stop = start + len(observations)
        envelope = ChunkEnvelope(
            source_id=self.source_id.value,
            chunk_id=_chunk_id(self.source_id, start, stop),
            sequence_number=sequence,
            row_start=start,
            row_stop=stop,
            byte_size=1,
        )
        chunk = CsvLifetimeChunk(envelope, observations, 1)
        measured = retained_object_graph_bytes(chunk)
        chunk = replace(
            chunk,
            envelope=replace(envelope, byte_size=measured),
            retained_payload_bytes=measured,
        )
        final = retained_object_graph_bytes(chunk)
        if final != measured:
            chunk = replace(
                chunk,
                envelope=replace(chunk.envelope, byte_size=final),
                retained_payload_bytes=final,
            )
        return chunk


def _chunk_id(source_id: PublicSourceId, start: int, stop: int) -> str:
    digest = hashlib.sha256(f"{source_id.value}:{start}:{stop}".encode("ascii")).hexdigest()
    value = f"chk_{digest[:32]}"
    assert _CHUNK_ID.fullmatch(value) is not None
    return value


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
