"""Immutable checkpoint records and a non-durable in-memory CAS test double."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Protocol

from veridist.engine.errors import EngineContractError, FailureCode

CHECKPOINT_FORMAT_VERSION = 1


def _require_text(label: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _canonical_payload(record: CheckpointRecord) -> bytes:
    value = {
        "accumulator_schema": record.accumulator_schema,
        "committed_ranges": [list(item) for item in record.committed_ranges],
        "cursor": record.cursor,
        "format_version": record.format_version,
        "generation": record.generation,
        "operation_digest": record.operation_digest,
        "operation_token": record.operation_token,
        "plan_digest": record.plan_digest,
        "reducer_id": record.reducer_id,
        "source_id": record.source_id,
        "source_revision": record.source_revision,
        "source_schema": record.source_schema,
        "state": base64.b64encode(record.state).decode("ascii"),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _checksum(record: CheckpointRecord) -> str:
    return hashlib.sha256(_canonical_payload(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Canonical private resume state protected by a SHA-256 integrity checksum."""

    format_version: int
    source_id: str
    source_schema: str
    source_revision: str
    reducer_id: str
    accumulator_schema: str
    plan_digest: str
    cursor: int
    committed_ranges: tuple[tuple[int, int], ...]
    generation: int
    operation_token: str | None
    operation_digest: str | None
    state: bytes
    checksum: str

    @classmethod
    def create(
        cls,
        *,
        format_version: int,
        source_id: str,
        source_schema: str,
        source_revision: str,
        reducer_id: str,
        accumulator_schema: str,
        plan_digest: str,
        cursor: int,
        committed_ranges: tuple[tuple[int, int], ...],
        generation: int,
        operation_token: str | None,
        operation_digest: str | None,
        state: bytes | bytearray | memoryview,
    ) -> CheckpointRecord:
        """Build a validated record and derive its canonical checksum."""

        for label, value in (
            ("source_id", source_id),
            ("source_schema", source_schema),
            ("source_revision", source_revision),
            ("reducer_id", reducer_id),
            ("accumulator_schema", accumulator_schema),
            ("plan_digest", plan_digest),
        ):
            _require_text(label, value)
        if isinstance(format_version, bool) or format_version < 1:
            raise ValueError("format_version must be positive")
        if (
            isinstance(cursor, bool)
            or isinstance(generation, bool)
            or cursor < 0
            or generation < 0
        ):
            raise ValueError("cursor and generation must be non-negative")
        if (operation_token is None) != (operation_digest is None):
            raise ValueError("operation token and digest must both be present or absent")
        if operation_token is not None:
            _require_text("operation_token", operation_token)
            assert operation_digest is not None
            _require_text("operation_digest", operation_digest)
        if not isinstance(state, bytes | bytearray | memoryview):
            raise TypeError("state must be bytes-like")
        for start, stop in committed_ranges:
            if (
                isinstance(start, bool)
                or isinstance(stop, bool)
                or start < 0
                or stop <= start
            ):
                raise ValueError("committed ranges must be non-empty non-negative intervals")
        canonical_ranges = () if cursor == 0 else ((0, cursor),)
        if committed_ranges != canonical_ranges:
            raise ValueError("committed ranges must be the canonical contiguous cursor prefix")
        provisional = cls(
            format_version=format_version,
            source_id=source_id,
            source_schema=source_schema,
            source_revision=source_revision,
            reducer_id=reducer_id,
            accumulator_schema=accumulator_schema,
            plan_digest=plan_digest,
            cursor=cursor,
            committed_ranges=committed_ranges,
            generation=generation,
            operation_token=operation_token,
            operation_digest=operation_digest,
            state=bytes(state),
            checksum="",
        )
        return replace(provisional, checksum=_checksum(provisional))

    def has_valid_checksum(self) -> bool:
        """Return whether the stored checksum matches the canonical record."""

        return self.checksum == _checksum(self)

    def next_generation(
        self,
        *,
        cursor: int,
        committed_ranges: tuple[tuple[int, int], ...],
        operation_token: str,
        operation_digest: str,
        state: bytes,
    ) -> CheckpointRecord:
        """Return the immutable candidate for the next atomic CAS transition."""

        return self.create(
            format_version=self.format_version,
            source_id=self.source_id,
            source_schema=self.source_schema,
            source_revision=self.source_revision,
            reducer_id=self.reducer_id,
            accumulator_schema=self.accumulator_schema,
            plan_digest=self.plan_digest,
            cursor=cursor,
            committed_ranges=committed_ranges,
            generation=self.generation + 1,
            operation_token=operation_token,
            operation_digest=operation_digest,
            state=state,
        )


class CheckpointCommitUncertain(RuntimeError):
    """The store cannot say whether a requested CAS transition committed."""


class CheckpointStore(Protocol):
    """Minimal atomic checkpoint protocol; implementations define durability."""

    def read(self) -> CheckpointRecord: ...

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord: ...


class InMemoryCheckpointStore:
    """Deterministic test double with no durability or cross-process guarantees."""

    __slots__ = ("_lock", "_record", "_read_count", "_write_count")

    def __init__(self, initial: CheckpointRecord) -> None:
        self._record = initial
        self._lock = Lock()
        self._read_count = 0
        self._write_count = 0

    @property
    def read_count(self) -> int:
        return self._read_count

    @property
    def write_count(self) -> int:
        return self._write_count

    def read(self) -> CheckpointRecord:
        with self._lock:
            self._read_count += 1
            return self._record

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        with self._lock:
            if self._record.generation != expected_generation:
                raise EngineContractError(
                    FailureCode.CHECKPOINT_CONFLICT,
                    {
                        "actual_generation": self._record.generation,
                        "expected_generation": expected_generation,
                    },
                )
            if candidate.generation != expected_generation + 1:
                raise EngineContractError(
                    FailureCode.CHECKPOINT_CONFLICT,
                    {
                        "candidate_generation": candidate.generation,
                        "expected_generation": expected_generation + 1,
                    },
                )
            if not candidate.has_valid_checksum():
                raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
            self._record = candidate
            self._write_count += 1
            return self._record


class FileCheckpointStore:
    """Durable single-file checkpoint store with atomic replacement."""

    __slots__ = ("_path", "_lock")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _serialize(record: CheckpointRecord) -> str:
        payload = json.loads(_canonical_payload(record).decode("utf-8"))
        payload["checksum"] = record.checksum
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _read_unlocked(self) -> CheckpointRecord:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            record = CheckpointRecord(
                format_version=payload["format_version"], source_id=payload["source_id"],
                source_schema=payload["source_schema"], source_revision=payload["source_revision"],
                reducer_id=payload["reducer_id"], accumulator_schema=payload["accumulator_schema"],
                plan_digest=payload["plan_digest"], cursor=payload["cursor"],
                committed_ranges=tuple(tuple(item) for item in payload["committed_ranges"]),
                generation=payload["generation"], operation_token=payload["operation_token"],
                operation_digest=payload["operation_digest"],
                state=base64.b64decode(payload["state"], validate=True),
                checksum=payload["checksum"],
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH) from None
        if not record.has_valid_checksum():
            raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
        return record

    @classmethod
    def create(cls, path: str | os.PathLike[str], initial: CheckpointRecord) -> FileCheckpointStore:
        store = cls(path)
        store._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = store._path.with_suffix(store._path.suffix + ".tmp")
        temporary.write_text(store._serialize(initial), encoding="utf-8")
        os.replace(temporary, store._path)
        return store

    def read(self) -> CheckpointRecord:
        with self._lock:
            return self._read_unlocked()

    def compare_and_swap(
        self, expected_generation: int, candidate: CheckpointRecord
    ) -> CheckpointRecord:
        with self._lock:
            current = self._read_unlocked()
            if (
                current.generation != expected_generation
                or candidate.generation != expected_generation + 1
            ):
                raise EngineContractError(FailureCode.CHECKPOINT_CONFLICT)
            if not candidate.has_valid_checksum():
                raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(self._serialize(candidate), encoding="utf-8")
            os.replace(temporary, self._path)
            return candidate
