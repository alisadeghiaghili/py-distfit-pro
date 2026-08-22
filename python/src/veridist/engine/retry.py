"""Sequential logical-once retry protocols accepted by ADR-0015."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Generic, TypeVar

from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    CheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode, safe_exception_type

Accumulator = TypeVar("Accumulator")


class PureReducer(ABC, Generic[Accumulator]):
    """Explicit admission surface for a deterministic, side-effect-free reducer."""

    reducer_id: str
    accumulator_schema: str

    @abstractmethod
    def decode_state(self, state: bytes) -> Accumulator:
        """Decode immutable checkpoint bytes."""

    @abstractmethod
    def reduce(self, accumulator: Accumulator, payload: bytes) -> Accumulator:
        """Produce the next accumulator without mutating external state."""

    @abstractmethod
    def encode_state(self, accumulator: Accumulator) -> bytes:
        """Encode the next accumulator as immutable bytes."""


class SinkResult(StrEnum):
    """A durable idempotent sink's result for one stable operation token."""

    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"


class EffectStatus(StrEnum):
    """Safe effect status reported when checkpoint reconciliation is exhausted."""

    APPLIED_OR_UNKNOWN = "applied_or_unknown"


class IdempotentSink(ABC):
    """Explicit external-effect protocol; checkpoint and sink are not atomic together."""

    @abstractmethod
    def apply_once(
        self,
        operation_token: str,
        operation_digest: str,
        payload: bytes,
    ) -> SinkResult:
        """Durably apply or recognize one operation under a stable token."""


def _validate_request(
    *,
    payload: bytes,
    payload_sha256: str,
    row_start: int,
    row_stop: int,
    operation_token: str,
    max_attempts: int,
) -> None:
    if (
        isinstance(row_start, bool)
        or isinstance(row_stop, bool)
        or row_start < 0
        or row_stop < row_start
    ):
        raise ValueError("row range must be an ordered non-negative interval")
    if not operation_token.strip():
        raise ValueError("operation_token must be non-empty")
    if isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != payload_sha256:
        raise EngineContractError(
            FailureCode.PAYLOAD_CHECKSUM_MISMATCH,
            {"integrity_check": "sha256", "match": False},
        )


def _operation_digest(
    record: CheckpointRecord,
    *,
    payload_sha256: str,
    row_start: int,
    row_stop: int,
    operation_token: str,
) -> str:
    value = {
        "operation_token": operation_token,
        "payload_sha256": payload_sha256,
        "plan_digest": record.plan_digest,
        "reducer_id": record.reducer_id,
        "row_start": row_start,
        "row_stop": row_stop,
        "source_id": record.source_id,
        "source_revision": record.source_revision,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_loaded_record(record: CheckpointRecord, source_revision: str) -> None:
    if not record.has_valid_checksum():
        raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
    if record.source_revision != source_revision:
        raise EngineContractError(FailureCode.SOURCE_REVISION_MISMATCH)


def _validate_update_position(
    record: CheckpointRecord,
    *,
    operation_token: str,
    operation_digest: str,
    row_start: int,
    row_stop: int,
) -> bool:
    if record.operation_token == operation_token:
        if record.operation_digest != operation_digest:
            raise EngineContractError(FailureCode.OPERATION_DIGEST_CONFLICT)
        return True
    if row_start != record.cursor or row_stop < row_start:
        raise EngineContractError(
            FailureCode.RANGE_MISMATCH,
            {"checkpoint_cursor": record.cursor, "row_start": row_start, "row_stop": row_stop},
        )
    return False


def _committed_ranges(row_start: int, row_stop: int) -> tuple[tuple[int, int], ...]:
    return () if row_stop == 0 else ((0, row_stop),)


def _commit_with_reconciliation(
    store: CheckpointStore,
    base: CheckpointRecord,
    candidate: CheckpointRecord,
    *,
    max_attempts: int,
    effect_status: EffectStatus | None = None,
) -> CheckpointRecord:
    for _attempt in range(max_attempts):
        try:
            return store.compare_and_swap(base.generation, candidate)
        except CheckpointCommitUncertain:
            observed = store.read()
            if (
                observed.generation == candidate.generation
                and observed.operation_token == candidate.operation_token
                and observed.operation_digest == candidate.operation_digest
            ):
                return observed
            if observed.generation != base.generation:
                raise EngineContractError(FailureCode.CHECKPOINT_CONFLICT) from None
    context: dict[str, object] = {
        "attempts": max_attempts,
        "operation_id_present": candidate.operation_token is not None,
    }
    if effect_status is not None:
        context["effect_status"] = effect_status.value
    raise EngineContractError(FailureCode.RETRY_EXHAUSTED, context)


def apply_pure_update(
    *,
    store: CheckpointStore,
    source_revision: str,
    payload: bytes,
    payload_sha256: str,
    row_start: int,
    row_stop: int,
    operation_token: str,
    reducer: PureReducer[Accumulator],
    max_attempts: int = 3,
) -> CheckpointRecord:
    """Commit one pure reducer transition with checkpoint advancement in one CAS."""

    if not isinstance(reducer, PureReducer):
        raise EngineContractError(FailureCode.RETRY_NOT_ADMISSIBLE)
    _validate_request(
        payload=payload,
        payload_sha256=payload_sha256,
        row_start=row_start,
        row_stop=row_stop,
        operation_token=operation_token,
        max_attempts=max_attempts,
    )
    base = store.read()
    _validate_loaded_record(base, source_revision)
    if base.reducer_id != reducer.reducer_id:
        raise EngineContractError(FailureCode.REDUCER_MISMATCH)
    if base.accumulator_schema != reducer.accumulator_schema:
        raise EngineContractError(FailureCode.ACCUMULATOR_SCHEMA_MISMATCH)
    operation_digest = _operation_digest(
        base,
        payload_sha256=payload_sha256,
        row_start=row_start,
        row_stop=row_stop,
        operation_token=operation_token,
    )
    if _validate_update_position(
        base,
        operation_token=operation_token,
        operation_digest=operation_digest,
        row_start=row_start,
        row_stop=row_stop,
    ):
        return base
    try:
        decoded = reducer.decode_state(base.state)
        updated = reducer.reduce(decoded, payload)
        encoded = bytes(reducer.encode_state(updated))
    except Exception as exc:
        raise EngineContractError(
            FailureCode.REDUCER_FAILURE,
            {"failure_type": safe_exception_type(exc)},
        ) from None
    candidate = base.next_generation(
        cursor=row_stop,
        committed_ranges=_committed_ranges(row_start, row_stop),
        operation_token=operation_token,
        operation_digest=operation_digest,
        state=encoded,
    )
    return _commit_with_reconciliation(store, base, candidate, max_attempts=max_attempts)


def apply_sink_update(
    *,
    store: CheckpointStore,
    sink: IdempotentSink,
    source_revision: str,
    payload: bytes,
    payload_sha256: str,
    row_start: int,
    row_stop: int,
    operation_token: str,
    max_attempts: int = 3,
) -> CheckpointRecord:
    """Apply one idempotent effect, then reconcile its non-atomic checkpoint."""

    if not isinstance(sink, IdempotentSink):
        raise EngineContractError(FailureCode.RETRY_NOT_ADMISSIBLE)
    _validate_request(
        payload=payload,
        payload_sha256=payload_sha256,
        row_start=row_start,
        row_stop=row_stop,
        operation_token=operation_token,
        max_attempts=max_attempts,
    )
    base = store.read()
    _validate_loaded_record(base, source_revision)
    operation_digest = _operation_digest(
        base,
        payload_sha256=payload_sha256,
        row_start=row_start,
        row_stop=row_stop,
        operation_token=operation_token,
    )
    if _validate_update_position(
        base,
        operation_token=operation_token,
        operation_digest=operation_digest,
        row_start=row_start,
        row_stop=row_stop,
    ):
        return base
    try:
        result = sink.apply_once(operation_token, operation_digest, payload)
    except Exception as exc:
        raise EngineContractError(
            FailureCode.SINK_FAILURE,
            {"failure_type": safe_exception_type(exc)},
        ) from None
    if result not in (SinkResult.APPLIED, SinkResult.ALREADY_APPLIED):
        raise EngineContractError(FailureCode.SINK_FAILURE)
    candidate = base.next_generation(
        cursor=row_stop,
        committed_ranges=_committed_ranges(row_start, row_stop),
        operation_token=operation_token,
        operation_digest=operation_digest,
        state=base.state,
    )
    return _commit_with_reconciliation(
        store,
        base,
        candidate,
        max_attempts=max_attempts,
        effect_status=EffectStatus.APPLIED_OR_UNKNOWN,
    )
