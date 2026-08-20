"""Typed, localization-independent failures for engine contracts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class FailureCode(StrEnum):
    """Stable machine-readable failure codes exposed by the execution engine."""

    RETRY_NOT_ADMISSIBLE = "RETRY_NOT_ADMISSIBLE"
    PAYLOAD_CHECKSUM_MISMATCH = "PAYLOAD_CHECKSUM_MISMATCH"
    SOURCE_REVISION_MISMATCH = "SOURCE_REVISION_MISMATCH"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    CHECKPOINT_CONFLICT = "CHECKPOINT_CONFLICT"
    OPERATION_DIGEST_CONFLICT = "OPERATION_DIGEST_CONFLICT"
    CHECKPOINT_CHECKSUM_MISMATCH = "CHECKPOINT_CHECKSUM_MISMATCH"
    CHECKPOINT_FORMAT_UNSUPPORTED = "CHECKPOINT_FORMAT_UNSUPPORTED"
    SOURCE_ID_MISMATCH = "SOURCE_ID_MISMATCH"
    SOURCE_SCHEMA_MISMATCH = "SOURCE_SCHEMA_MISMATCH"
    REDUCER_MISMATCH = "REDUCER_MISMATCH"
    ACCUMULATOR_SCHEMA_MISMATCH = "ACCUMULATOR_SCHEMA_MISMATCH"
    PLAN_MISMATCH = "PLAN_MISMATCH"
    RANGE_MISMATCH = "RANGE_MISMATCH"
    REDUCER_FAILURE = "REDUCER_FAILURE"
    SINK_FAILURE = "SINK_FAILURE"


class EngineContractError(Exception):
    """A typed engine failure with immutable diagnostic context."""

    def __init__(self, code: FailureCode, context: Mapping[str, object] | None = None) -> None:
        super().__init__(code.value)
        self.code = code
        self.context: Mapping[str, object] = MappingProxyType(dict(context or {}))
