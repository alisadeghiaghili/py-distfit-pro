"""Typed, localization-independent failures for engine contracts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, StrEnum
from math import isfinite
from types import MappingProxyType
from typing import cast


class FailureCode(StrEnum):
    """Stable machine-readable failure codes exposed by the execution engine."""

    ACCUMULATOR_SCHEMA_MISMATCH = "ACCUMULATOR_SCHEMA_MISMATCH"
    BUFFER_TIMEOUT = "BUFFER_TIMEOUT"
    CANCELLED = "CANCELLED"
    CHECKPOINT_CHECKSUM_MISMATCH = "CHECKPOINT_CHECKSUM_MISMATCH"
    CHECKPOINT_CONFLICT = "CHECKPOINT_CONFLICT"
    CHECKPOINT_FORMAT_UNSUPPORTED = "CHECKPOINT_FORMAT_UNSUPPORTED"
    CHECKPOINT_REQUIRED = "CHECKPOINT_REQUIRED"
    CHECKPOINT_SCHEMA_MISMATCH = "CHECKPOINT_SCHEMA_MISMATCH"
    CHECKPOINT_SOURCE_ID_MISMATCH = "CHECKPOINT_SOURCE_ID_MISMATCH"
    CHUNK_TOO_LARGE = "CHUNK_TOO_LARGE"
    DUPLICATE_CHUNK = "DUPLICATE_CHUNK"
    INVALID_RETAINED_BYTES = "INVALID_RETAINED_BYTES"
    MISSING_CHUNK = "MISSING_CHUNK"
    OPERATION_DIGEST_CONFLICT = "OPERATION_DIGEST_CONFLICT"
    OUT_OF_ORDER_CHUNK = "OUT_OF_ORDER_CHUNK"
    PASS_BUDGET_EXCEEDED = "PASS_BUDGET_EXCEEDED"
    PAYLOAD_CHECKSUM_MISMATCH = "PAYLOAD_CHECKSUM_MISMATCH"
    PLAN_MISMATCH = "PLAN_MISMATCH"
    RANGE_MISMATCH = "RANGE_MISMATCH"
    REDUCER_FAILURE = "REDUCER_FAILURE"
    REDUCER_MISMATCH = "REDUCER_MISMATCH"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    RETRY_NOT_ADMISSIBLE = "RETRY_NOT_ADMISSIBLE"
    SINK_FAILURE = "SINK_FAILURE"
    SOURCE_ID_MISMATCH = "SOURCE_ID_MISMATCH"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    SOURCE_REVISION_MISMATCH = "SOURCE_REVISION_MISMATCH"
    SOURCE_REVISION_UNAVAILABLE = "SOURCE_REVISION_UNAVAILABLE"
    SOURCE_SCHEMA_MISMATCH = "SOURCE_SCHEMA_MISMATCH"
    SPOOL_REQUIRED = "SPOOL_REQUIRED"


_SENSITIVE_CONTEXT_KEY_PARTS = frozenset(
    {
        "checksum",
        "credential",
        "digest",
        "dsn",
        "exception",
        "message",
        "password",
        "path",
        "payload",
        "query",
        "revision",
        "token",
        "traceback",
        "uri",
    }
)


def _freeze_context_value(value: object) -> object:
    if isinstance(value, Enum):
        return _freeze_context_value(value.value)
    if type(value) is float:
        if not isfinite(value):
            raise TypeError("failure context floats must be finite")
        return value
    if type(value) in {str, int, bool, type(None)}:
        return value
    if type(value) is tuple:
        return tuple(_freeze_context_value(item) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("failure context mapping keys must be strings")
            key_parts = frozenset(key.casefold().split("_"))
            if key_parts & _SENSITIVE_CONTEXT_KEY_PARTS:
                raise TypeError("failure context contains a sensitive key")
            frozen[key] = _freeze_context_value(item)
        return MappingProxyType(frozen)
    raise TypeError("failure context contains an unsafe value type")


class EngineContractError(Exception):
    """A typed engine failure with immutable diagnostic context."""

    def __init__(self, code: FailureCode, context: Mapping[str, object] | None = None) -> None:
        if not isinstance(code, FailureCode):
            raise TypeError("code must be a FailureCode")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("failure context must be a mapping")
        super().__init__(code.value)
        self.code = code
        self.context = cast(Mapping[str, object], _freeze_context_value(context or {}))

    def __str__(self) -> str:
        return self.code.value

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value})"


def safe_exception_type(exc: Exception) -> str:
    """Return an allowlist-safe type label without exposing exception text."""

    return type(exc).__name__ if type(exc).__module__ == "builtins" else "Exception"
