"""DS-10 contracts for the public engine failure surface."""

from __future__ import annotations

import unittest
from enum import StrEnum
from math import inf, nan
from pathlib import Path
from types import MappingProxyType

from veridist.engine.data_source import DataSourceCapabilityError
from veridist.engine.delivery import (
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.pass_budget import PassBudgetError

EXPECTED_FAILURE_CODES = {
    "ACCUMULATOR_SCHEMA_MISMATCH",
    "BUFFER_TIMEOUT",
    "CANCELLED",
    "CHECKPOINT_CHECKSUM_MISMATCH",
    "CHECKPOINT_CONFLICT",
    "CHECKPOINT_FORMAT_UNSUPPORTED",
    "CHECKPOINT_REQUIRED",
    "CHECKPOINT_SCHEMA_MISMATCH",
    "CHECKPOINT_SOURCE_ID_MISMATCH",
    "CHUNK_TOO_LARGE",
    "DUPLICATE_CHUNK",
    "INVALID_RETAINED_BYTES",
    "MISSING_CHUNK",
    "OPERATION_DIGEST_CONFLICT",
    "OUT_OF_ORDER_CHUNK",
    "PASS_BUDGET_EXCEEDED",
    "PAYLOAD_CHECKSUM_MISMATCH",
    "PLAN_MISMATCH",
    "RANGE_MISMATCH",
    "REDUCER_FAILURE",
    "REDUCER_MISMATCH",
    "RETRY_EXHAUSTED",
    "RETRY_NOT_ADMISSIBLE",
    "SINK_FAILURE",
    "SOURCE_ID_MISMATCH",
    "SOURCE_MISMATCH",
    "SOURCE_REVISION_MISMATCH",
    "SOURCE_REVISION_UNAVAILABLE",
    "SOURCE_SCHEMA_MISMATCH",
    "SPOOL_REQUIRED",
}


class ContextLabel(StrEnum):
    VALUE = "safe-label"


class TypedFailureSurfaceTests(unittest.TestCase):
    def test_ds10_failure_code_registry_is_complete_and_stable(self) -> None:
        self.assertEqual({code.name for code in FailureCode}, EXPECTED_FAILURE_CODES)
        for code in FailureCode:
            with self.subTest(code=code):
                self.assertEqual(code.value, code.name)

    def test_ds10_public_failure_classes_share_one_typed_constructor(self) -> None:
        cases = (
            (EngineContractError, FailureCode.RANGE_MISMATCH),
            (DataSourceCapabilityError, FailureCode.SPOOL_REQUIRED),
            (DeliveryContractError, FailureCode.MISSING_CHUNK),
            (PassBudgetError, FailureCode.PASS_BUDGET_EXCEEDED),
        )
        for error_type, code in cases:
            with self.subTest(error_type=error_type.__name__):
                error = error_type(code, {"attempt": 1})
                self.assertIsInstance(error, EngineContractError)
                self.assertIs(error.code, code)
                with self.assertRaises(TypeError):
                    error_type(code.value, {})  # type: ignore[arg-type]

    def test_ds10_context_is_recursively_immutable_and_enum_normalized(self) -> None:
        source = {
            "nested": {"labels": (ContextLabel.VALUE,)},
            "complete": False,
            "estimate": 1.5,
        }
        error = EngineContractError(FailureCode.RANGE_MISMATCH, source)

        self.assertIsInstance(error.context, MappingProxyType)
        nested = error.context["nested"]
        self.assertIsInstance(nested, MappingProxyType)
        self.assertEqual(nested["labels"], ("safe-label",))
        self.assertEqual(error.context["estimate"], 1.5)

        source["nested"]["labels"] = ("changed",)  # type: ignore[index]
        self.assertEqual(nested["labels"], ("safe-label",))
        with self.assertRaises(TypeError):
            error.context["new"] = "value"  # type: ignore[index]
        with self.assertRaises(TypeError):
            nested["new"] = "value"  # type: ignore[index]

    def test_ds10_context_rejects_unsafe_keys_and_values(self) -> None:
        unsafe_contexts: tuple[object, ...] = (
            ["not a mapping"],
            {1: "non-string key"},
            {"payload": b"raw"},
            {"file": Path("private/source.csv")},
            {"items": [1, 2]},
            {"items": {1, 2}},
            {"exception": RuntimeError("private payload")},
            {"object": object()},
            {"estimate": nan},
            {"estimate": inf},
            {"estimate": -inf},
        )
        for context in unsafe_contexts:
            with self.subTest(context_type=type(context).__name__):
                with self.assertRaises(TypeError):
                    EngineContractError(FailureCode.RANGE_MISMATCH, context)  # type: ignore[arg-type]

    def test_ds10_sensitive_context_keys_are_rejected_even_for_strings(self) -> None:
        sensitive_keys = (
            "payload",
            "path",
            "uri",
            "query",
            "dsn",
            "credential",
            "password",
            "token",
            "source_revision",
            "checksum",
            "digest",
            "exception",
            "message",
            "traceback",
        )
        for key in sensitive_keys:
            with self.subTest(key=key), self.assertRaises(TypeError):
                EngineContractError(FailureCode.RANGE_MISMATCH, {key: "private-value"})

    def test_ds10_text_surfaces_code_but_never_context(self) -> None:
        sentinel = "C:/private/source.csv?credential=secret"
        error = EngineContractError(
            FailureCode.RANGE_MISMATCH,
            {"expected": sentinel, "actual": (0, 3)},
        )

        self.assertEqual(str(error), "RANGE_MISMATCH")
        self.assertEqual(repr(error), "EngineContractError(code=RANGE_MISMATCH)")
        self.assertNotIn(sentinel, str(error))
        self.assertNotIn(sentinel, repr(error))

    def test_ds10_buffer_timeouts_use_one_typed_distinct_code(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)

        def buffered(chunk_id: str, sequence: int) -> BufferedChunk:
            return BufferedChunk(
                envelope=ChunkEnvelope(
                    source_id="source",
                    chunk_id=chunk_id,
                    sequence_number=sequence,
                    row_start=sequence,
                    row_stop=sequence + 1,
                    byte_size=4,
                ),
                payload=None,
            )

        buffer.put(buffered("first", 0))
        with self.assertRaises(DeliveryContractError) as put_error:
            buffer.put(buffered("second", 1), timeout=0.001)
        self.assertIs(put_error.exception.code, FailureCode.BUFFER_TIMEOUT)
        self.assertEqual(put_error.exception.context, {"operation": "put"})

        buffer.get()

        with self.assertRaises(DeliveryContractError) as caught:
            buffer.get(timeout=0.001)

        self.assertIs(caught.exception.code, FailureCode.BUFFER_TIMEOUT)
        self.assertNotEqual(caught.exception.code, FailureCode.CANCELLED)
        self.assertEqual(caught.exception.context, {"operation": "get"})


if __name__ == "__main__":
    unittest.main()
