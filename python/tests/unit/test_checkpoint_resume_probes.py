"""Focused probes for checkpoint-resume preflight validation."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from veridist.engine import checkpoint as checkpoint_module
from veridist.engine.checkpoint import CheckpointRecord, InMemoryCheckpointStore
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.resume import ResumeExpectation, resume_checkpoint
from veridist.engine.retry import PureReducer

PLAN_DIGEST = hashlib.sha256(b"plan").hexdigest()


class BytesReducer(PureReducer[bytes]):
    reducer_id = "bytes-v1"
    accumulator_schema = "bytes-v1"

    def __init__(self) -> None:
        self.decode_calls = 0

    def decode_state(self, state: bytes) -> bytes:
        self.decode_calls += 1
        return state

    def reduce(self, accumulator: bytes, payload: bytes) -> bytes:
        return accumulator + payload

    def encode_state(self, accumulator: bytes) -> bytes:
        return accumulator


def record() -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id="dataset:probe",
        source_schema="source-v1",
        source_revision="private-revision",
        reducer_id="bytes-v1",
        accumulator_schema="bytes-v1",
        plan_digest=PLAN_DIGEST,
        cursor=1,
        committed_ranges=((0, 1),),
        generation=1,
        operation_token="chunk-1",
        operation_digest=hashlib.sha256(b"operation").hexdigest(),
        state=b"state",
    )


def expectation(**overrides: object) -> ResumeExpectation:
    values: dict[str, object] = {
        "format_version": 1,
        "source_id": "dataset:probe",
        "source_schema": "source-v1",
        "source_revision": "private-revision",
        "reducer_id": "bytes-v1",
        "accumulator_schema": "bytes-v1",
        "plan_digest": PLAN_DIGEST,
        "cursor": 1,
    }
    values.update(overrides)
    return ResumeExpectation(**values)  # type: ignore[arg-type]


class ResumeValidationProbeTests(unittest.TestCase):
    def test_expectation_rejects_bool_bounds_and_blank_identifiers(self) -> None:
        for overrides in (
            {"format_version": 0},
            {"format_version": True},
            {"cursor": -1},
            {"cursor": True},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                expectation(**overrides)

        for field in (
            "source_id",
            "source_schema",
            "reducer_id",
            "accumulator_schema",
            "plan_digest",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                expectation(**{field: " "})

    def test_expected_unknown_format_is_rejected_after_checksum(self) -> None:
        reducer = BytesReducer()
        store = InMemoryCheckpointStore(record())
        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(store=store, expected=expectation(format_version=2), reducer=reducer)
        self.assertIs(caught.exception.code, FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)
        self.assertEqual(reducer.decode_calls, 0)
        self.assertEqual(store.write_count, 0)

    def test_noncanonical_range_with_valid_checksum_is_rejected_before_decode(self) -> None:
        valid = record()
        malformed = replace(valid, committed_ranges=((0, 1), (1, 1)), checksum="")
        malformed = replace(malformed, checksum=checkpoint_module._checksum(malformed))
        store = InMemoryCheckpointStore(malformed)

        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(store=store, expected=expectation(), reducer=BytesReducer())
        self.assertIs(caught.exception.code, FailureCode.RANGE_MISMATCH)
        self.assertEqual(store.write_count, 0)

    def test_arbitrary_decoder_is_rejected_before_checkpoint_read(self) -> None:
        store = InMemoryCheckpointStore(record())
        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(
                store=store,
                expected=expectation(),
                reducer=object(),  # type: ignore[arg-type]
            )
        self.assertIs(caught.exception.code, FailureCode.RETRY_NOT_ADMISSIBLE)
        self.assertEqual(store.read_count, 0)


if __name__ == "__main__":
    unittest.main()
