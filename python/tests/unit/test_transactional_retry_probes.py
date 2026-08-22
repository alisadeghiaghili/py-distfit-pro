"""Focused probes for validation branches in the transactional retry kernel."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    InMemoryCheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.retry import IdempotentSink, PureReducer, SinkResult, apply_pure_update
from veridist.engine.retry import apply_sink_update as apply_sink

SOURCE_REVISION = "private-revision-7"
PLAN_DIGEST = hashlib.sha256(b"plan").hexdigest()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record(**overrides: object) -> CheckpointRecord:
    values: dict[str, object] = {
        "format_version": 1,
        "source_id": "dataset:probe",
        "source_schema": "source-v1",
        "source_revision": SOURCE_REVISION,
        "reducer_id": "bytes-v1",
        "accumulator_schema": "bytes-v1",
        "plan_digest": PLAN_DIGEST,
        "cursor": 0,
        "committed_ranges": (),
        "generation": 0,
        "operation_token": None,
        "operation_digest": None,
        "state": b"",
    }
    values.update(overrides)
    return CheckpointRecord.create(**values)  # type: ignore[arg-type]


class BytesReducer(PureReducer[bytes]):
    reducer_id = "bytes-v1"
    accumulator_schema = "bytes-v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def decode_state(self, state: bytes) -> bytes:
        self.calls += 1
        if self.fail:
            raise RuntimeError("decode failed")
        return state

    def reduce(self, accumulator: bytes, payload: bytes) -> bytes:
        return accumulator + payload

    def encode_state(self, accumulator: bytes) -> bytes:
        return accumulator


class LocalFailure(Exception):
    pass


class LocalFailureReducer(BytesReducer):
    def decode_state(self, state: bytes) -> bytes:
        del state
        raise LocalFailure("private diagnostic")


class InvalidSink(IdempotentSink):
    def __init__(self, result: object = SinkResult.APPLIED, *, fail: bool = False) -> None:
        self.result = result
        self.fail = fail

    def apply_once(
        self,
        operation_token: str,
        operation_digest: str,
        payload: bytes,
    ) -> SinkResult:
        del operation_token, operation_digest, payload
        if self.fail:
            raise OSError("sink unavailable")
        return self.result  # type: ignore[return-value]


class DivergedStore(InMemoryCheckpointStore):
    """Expose a competing generation after an ambiguous CAS response."""

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        competing = candidate.next_generation(
            cursor=candidate.cursor,
            committed_ranges=candidate.committed_ranges,
            operation_token="competitor",
            operation_digest=sha256(b"competitor"),
            state=candidate.state,
        )
        self._record = competing
        raise CheckpointCommitUncertain("ambiguous competing commit")


class AlwaysUncertainStore(InMemoryCheckpointStore):
    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        del expected_generation, candidate
        raise CheckpointCommitUncertain("unacknowledged")


class CheckpointValidationProbeTests(unittest.TestCase):
    def test_record_rejects_empty_identity_fields(self) -> None:
        for field in (
            "source_id",
            "source_schema",
            "source_revision",
            "reducer_id",
            "accumulator_schema",
            "plan_digest",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                record(**{field: " "})

    def test_record_rejects_invalid_version_position_token_and_ranges(self) -> None:
        invalid = (
            {"format_version": 0},
            {"format_version": True},
            {"cursor": -1},
            {"cursor": True},
            {"generation": -1},
            {"generation": True},
            {"operation_token": "token", "operation_digest": None},
            {"operation_token": " ", "operation_digest": "digest"},
            {"operation_token": "token", "operation_digest": " "},
            {"committed_ranges": ((-1, 0),)},
            {"committed_ranges": ((2, 1),)},
            {"committed_ranges": ((0, 0),)},
            {"committed_ranges": ((False, 1),), "cursor": 1},
            {"committed_ranges": ((0, True),), "cursor": 1},
            {"committed_ranges": ((0, 1),), "cursor": 2},
            {"committed_ranges": ((0, 1), (1, 2)), "cursor": 2},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                record(**overrides)

        with self.assertRaises(TypeError):
            record(state="not-bytes")

    def test_record_copies_supported_mutable_bytes_like_state(self) -> None:
        mutable = bytearray(b"state")
        from_bytearray = record(state=mutable)
        from_memoryview = record(state=memoryview(b"view"))
        mutable[:] = b"other"
        self.assertEqual(from_bytearray.state, b"state")
        self.assertEqual(from_memoryview.state, b"view")

    def test_store_rejects_invalid_candidate_generation_and_checksum(self) -> None:
        store = InMemoryCheckpointStore(record())
        wrong_generation = record(generation=3)
        with self.assertRaises(EngineContractError) as generation_error:
            store.compare_and_swap(0, wrong_generation)
        self.assertIs(generation_error.exception.code, FailureCode.CHECKPOINT_CONFLICT)

        candidate = record(generation=1)
        corrupt = replace(candidate, checksum="0" * 64)
        with self.assertRaises(EngineContractError) as checksum_error:
            store.compare_and_swap(0, corrupt)
        self.assertIs(
            checksum_error.exception.code,
            FailureCode.CHECKPOINT_CHECKSUM_MISMATCH,
        )
        self.assertEqual(store.write_count, 0)


class RetryValidationProbeTests(unittest.TestCase):
    def _pure(
        self,
        store: InMemoryCheckpointStore,
        reducer: PureReducer[bytes],
        **overrides: object,
    ) -> CheckpointRecord:
        values: dict[str, object] = {
            "store": store,
            "source_revision": SOURCE_REVISION,
            "payload": b"x",
            "payload_sha256": sha256(b"x"),
            "row_start": 0,
            "row_stop": 1,
            "operation_token": "token-1",
            "reducer": reducer,
        }
        values.update(overrides)
        return apply_pure_update(**values)  # type: ignore[arg-type]

    def test_request_validation_rejects_bad_range_token_and_attempts_before_read(self) -> None:
        invalid = (
            {"row_start": -1},
            {"row_start": 2, "row_stop": 1},
            {"operation_token": " "},
            {"max_attempts": 0},
            {"row_start": False},
            {"row_stop": True},
            {"max_attempts": True},
        )
        for overrides in invalid:
            store = InMemoryCheckpointStore(record())
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._pure(store, BytesReducer(), **overrides)
            self.assertEqual(store.read_count, 0)

    def test_corrupt_checkpoint_reducer_and_schema_fail_before_decode(self) -> None:
        cases = (
            (
                replace(record(), checksum="bad"),
                BytesReducer(),
                FailureCode.CHECKPOINT_CHECKSUM_MISMATCH,
            ),
            (record(reducer_id="other"), BytesReducer(), FailureCode.REDUCER_MISMATCH),
            (
                record(accumulator_schema="other"),
                BytesReducer(),
                FailureCode.ACCUMULATOR_SCHEMA_MISMATCH,
            ),
        )
        for initial, reducer, code in cases:
            store = InMemoryCheckpointStore(initial)
            with self.subTest(code=code), self.assertRaises(EngineContractError) as caught:
                self._pure(store, reducer)
            self.assertIs(caught.exception.code, code)
            self.assertEqual(reducer.calls, 0)
            self.assertEqual(store.write_count, 0)

    def test_range_mismatch_and_reducer_failure_are_typed(self) -> None:
        store = InMemoryCheckpointStore(record(cursor=1, committed_ranges=((0, 1),)))
        with self.assertRaises(EngineContractError) as range_error:
            self._pure(store, BytesReducer())
        self.assertIs(range_error.exception.code, FailureCode.RANGE_MISMATCH)

        failing = BytesReducer(fail=True)
        with self.assertRaises(EngineContractError) as reducer_error:
            self._pure(InMemoryCheckpointStore(record()), failing)
        self.assertIs(reducer_error.exception.code, FailureCode.REDUCER_FAILURE)
        self.assertEqual(reducer_error.exception.context["failure_type"], "RuntimeError")
        self.assertIsNone(reducer_error.exception.__cause__)

        with self.assertRaises(EngineContractError) as local_error:
            self._pure(InMemoryCheckpointStore(record()), LocalFailureReducer())
        self.assertEqual(local_error.exception.context["failure_type"], "Exception")

    def test_same_operation_is_returned_without_second_reduce_or_write(self) -> None:
        store = InMemoryCheckpointStore(record())
        reducer = BytesReducer()
        first = self._pure(store, reducer)
        second = self._pure(store, reducer)
        self.assertIs(second, first)
        self.assertEqual(reducer.calls, 1)
        self.assertEqual(store.write_count, 1)

    def test_empty_first_range_keeps_canonical_ranges_empty(self) -> None:
        committed = self._pure(
            InMemoryCheckpointStore(record()),
            BytesReducer(),
            payload=b"",
            payload_sha256=sha256(b""),
            row_stop=0,
        )
        self.assertEqual(committed.committed_ranges, ())

    def test_ambiguous_competing_generation_is_a_conflict(self) -> None:
        with self.assertRaises(EngineContractError) as caught:
            self._pure(DivergedStore(record()), BytesReducer())
        self.assertIs(caught.exception.code, FailureCode.CHECKPOINT_CONFLICT)

    def test_sink_admission_failure_and_invalid_results_precede_checkpoint_write(self) -> None:
        store = InMemoryCheckpointStore(record())
        with self.assertRaises(EngineContractError) as admission_error:
            apply_sink(
                store=store,
                sink=object(),  # type: ignore[arg-type]
                source_revision=SOURCE_REVISION,
                payload=b"x",
                payload_sha256=sha256(b"x"),
                row_start=0,
                row_stop=1,
                operation_token="token-1",
            )
        self.assertIs(admission_error.exception.code, FailureCode.RETRY_NOT_ADMISSIBLE)
        self.assertEqual(store.read_count, 0)

        for sink in (InvalidSink("unexpected"), InvalidSink(fail=True)):
            fresh = InMemoryCheckpointStore(record())
            with self.subTest(sink=sink), self.assertRaises(EngineContractError) as caught:
                apply_sink(
                    store=fresh,
                    sink=sink,
                    source_revision=SOURCE_REVISION,
                    payload=b"x",
                    payload_sha256=sha256(b"x"),
                    row_start=0,
                    row_stop=1,
                    operation_token="token-1",
                )
            self.assertIs(caught.exception.code, FailureCode.SINK_FAILURE)
            self.assertEqual(fresh.write_count, 0)
            self.assertIsNone(caught.exception.__cause__)

    def test_same_sink_operation_returns_checkpoint_without_second_effect(self) -> None:
        store = InMemoryCheckpointStore(record())
        sink = InvalidSink()
        first = apply_sink(
            store=store,
            sink=sink,
            source_revision=SOURCE_REVISION,
            payload=b"x",
            payload_sha256=sha256(b"x"),
            row_start=0,
            row_stop=1,
            operation_token="token-1",
        )
        second = apply_sink(
            store=store,
            sink=sink,
            source_revision=SOURCE_REVISION,
            payload=b"x",
            payload_sha256=sha256(b"x"),
            row_start=0,
            row_stop=1,
            operation_token="token-1",
        )
        self.assertIs(second, first)
        self.assertEqual(store.write_count, 1)

    def test_sink_retry_exhaustion_warns_that_effect_may_have_applied(self) -> None:
        store = AlwaysUncertainStore(record())
        with self.assertRaises(EngineContractError) as caught:
            apply_sink(
                store=store,
                sink=InvalidSink(),
                source_revision=SOURCE_REVISION,
                payload=b"x",
                payload_sha256=sha256(b"x"),
                row_start=0,
                row_stop=1,
                operation_token="token-1",
                max_attempts=1,
            )
        self.assertIs(caught.exception.code, FailureCode.RETRY_EXHAUSTED)
        self.assertEqual(caught.exception.context["effect_status"], "applied_or_unknown")


if __name__ == "__main__":
    unittest.main()
