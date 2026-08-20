"""DS-09 contracts for deterministic and strictly validated checkpoint resume."""

from __future__ import annotations

import hashlib
import sys
import unittest
from dataclasses import replace

from veridist.engine.resume import ResumeExpectation, resume_checkpoint

from veridist.engine.checkpoint import CheckpointRecord, InMemoryCheckpointStore
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.retry import PureReducer, apply_pure_update

SOURCE_ID = "dataset:resume-001"
SOURCE_SCHEMA = "source-v1"
SOURCE_REVISION = "private-etag-91"
PLAN_DIGEST = hashlib.sha256(b"resume-plan-v1").hexdigest()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class IntegerSumReducer(PureReducer[int]):
    reducer_id = "integer-sum-v1"
    accumulator_schema = "integer-v1"

    def __init__(self, *, fail_decode: bool = False) -> None:
        self.fail_decode = fail_decode
        self.decode_calls = 0

    def decode_state(self, state: bytes) -> int:
        self.decode_calls += 1
        if self.fail_decode:
            raise RuntimeError("private state must not escape")
        return int(state.decode("ascii"))

    def reduce(self, accumulator: int, payload: bytes) -> int:
        return accumulator + int(payload.decode("ascii"))

    def encode_state(self, accumulator: int) -> bytes:
        return str(accumulator).encode("ascii")


def checkpoint(
    *,
    format_version: int = 1,
    source_id: str = SOURCE_ID,
    source_schema: str = SOURCE_SCHEMA,
    source_revision: str = SOURCE_REVISION,
    reducer_id: str = "integer-sum-v1",
    accumulator_schema: str = "integer-v1",
    plan_digest: str = PLAN_DIGEST,
    cursor: int = 1,
    state: bytes = b"1",
) -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=format_version,
        source_id=source_id,
        source_schema=source_schema,
        source_revision=source_revision,
        reducer_id=reducer_id,
        accumulator_schema=accumulator_schema,
        plan_digest=plan_digest,
        cursor=cursor,
        committed_ranges=() if cursor == 0 else ((0, cursor),),
        generation=cursor,
        operation_token=None if cursor == 0 else f"chunk-{cursor}",
        operation_digest=None if cursor == 0 else sha256(f"operation-{cursor}".encode()),
        state=state,
    )


def expectation(**overrides: object) -> ResumeExpectation:
    values: dict[str, object] = {
        "format_version": 1,
        "source_id": SOURCE_ID,
        "source_schema": SOURCE_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "reducer_id": "integer-sum-v1",
        "accumulator_schema": "integer-v1",
        "plan_digest": PLAN_DIGEST,
        "cursor": 1,
    }
    values.update(overrides)
    return ResumeExpectation(**values)  # type: ignore[arg-type]


class CheckpointResumeContractTests(unittest.TestCase):
    def test_ds09_compatible_resume_matches_canonical_reduction(self) -> None:
        reducer = IntegerSumReducer()
        store = InMemoryCheckpointStore(checkpoint())
        resumed = resume_checkpoint(store=store, expected=expectation(), reducer=reducer)

        self.assertEqual(resumed.accumulator, 1)
        self.assertEqual(resumed.cursor, 1)
        self.assertEqual(resumed.committed_ranges, ((0, 1),))
        self.assertEqual(reducer.decode_calls, 1)

        current = store.read()
        for index, value in enumerate((2, 3), start=1):
            payload = str(value).encode("ascii")
            current = apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=payload,
                payload_sha256=sha256(payload),
                row_start=index,
                row_stop=index + 1,
                operation_token=f"chunk-{index + 1}",
                reducer=reducer,
            )

        canonical = sum((1, 2, 3))
        self.assertEqual(int(current.state.decode("ascii")), canonical)
        self.assertEqual(current.committed_ranges, ((0, 3),))

    def test_ds09_validation_order_is_fixed_and_precedes_decode_or_write(self) -> None:
        cases = (
            (
                replace(checkpoint(), checksum="corrupt"),
                expectation(format_version=2, source_id="other"),
                FailureCode.CHECKPOINT_CHECKSUM_MISMATCH,
            ),
            (
                checkpoint(format_version=2, source_id="other"),
                expectation(format_version=2, source_id="other"),
                FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED,
            ),
            (
                checkpoint(source_id="other", source_revision="other-revision"),
                expectation(),
                FailureCode.SOURCE_ID_MISMATCH,
            ),
            (
                checkpoint(source_schema="source-v2", source_revision="other-revision"),
                expectation(),
                FailureCode.SOURCE_SCHEMA_MISMATCH,
            ),
            (
                checkpoint(source_revision="other-revision", reducer_id="other"),
                expectation(),
                FailureCode.SOURCE_REVISION_MISMATCH,
            ),
            (
                checkpoint(reducer_id="other", plan_digest=sha256(b"other-plan")),
                expectation(),
                FailureCode.REDUCER_MISMATCH,
            ),
            (
                checkpoint(accumulator_schema="other", plan_digest=sha256(b"other-plan")),
                expectation(),
                FailureCode.ACCUMULATOR_SCHEMA_MISMATCH,
            ),
            (
                checkpoint(plan_digest=sha256(b"other-plan"), cursor=2),
                expectation(),
                FailureCode.PLAN_MISMATCH,
            ),
            (checkpoint(cursor=2), expectation(), FailureCode.RANGE_MISMATCH),
        )

        for record, expected, code in cases:
            reducer = IntegerSumReducer()
            store = InMemoryCheckpointStore(record)
            with self.subTest(code=code), self.assertRaises(EngineContractError) as caught:
                resume_checkpoint(store=store, expected=expected, reducer=reducer)
            self.assertIs(caught.exception.code, code)
            self.assertEqual(store.read_count, 1)
            self.assertEqual(store.write_count, 0)
            self.assertEqual(reducer.decode_calls, 0)

    def test_ds09_decode_is_last_and_failure_has_no_raw_exception_chain(self) -> None:
        reducer = IntegerSumReducer(fail_decode=True)
        store = InMemoryCheckpointStore(checkpoint())

        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(store=store, expected=expectation(), reducer=reducer)

        self.assertIs(caught.exception.code, FailureCode.REDUCER_FAILURE)
        self.assertEqual(caught.exception.context["exception_type"], "RuntimeError")
        self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(store.write_count, 0)

    def test_ds09_unknown_version_is_rejected_without_automigration(self) -> None:
        record = checkpoint(format_version=7)
        store = InMemoryCheckpointStore(record)
        reducer = IntegerSumReducer()

        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(
                store=store,
                expected=expectation(format_version=7),
                reducer=reducer,
            )

        self.assertIs(caught.exception.code, FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)
        self.assertIs(store.read(), record)
        self.assertEqual(store.write_count, 0)
        self.assertEqual(reducer.decode_calls, 0)

    def test_ds09_public_resume_metadata_never_contains_private_revision(self) -> None:
        resumed = resume_checkpoint(
            store=InMemoryCheckpointStore(checkpoint()),
            expected=expectation(),
            reducer=IntegerSumReducer(),
        )

        public = resumed.public_metadata
        self.assertFalse(hasattr(public, "source_revision"))
        self.assertNotIn(SOURCE_REVISION, repr(public))
        self.assertEqual(public.source_id, SOURCE_ID)
        self.assertEqual(public.source_schema, SOURCE_SCHEMA)

    def test_ds09_sequential_updates_keep_one_canonical_range_and_fixed_shape(self) -> None:
        reducer = IntegerSumReducer()
        store = InMemoryCheckpointStore(checkpoint(cursor=0, state=b"0"))
        initial = store.read()
        initial_orchestration_size = (
            sys.getsizeof(initial)
            + sys.getsizeof(initial.committed_ranges)
            + sys.getsizeof(initial.operation_token)
            + sys.getsizeof(initial.operation_digest)
        )

        for index in range(1, 2_049):
            payload = b"1"
            current = apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=payload,
                payload_sha256=sha256(payload),
                row_start=index - 1,
                row_stop=index,
                operation_token=f"chunk-{index}",
                reducer=reducer,
            )
            self.assertLessEqual(len(current.committed_ranges), 1)

        final = store.read()
        final_orchestration_size = (
            sys.getsizeof(final)
            + sys.getsizeof(final.committed_ranges)
            + sys.getsizeof(final.operation_token)
            + sys.getsizeof(final.operation_digest)
        )
        self.assertEqual(final.committed_ranges, ((0, 2_048),))
        self.assertEqual(int(final.state), 2_048)
        self.assertLessEqual(final_orchestration_size - initial_orchestration_size, 256)
        self.assertEqual(len(final.__slots__), len(initial.__slots__))


if __name__ == "__main__":
    unittest.main()
