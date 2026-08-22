"""DS-08 contracts for admitted retry protocols and atomic checkpoint progress."""

from __future__ import annotations

import hashlib
import unittest

from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    InMemoryCheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.retry import (
    IdempotentSink,
    PureReducer,
    SinkResult,
    apply_pure_update,
    apply_sink_update,
)

SOURCE_ID = "dataset:retry-001"
SOURCE_REVISION = "private-revision-7"
PLAN_DIGEST = hashlib.sha256(b"canonical-plan").hexdigest()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def initial_record(*, state: bytes = b"0") -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id=SOURCE_ID,
        source_schema="source-v1",
        source_revision=SOURCE_REVISION,
        reducer_id="integer-sum-v1",
        accumulator_schema="integer-v1",
        plan_digest=PLAN_DIGEST,
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=state,
    )


class IntegerSumReducer(PureReducer[int]):
    reducer_id = "integer-sum-v1"
    accumulator_schema = "integer-v1"

    def __init__(self) -> None:
        self.decode_calls = 0
        self.reduce_calls = 0
        self.encode_calls = 0

    def decode_state(self, state: bytes) -> int:
        self.decode_calls += 1
        return int(state.decode("ascii"))

    def reduce(self, accumulator: int, payload: bytes) -> int:
        self.reduce_calls += 1
        return accumulator + int(payload.decode("ascii"))

    def encode_state(self, accumulator: int) -> bytes:
        self.encode_calls += 1
        return str(accumulator).encode("ascii")


class CommitThenTimeoutStore(InMemoryCheckpointStore):
    """Inject one ambiguous response after a successful atomic commit."""

    def __init__(self, initial: CheckpointRecord) -> None:
        super().__init__(initial)
        self._inject = True

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        committed = super().compare_and_swap(expected_generation, candidate)
        if self._inject:
            self._inject = False
            raise CheckpointCommitUncertain("commit response was lost")
        return committed


class AlwaysUncertainStore(InMemoryCheckpointStore):
    def __init__(self, initial: CheckpointRecord) -> None:
        super().__init__(initial)
        self.attempts = 0

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        del expected_generation, candidate
        self.attempts += 1
        raise CheckpointCommitUncertain("no commit acknowledgement")


class RecordingSink(IdempotentSink):
    def __init__(self, *, already_applied: bool = False) -> None:
        self.calls: list[tuple[str, str, bytes]] = []
        self.applied_tokens: set[str] = {"chunk-1"} if already_applied else set()

    def apply_once(
        self,
        operation_token: str,
        operation_digest: str,
        payload: bytes,
    ) -> SinkResult:
        self.calls.append((operation_token, operation_digest, payload))
        if operation_token in self.applied_tokens:
            return SinkResult.ALREADY_APPLIED
        self.applied_tokens.add(operation_token)
        return SinkResult.APPLIED


class TransactionalRetryContractTests(unittest.TestCase):
    """Retry is admitted only for the two protocols accepted in ADR-0015."""

    def test_ds08_arbitrary_callback_is_rejected_before_read_or_invocation(self) -> None:
        store = InMemoryCheckpointStore(initial_record())
        callback_calls = 0

        def arbitrary_callback(state: bytes, payload: bytes) -> bytes:
            nonlocal callback_calls
            callback_calls += 1
            return state + payload

        with self.assertRaises(EngineContractError) as caught:
            apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=b"2",
                payload_sha256=digest(b"2"),
                row_start=0,
                row_stop=1,
                operation_token="chunk-1",
                reducer=arbitrary_callback,  # type: ignore[arg-type]
            )

        self.assertIs(caught.exception.code, FailureCode.RETRY_NOT_ADMISSIBLE)
        self.assertEqual(store.read_count, 0)
        self.assertEqual(callback_calls, 0)

    def test_ds08_payload_integrity_failure_precedes_read_decode_and_write(self) -> None:
        store = InMemoryCheckpointStore(initial_record())
        reducer = IntegerSumReducer()

        with self.assertRaises(EngineContractError) as caught:
            apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=b"2",
                payload_sha256=digest(b"different"),
                row_start=0,
                row_stop=1,
                operation_token="chunk-1",
                reducer=reducer,
            )

        self.assertIs(caught.exception.code, FailureCode.PAYLOAD_CHECKSUM_MISMATCH)
        self.assertEqual(
            dict(caught.exception.context),
            {"integrity_check": "sha256", "match": False},
        )
        self.assertNotIn(digest(b"2"), repr(caught.exception.context))
        self.assertNotIn(digest(b"different"), repr(caught.exception.context))
        self.assertEqual(store.read_count, 0)
        self.assertEqual(store.write_count, 0)
        self.assertEqual(reducer.decode_calls, 0)

    def test_ds08_source_revision_mismatch_precedes_decode_reduce_and_write(self) -> None:
        store = InMemoryCheckpointStore(initial_record())
        reducer = IntegerSumReducer()

        with self.assertRaises(EngineContractError) as caught:
            apply_pure_update(
                store=store,
                source_revision="mutated-revision",
                payload=b"2",
                payload_sha256=digest(b"2"),
                row_start=0,
                row_stop=1,
                operation_token="chunk-1",
                reducer=reducer,
            )

        self.assertIs(caught.exception.code, FailureCode.SOURCE_REVISION_MISMATCH)
        self.assertEqual(store.write_count, 0)
        self.assertEqual(reducer.decode_calls, 0)
        self.assertEqual(reducer.reduce_calls, 0)

    def test_ds08_commit_then_timeout_reloads_without_double_transition(self) -> None:
        store = CommitThenTimeoutStore(initial_record(state=b"1"))
        reducer = IntegerSumReducer()

        committed = apply_pure_update(
            store=store,
            source_revision=SOURCE_REVISION,
            payload=b"2",
            payload_sha256=digest(b"2"),
            row_start=0,
            row_stop=1,
            operation_token="chunk-1",
            reducer=reducer,
            max_attempts=2,
        )

        self.assertEqual(committed.state, b"3")
        self.assertEqual(committed.generation, 1)
        self.assertEqual(committed.cursor, 1)
        self.assertEqual(committed.committed_ranges, ((0, 1),))
        self.assertEqual(store.write_count, 1)
        self.assertEqual(reducer.reduce_calls, 1)

    def test_ds08_retry_exhaustion_is_distinct_and_commits_nothing(self) -> None:
        store = AlwaysUncertainStore(initial_record())
        reducer = IntegerSumReducer()

        with self.assertRaises(EngineContractError) as caught:
            apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=b"2",
                payload_sha256=digest(b"2"),
                row_start=0,
                row_stop=1,
                operation_token="chunk-1",
                reducer=reducer,
                max_attempts=2,
            )

        self.assertIs(caught.exception.code, FailureCode.RETRY_EXHAUSTED)
        self.assertEqual(store.attempts, 2)
        self.assertEqual(store.write_count, 0)
        self.assertEqual(store.read().generation, 0)

    def test_ds08_already_applied_sink_reconciles_lagging_checkpoint(self) -> None:
        store = InMemoryCheckpointStore(initial_record())
        sink = RecordingSink(already_applied=True)

        committed = apply_sink_update(
            store=store,
            sink=sink,
            source_revision=SOURCE_REVISION,
            payload=b"external-effect",
            payload_sha256=digest(b"external-effect"),
            row_start=0,
            row_stop=1,
            operation_token="chunk-1",
        )

        self.assertEqual(len(sink.calls), 1)
        self.assertEqual(committed.generation, 1)
        self.assertEqual(committed.cursor, 1)
        self.assertEqual(store.write_count, 1)

    def test_ds08_same_token_with_different_digest_is_a_conflict(self) -> None:
        store = InMemoryCheckpointStore(initial_record())
        reducer = IntegerSumReducer()
        apply_pure_update(
            store=store,
            source_revision=SOURCE_REVISION,
            payload=b"2",
            payload_sha256=digest(b"2"),
            row_start=0,
            row_stop=1,
            operation_token="chunk-1",
            reducer=reducer,
        )

        with self.assertRaises(EngineContractError) as caught:
            apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=b"9",
                payload_sha256=digest(b"9"),
                row_start=0,
                row_stop=1,
                operation_token="chunk-1",
                reducer=reducer,
            )

        self.assertIs(caught.exception.code, FailureCode.OPERATION_DIGEST_CONFLICT)
        self.assertEqual(store.read().state, b"2")
        self.assertEqual(store.write_count, 1)

    def test_ds08_store_cas_rejects_stale_generation_without_mutation(self) -> None:
        store = InMemoryCheckpointStore(initial_record())
        candidate = initial_record(state=b"2").next_generation(
            cursor=1,
            committed_ranges=((0, 1),),
            operation_token="chunk-1",
            operation_digest=digest(b"operation-1"),
            state=b"2",
        )
        committed = store.compare_and_swap(0, candidate)

        with self.assertRaises(EngineContractError) as caught:
            store.compare_and_swap(0, candidate.next_generation(
                cursor=2,
                committed_ranges=((0, 2),),
                operation_token="chunk-2",
                operation_digest=digest(b"operation-2"),
                state=b"4",
            ))

        self.assertIs(caught.exception.code, FailureCode.CHECKPOINT_CONFLICT)
        self.assertIs(store.read(), committed)
        self.assertEqual(store.write_count, 1)


if __name__ == "__main__":
    unittest.main()
