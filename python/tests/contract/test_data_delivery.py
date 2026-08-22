"""DS-04 through DS-06 contracts for delivery metadata and bounded buffering."""

from __future__ import annotations

import threading
import time
import unittest
from sys import getsizeof

from veridist.engine.data_source import Replayability
from veridist.engine.delivery import (
    AdapterCapabilities,
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
    DeliveryValidator,
    OrderingGuarantee,
)


def chunk(
    chunk_id: str,
    start: int,
    stop: int,
    *,
    sequence_number: int = 0,
    byte_size: int = 1,
) -> ChunkEnvelope:
    return ChunkEnvelope(
        source_id="dataset:delivery-001",
        chunk_id=chunk_id,
        sequence_number=sequence_number,
        row_start=start,
        row_stop=stop,
        byte_size=byte_size,
    )


def wait_for_waiting_producer(buffer: BoundedChunkBuffer, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if buffer.waiting_producers == 1:
            return
        time.sleep(0.001)
    raise AssertionError("producer did not enter backpressure before the deadline")


def buffered(envelope: ChunkEnvelope) -> BufferedChunk:
    return BufferedChunk(envelope=envelope, payload=object())


class AdapterAndIdentityContractTests(unittest.TestCase):
    """DS-04: declarations and source-offset row identities are stable."""

    def test_ds04_seven_adapter_kinds_declare_capabilities_without_imports(self) -> None:
        replayability = {
            AdapterKind.CSV: Replayability.REPLAYABLE,
            AdapterKind.PARQUET: Replayability.REPLAYABLE,
            AdapterKind.ARROW: Replayability.REPLAYABLE,
            AdapterKind.PANDAS: Replayability.REPLAYABLE,
            AdapterKind.POLARS: Replayability.REPLAYABLE,
            AdapterKind.DASK: Replayability.REPLAYABLE,
            AdapterKind.DATABASE: Replayability.CHECKPOINT_REPLAYABLE,
        }

        declarations = tuple(
            AdapterCapabilities(
                kind=kind,
                replayability=mode,
                ordering=OrderingGuarantee.STABLE_ROW_OFFSETS,
                stable_offsets=True,
            )
            for kind, mode in replayability.items()
        )

        self.assertEqual({item.kind for item in declarations}, set(AdapterKind))
        self.assertTrue(all(item.stable_offsets for item in declarations))
        self.assertFalse(hasattr(declarations[0], "__dict__"))
        with self.assertRaises((AttributeError, TypeError)):
            declarations[0].stable_offsets = False  # type: ignore[misc]

    def test_ds04_row_identity_survives_boundary_changes(self) -> None:
        first_partition = (chunk("a", 0, 2), chunk("b", 2, 5))
        second_partition = (chunk("x", 0, 1), chunk("y", 1, 4), chunk("z", 4, 5))

        first_identities = [
            envelope.row_identity(index)
            for envelope in first_partition
            for index in range(envelope.row_count)
        ]
        second_identities = [
            envelope.row_identity(index)
            for envelope in second_partition
            for index in range(envelope.row_count)
        ]

        self.assertEqual(first_identities, second_identities)
        self.assertEqual(first_identities[3], ("dataset:delivery-001", 3))

    def test_ds04_envelope_rejects_invalid_identity_offsets_and_sizes(self) -> None:
        invalid = (
            {
                "source_id": "",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": -1,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 2,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": -1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": -1,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            },
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ChunkEnvelope(**arguments)

        envelope = chunk("valid", 3, 5)
        with self.assertRaises(IndexError):
            envelope.row_identity(-1)
        with self.assertRaises(IndexError):
            envelope.row_identity(2)


class DeliveryValidationContractTests(unittest.TestCase):
    """DS-05: invalid delivery modes are distinct and never mutate state."""

    def test_ds05_contiguous_ragged_and_empty_chunks_are_valid(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")

        for envelope in (
            chunk("empty-start", 0, 0, sequence_number=0, byte_size=0),
            chunk("first", 0, 1, sequence_number=1),
            chunk("empty-middle", 1, 1, sequence_number=2, byte_size=0),
            chunk("ragged", 1, 4, sequence_number=3),
        ):
            validator.accept(envelope)

        self.assertEqual(validator.next_offset, 4)
        self.assertEqual(validator.next_sequence, 4)
        self.assertEqual(validator.accepted_rows, 4)
        self.assertEqual(validator.accepted_chunks, 4)

    def test_ds05_validator_and_finish_reject_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DeliveryValidator("")
        with self.assertRaises(ValueError):
            DeliveryValidator("dataset:delivery-001", initial_offset=-1)
        with self.assertRaises(ValueError):
            DeliveryValidator("dataset:delivery-001", initial_sequence=-1)

        validator = DeliveryValidator("dataset:delivery-001", initial_offset=2)
        with self.assertRaises(ValueError):
            validator.finish(expected_row_stop=-1)
        with self.assertRaises(ValueError):
            validator.finish(expected_row_stop=2, expected_chunk_count=-1)
        with self.assertRaises(DeliveryContractError) as caught:
            validator.finish(expected_row_stop=1)
        self.assertEqual(caught.exception.code, "OUT_OF_ORDER_CHUNK")

        resumed = DeliveryValidator("dataset:delivery-001", initial_sequence=2)
        with self.assertRaises(DeliveryContractError) as excess_chunks:
            resumed.finish(expected_row_stop=0, expected_chunk_count=1)
        self.assertEqual(excess_chunks.exception.code, "OUT_OF_ORDER_CHUNK")

    def test_ds05_duplicate_chunk_has_stable_code_and_no_double_count(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        accepted = chunk("same", 0, 2, sequence_number=0)
        validator.accept(accepted)

        with self.assertRaises(DeliveryContractError) as caught:
            validator.accept(accepted)

        self.assertEqual(caught.exception.code, "DUPLICATE_CHUNK")
        self.assertEqual(validator.next_offset, 2)
        self.assertEqual(validator.accepted_rows, 2)
        self.assertEqual(validator.accepted_chunks, 1)

    def test_ds05_missing_and_out_of_order_codes_are_distinct_and_recoverable(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("first", 0, 2, sequence_number=0))

        with self.assertRaises(DeliveryContractError) as missing:
            validator.accept(chunk("gap", 4, 5, sequence_number=1))
        self.assertEqual(missing.exception.code, "MISSING_CHUNK")
        self.assertEqual(missing.exception.context["expected_offset"], 2)

        with self.assertRaises(DeliveryContractError) as out_of_order:
            validator.accept(chunk("late", 1, 2, sequence_number=1))
        self.assertEqual(out_of_order.exception.code, "OUT_OF_ORDER_CHUNK")
        self.assertEqual(out_of_order.exception.context["expected_offset"], 2)

        with self.assertRaises(DeliveryContractError) as skipped_sequence:
            validator.accept(chunk("skipped", 2, 4, sequence_number=2))
        self.assertEqual(skipped_sequence.exception.code, "OUT_OF_ORDER_CHUNK")

        validator.accept(chunk("recovery", 2, 4, sequence_number=1))
        self.assertEqual(validator.next_offset, 4)
        self.assertEqual(validator.next_sequence, 2)
        self.assertEqual(validator.accepted_rows, 4)
        self.assertEqual(validator.accepted_chunks, 2)

    def test_ds05_finish_detects_tail_loss_with_typed_missing_code(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("first", 0, 2, sequence_number=0))

        with self.assertRaises(DeliveryContractError) as missing:
            validator.finish(expected_row_stop=4)

        self.assertEqual(missing.exception.code, "MISSING_CHUNK")
        self.assertEqual(missing.exception.context["expected_offset"], 2)
        self.assertEqual(missing.exception.context["expected_row_stop"], 4)
        validator.accept(chunk("tail", 2, 4, sequence_number=1))
        validator.finish(expected_row_stop=4, expected_chunk_count=2)

    def test_ds05_finish_detects_missing_empty_tail_by_chunk_count(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("data", 0, 2, sequence_number=0))

        with self.assertRaises(DeliveryContractError) as missing:
            validator.finish(expected_row_stop=2, expected_chunk_count=2)

        self.assertEqual(missing.exception.code, "MISSING_CHUNK")
        self.assertEqual(missing.exception.context["expected_sequence"], 1)
        self.assertEqual(missing.exception.context["expected_chunk_count"], 2)

    def test_ds05_validator_state_is_constant_size_in_chunk_count(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        initial_state_bytes = getsizeof(validator) + sum(
            getsizeof(getattr(validator, slot)) for slot in validator.__slots__
        )

        for sequence_number in range(2_048):
            validator.accept(
                chunk(
                    f"empty-{sequence_number}",
                    0,
                    0,
                    sequence_number=sequence_number,
                    byte_size=0,
                )
            )

        self.assertEqual(validator.next_sequence, 2_048)
        self.assertEqual(validator.accepted_chunks, 2_048)
        final_state_bytes = getsizeof(validator) + sum(
            getsizeof(getattr(validator, slot)) for slot in validator.__slots__
        )
        self.assertEqual(final_state_bytes, initial_state_bytes)

    def test_ds05_source_mismatch_is_typed_and_preflight_only(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        other_source = ChunkEnvelope("dataset:other", "other", 0, 0, 1, 1)

        with self.assertRaises(DeliveryContractError) as caught:
            validator.accept(other_source)

        self.assertEqual(caught.exception.code, "SOURCE_MISMATCH")
        self.assertEqual(validator.next_offset, 0)
        self.assertEqual(validator.accepted_chunks, 0)


class BoundedBufferContractTests(unittest.TestCase):
    """DS-06: byte budgets, real backpressure and cancellation are observable."""

    def test_ds06_constructor_and_chunk_limits_are_hard_failures(self) -> None:
        for arguments in (
            {"chunk_bytes": 0, "max_inflight_bytes": 1},
            {"chunk_bytes": 2, "max_inflight_bytes": 1},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                BoundedChunkBuffer(**arguments)

        buffer = BoundedChunkBuffer(chunk_bytes=3, max_inflight_bytes=6)
        with self.assertRaises(DeliveryContractError) as unaccounted:
            buffer.put(buffered(chunk("unaccounted", 0, 0, byte_size=0)), timeout=0.1)
        self.assertEqual(unaccounted.exception.code, "INVALID_RETAINED_BYTES")

        with self.assertRaises(DeliveryContractError) as caught:
            buffer.put(buffered(chunk("oversized", 0, 1, byte_size=4)), timeout=0.1)
        self.assertEqual(caught.exception.code, "CHUNK_TOO_LARGE")
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.queued_chunks, 0)

    def test_ds06_producer_blocks_until_get_releases_budget(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        second = buffered(chunk("second", 1, 2, byte_size=4))
        buffer.put(first)
        outcome: list[str] = []

        def produce() -> None:
            buffer.put(second, timeout=1.0)
            outcome.append("accepted")

        producer = threading.Thread(target=produce, name="blocked-producer")
        producer.start()
        wait_for_waiting_producer(buffer)
        self.assertTrue(producer.is_alive())
        self.assertEqual(buffer.inflight_bytes, 4)
        self.assertEqual(buffer.peak_inflight_bytes, 4)
        self.assertLessEqual(buffer.peak_inflight_bytes, buffer.max_inflight_bytes)

        self.assertEqual(buffer.get(timeout=0.2), first)
        producer.join(timeout=1.0)

        self.assertFalse(producer.is_alive())
        self.assertEqual(outcome, ["accepted"])
        self.assertEqual(buffer.inflight_bytes, 4)
        self.assertEqual(buffer.get(timeout=0.2), second)
        self.assertEqual(buffer.inflight_bytes, 0)

    def test_ds06_cancel_releases_queued_resources_exactly_once(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        released: list[str] = []
        first = BufferedChunk(
            envelope=chunk("first", 0, 1, byte_size=4),
            payload=object(),
            release_callback=lambda: released.append("first"),
        )
        buffer.put(first)

        buffer.cancel()
        buffer.cancel()
        first.release()

        self.assertEqual(released, ["first"])
        self.assertTrue(first.released)
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.queued_chunks, 0)

    def test_ds06_full_buffer_put_timeout_is_bounded(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        buffer.put(first)

        with self.assertRaises(DeliveryContractError) as caught:
            buffer.put(buffered(chunk("timeout", 1, 2, byte_size=4)), timeout=0.01)
        self.assertEqual(caught.exception.code, "BUFFER_TIMEOUT")
        self.assertEqual(caught.exception.context, {"operation": "put"})

        self.assertEqual(buffer.inflight_bytes, 4)
        self.assertEqual(buffer.queued_chunks, 1)
        self.assertEqual(buffer.get(timeout=0.1), first)

    def test_ds06_cancel_wakes_waiter_and_stops_future_put_get(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(buffered(chunk("first", 0, 1, byte_size=4)))
        outcome: list[str] = []

        def produce() -> None:
            try:
                buffer.put(buffered(chunk("blocked", 1, 2, byte_size=4)), timeout=1.0)
            except DeliveryContractError as error:
                outcome.append(error.code)

        producer = threading.Thread(target=produce, name="cancelled-producer")
        producer.start()
        wait_for_waiting_producer(buffer)
        buffer.cancel()
        producer.join(timeout=1.0)

        self.assertFalse(producer.is_alive())
        self.assertEqual(outcome, ["CANCELLED"])
        self.assertTrue(buffer.cancelled)
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.queued_chunks, 0)
        with self.assertRaises(DeliveryContractError) as put_cancelled:
            buffer.put(buffered(chunk("future", 2, 3)), timeout=0.1)
        with self.assertRaises(DeliveryContractError) as get_cancelled:
            buffer.get(timeout=0.1)
        self.assertEqual(put_cancelled.exception.code, "CANCELLED")
        self.assertEqual(get_cancelled.exception.code, "CANCELLED")

    def test_ds06_cancel_prevents_new_reads_and_releases_rejected_read(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(buffered(chunk("first", 0, 1, byte_size=4)))
        read_count = 0
        released: list[str] = []
        outcome: list[str] = []

        def read_next() -> BufferedChunk:
            nonlocal read_count
            read_count += 1
            return BufferedChunk(
                envelope=chunk("read", 1, 2, byte_size=4),
                payload=object(),
                release_callback=lambda: released.append("read"),
            )

        def produce() -> None:
            try:
                buffer.read_and_put(read_next, timeout=1.0)
            except DeliveryContractError as error:
                outcome.append(error.code)

        producer = threading.Thread(target=produce, name="instrumented-reader")
        producer.start()
        wait_for_waiting_producer(buffer)
        self.assertEqual(read_count, 1)
        buffer.cancel()
        producer.join(timeout=1.0)

        self.assertFalse(producer.is_alive())
        self.assertEqual(outcome, ["CANCELLED"])
        self.assertEqual(released, ["read"])
        with self.assertRaises(DeliveryContractError) as caught:
            buffer.read_and_put(read_next, timeout=0.1)
        self.assertEqual(caught.exception.code, "CANCELLED")
        self.assertEqual(read_count, 1)

    def test_ds06_cancel_does_not_wait_for_external_read_callback(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        read_started = threading.Event()
        allow_read_return = threading.Event()
        cancel_returned = threading.Event()
        released: list[str] = []
        outcome: list[str] = []

        def blocking_read() -> BufferedChunk:
            read_started.set()
            if not allow_read_return.wait(1.0):
                raise AssertionError("test did not release the read callback")
            return BufferedChunk(
                envelope=chunk("read-after-cancel", 0, 1, byte_size=4),
                payload=object(),
                release_callback=lambda: released.append("read-after-cancel"),
            )

        def produce() -> None:
            try:
                buffer.read_and_put(blocking_read, timeout=1.0)
            except DeliveryContractError as error:
                outcome.append(error.code)

        def cancel() -> None:
            buffer.cancel()
            cancel_returned.set()

        producer = threading.Thread(target=produce, name="blocked-external-reader")
        canceller = threading.Thread(target=cancel, name="nonblocking-canceller")
        producer.start()
        self.assertTrue(read_started.wait(1.0))
        canceller.start()
        returned_while_read_blocked = cancel_returned.wait(0.2)
        allow_read_return.set()
        canceller.join(timeout=1.0)
        producer.join(timeout=1.0)

        self.assertTrue(returned_while_read_blocked)
        self.assertFalse(canceller.is_alive())
        self.assertFalse(producer.is_alive())
        self.assertEqual(outcome, ["CANCELLED"])
        self.assertEqual(released, ["read-after-cancel"])

    def test_ds06_empty_queue_timeout_is_bounded(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        with self.assertRaises(DeliveryContractError) as caught:
            buffer.get(timeout=0.01)
        self.assertEqual(caught.exception.code, "BUFFER_TIMEOUT")
        self.assertEqual(caught.exception.context, {"operation": "get"})


if __name__ == "__main__":
    unittest.main()
