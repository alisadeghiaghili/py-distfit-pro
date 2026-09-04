"""Public stream-source and active-lease delivery contracts."""

from __future__ import annotations

import threading
import time
import unittest

from veridist.domain.lifetimes import ExactLifetime
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.delivery import BoundedChunkBuffer, BufferedChunk, ChunkEnvelope
from veridist.engine.errors import FailureCode
from veridist.engine.streaming import IterableDataSource, StreamSourceError
from veridist.families import ExponentialFitSuccess, fit_exponential_chunks
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks


def metadata(replayability: Replayability = Replayability.SINGLE_PASS) -> DataSourceMetadata:
    return DataSourceMetadata(
        source_id="stream-test",
        schema_version="1",
        provenance_schema_version="1",
        replayability=replayability,
        redaction_reason="test",
    )


def chunk(identifier: str, sequence: int, callback=None) -> BufferedChunk:
    return BufferedChunk(
        envelope=ChunkEnvelope("stream-test", identifier, sequence, sequence, sequence + 1, 4),
        payload=(sequence,),
        release_callback=callback,
    )


class StreamSourceContractTests(unittest.TestCase):
    def test_generic_source_is_consumed_by_current_streaming_reducers(self) -> None:
        likelihood_source = IterableDataSource(((0.0, 1.0),), metadata())
        likelihood = reduce_log_likelihood_chunks(
            FamilyId.NORMAL, likelihood_source, mu=0.0, sigma=1.0
        )
        self.assertIsInstance(likelihood, LogLikelihoodSuccess)
        self.assertEqual(likelihood.observation_count, 2)

        exponential_source = IterableDataSource(((ExactLifetime(2.0),),), metadata())
        exponential = fit_exponential_chunks(exponential_source)
        self.assertIsInstance(exponential, ExponentialFitSuccess)
        self.assertEqual(exponential.rate, 0.5)

    def test_single_pass_source_fails_with_typed_failure_on_second_acquisition(self) -> None:
        source = IterableDataSource(((0.0,),), metadata())
        self.assertEqual(tuple(source.iter_chunks()), ((0.0,),))
        with self.assertRaises(StreamSourceError) as caught:
            tuple(source.iter_chunks())
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)

    def test_replayable_source_requires_a_factory_and_can_be_acquired_twice(self) -> None:
        with self.assertRaises(ValueError):
            IterableDataSource(((0.0,),), metadata(Replayability.REPLAYABLE))
        source = IterableDataSource(lambda: iter(((0.0,),)), metadata(Replayability.REPLAYABLE))
        self.assertEqual(tuple(source.iter_chunks()), ((0.0,),))
        self.assertEqual(tuple(source.iter_chunks()), ((0.0,),))


class ActiveLeaseBufferTests(unittest.TestCase):
    def test_active_lease_remains_charged_until_release(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(chunk("first", 0))
        received = buffer.get()
        self.assertEqual(buffer.inflight_bytes, 4)
        self.assertEqual(buffer.queued_chunks, 0)
        received.release()
        self.assertEqual(buffer.inflight_bytes, 0)

    def test_producer_unblocks_only_after_active_lease_release(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(chunk("first", 0))
        received = buffer.get()
        completed = threading.Event()
        producer = threading.Thread(
            target=lambda: (buffer.put(chunk("second", 1)), completed.set())
        )
        producer.start()
        deadline = time.monotonic() + 1.0
        while buffer.waiting_producers == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(buffer.waiting_producers, 1)
        self.assertFalse(completed.is_set())
        received.release()
        producer.join(1.0)
        self.assertTrue(completed.is_set())
        buffer.get().release()
        self.assertEqual(buffer.inflight_bytes, 0)

    def test_cancel_releases_every_queued_item_then_reraises_first_callback_error(self) -> None:
        calls: list[str] = []

        def bad() -> None:
            calls.append("bad")
            raise RuntimeError("release failed")

        def good() -> None:
            calls.append("good")

        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)
        first = chunk("first", 0, bad)
        second = chunk("second", 1, good)
        buffer.put(first)
        buffer.put(second)
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            buffer.cancel()
        self.assertEqual(calls, ["bad", "good"])
        self.assertTrue(first.released)
        self.assertTrue(second.released)
        self.assertTrue(buffer.cancelled)
        self.assertEqual(buffer.inflight_bytes, 0)
