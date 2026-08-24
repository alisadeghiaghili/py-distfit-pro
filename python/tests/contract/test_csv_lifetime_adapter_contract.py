"""CSV-01--CSV-06 contracts for the first real lifetime source adapter."""

from __future__ import annotations

import tracemalloc
import unittest
from io import BytesIO
from pathlib import Path

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeAdapterError,
    CsvLifetimeErrorCode,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    fit_exponential_csv,
)

from veridist.engine.data_source import Replayability
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitSuccess

SCHEMA = CsvLifetimeSchema(time_column="time", event_observed_column="event_observed")
SOURCE_ID = "source_csv_lifetime_fixture_01"


class CountingBinarySource:
    """A test-only replayable source seam that reveals illegal rereads."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._identity = "revision-0"
        self.open_count = 0

    def open_binary(self, path: Path) -> BytesIO:
        del path
        self.open_count += 1
        return BytesIO(self._payload)

    def identity(self, path: Path) -> str:
        del path
        return self._identity


class MutatingBinarySource(CountingBinarySource):
    """A source whose public test seam proves the best-effort mutation check."""

    def open_binary(self, path: Path) -> BytesIO:
        handle = super().open_binary(path)
        self._identity = "private revision changed"
        return handle


class CsvLifetimeAdapterContracts(unittest.TestCase):
    """Strict sequential CSV parsing, bounded chunks, and one-pass fit semantics."""

    def adapter(
        self,
        payload: bytes,
        *,
        chunk_bytes: int = 16,
        max_inflight_bytes: int = 32,
    ) -> tuple[CsvLifetimeAdapter, CountingBinarySource]:
        source = CountingBinarySource(payload)
        adapter = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(
                chunk_bytes=chunk_bytes,
                max_inflight_bytes=max_inflight_bytes,
            ),
            opener=source,
        )
        return adapter, source

    def assert_redacted(self, error: CsvLifetimeAdapterError) -> None:
        public = f"{error!r} {error} {error.context!r}"
        self.assertNotIn("private-lifetime-data.csv", public)
        self.assertNotIn("super-secret-cell", public)
        self.assertNotIn("private revision", public)

    def test_csv01_exact_schema_header_event_tokens_and_time_grammar(self) -> None:
        adapter, _ = self.adapter(b"time,event_observed\n1.5,1\n2,0\n")

        chunks = tuple(adapter.iter_chunks())
        observations = tuple(item for chunk in chunks for item in chunk.observations)
        self.assertEqual(len(observations), 2)
        self.assertEqual(adapter.metadata.replayability, Replayability.REPLAYABLE)

        for payload in (
            b"time,event_observed\n1,true\n",
            b"time,event_observed\n1,01\n",
            b"time,event_observed\n1 ,1\n",
            b"time,event_observed\n,1\n",
        ):
            with self.subTest(payload=payload):
                malformed, _ = self.adapter(payload)
                with self.assertRaises(CsvLifetimeAdapterError) as captured:
                    tuple(malformed.iter_chunks())
                self.assertEqual(captured.exception.code, CsvLifetimeErrorCode.ROW_VALUE)
                self.assert_redacted(captured.exception)

    def test_csv02_logical_offsets_and_chunk_ids_ignore_physical_quoted_newlines(self) -> None:
        payload = b"time,event_observed\n1,1\n2,0\n3,1\n"
        small, _ = self.adapter(payload, chunk_bytes=4, max_inflight_bytes=8)
        large, _ = self.adapter(payload, chunk_bytes=64, max_inflight_bytes=64)

        small_chunks = tuple(small.iter_chunks())
        large_chunks = tuple(large.iter_chunks())
        self.assertEqual(
            [
                item.envelope.row_identity(index)
                for item in small_chunks
                for index in range(item.row_count)
            ],
            [
                item.envelope.row_identity(index)
                for item in large_chunks
                for index in range(item.row_count)
            ],
        )
        self.assertEqual(small_chunks[0].envelope.row_start, 0)
        self.assertEqual(small_chunks[-1].envelope.row_stop, 3)
        self.assertEqual(
            [item.envelope.chunk_id for item in small_chunks],
            [f"{SOURCE_ID}:0", f"{SOURCE_ID}:1", f"{SOURCE_ID}:2"],
        )

        embedded_newline, _ = self.adapter(b'time,event_observed\n1,1\n"super-secret-cell\n",0\n')
        with self.assertRaises(CsvLifetimeAdapterError) as captured:
            tuple(embedded_newline.iter_chunks())
        self.assertEqual(captured.exception.code, CsvLifetimeErrorCode.ROW_VALUE)
        self.assertEqual(captured.exception.context["record_offset"], 1)
        self.assert_redacted(captured.exception)

    def test_csv03_retained_logical_byte_bound_oversized_record_and_no_materialization(
        self,
    ) -> None:
        bounded, _ = self.adapter(
            b"time,event_observed\n1,1\n2,0\n3,1\n",
            chunk_bytes=4,
            max_inflight_bytes=8,
        )
        chunks = tuple(bounded.iter_chunks())
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.retained_payload_bytes <= 4 for chunk in chunks))
        self.assertTrue(
            all(chunk.envelope.byte_size == chunk.retained_payload_bytes for chunk in chunks)
        )

        oversized, _ = self.adapter(b"time,event_observed\n123456789,1\n", chunk_bytes=4)
        with self.assertRaises(CsvLifetimeAdapterError) as captured:
            tuple(oversized.iter_chunks())
        self.assertEqual(captured.exception.code, CsvLifetimeErrorCode.RECORD_TOO_LARGE)
        self.assert_redacted(captured.exception)

        generated = b"time,event_observed\n" + b"1,1\n" * 30_000
        streaming, _ = self.adapter(generated, chunk_bytes=4, max_inflight_bytes=4)
        tracemalloc.start()
        try:
            count = sum(chunk.row_count for chunk in streaming.iter_chunks())
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(count, 30_000)
        self.assertLess(peak, 3_000_000)

    def test_csv04_replayable_source_is_opened_exactly_once_for_one_pass_fit(self) -> None:
        adapter, source = self.adapter(b"time,event_observed\n1,1\n3,0\n")

        result = adapter.fit_exponential()

        self.assertIsInstance(result.fit, ExponentialFitSuccess)
        self.assertTrue(result.execution.complete)
        self.assertEqual(source.open_count, 1)
        self.assertEqual(result.execution.provenance.execution.passes.actual_pass_count, 1)
        self.assertEqual(result.execution.provenance.execution.passes.max_passes, 1)

    def test_csv05_typed_redacted_open_decode_schema_row_and_mutation_failures(self) -> None:
        cases: tuple[tuple[bytes, CsvLifetimeErrorCode], ...] = (
            (b"\\xff", CsvLifetimeErrorCode.DECODE),
            (b"time,wrong\n1,1\n", CsvLifetimeErrorCode.SCHEMA),
            (b"time,event_observed\nsuper-secret-cell,1\n", CsvLifetimeErrorCode.ROW_VALUE),
        )
        for payload, expected in cases:
            with self.subTest(expected=expected):
                adapter, _ = self.adapter(payload)
                with self.assertRaises(CsvLifetimeAdapterError) as captured:
                    tuple(adapter.iter_chunks())
                self.assertEqual(captured.exception.code, expected)
                self.assert_redacted(captured.exception)

        missing = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(chunk_bytes=16, max_inflight_bytes=16),
        )
        with self.assertRaises(CsvLifetimeAdapterError) as captured:
            tuple(missing.iter_chunks())
        self.assertEqual(captured.exception.code, CsvLifetimeErrorCode.OPEN)
        self.assert_redacted(captured.exception)

        mutating = MutatingBinarySource(b"time,event_observed\n1,1\n")
        changed = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(chunk_bytes=16, max_inflight_bytes=16),
            opener=mutating,
        )
        with self.assertRaises(CsvLifetimeAdapterError) as captured:
            tuple(changed.iter_chunks())
        self.assertEqual(captured.exception.code, CsvLifetimeErrorCode.SOURCE_MUTATED)
        self.assert_redacted(captured.exception)

    def test_csv06_exact_empty_header_blank_duplicate_malformed_and_extra_column_policy(
        self,
    ) -> None:
        invalid_cases: tuple[bytes, ...] = (
            b"",
            b"time,event_observed\n",
            b"time,event_observed\n\n",
            b"time,time\n1,1\n",
            b"time,event_observed,extra\n1,1,x\n",
            b"time,event_observed\n1,1,extra\n",
            b'time,event_observed\n"unterminated,1\n',
        )
        for payload in invalid_cases:
            with self.subTest(payload=payload):
                adapter, _ = self.adapter(payload)
                with self.assertRaises(CsvLifetimeAdapterError) as captured:
                    tuple(adapter.iter_chunks())
                self.assertIn(
                    captured.exception.code,
                    {
                        CsvLifetimeErrorCode.EMPTY,
                        CsvLifetimeErrorCode.SCHEMA,
                        CsvLifetimeErrorCode.ROW_VALUE,
                    },
                )
                self.assert_redacted(captured.exception)

        with self.assertRaises((TypeError, ValueError)):
            CsvLifetimeLimits(chunk_bytes=True, max_inflight_bytes=16)
        with self.assertRaises((TypeError, ValueError)):
            CsvLifetimeLimits(chunk_bytes=16, max_inflight_bytes=8)

        direct = fit_exponential_csv(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(chunk_bytes=16, max_inflight_bytes=16),
        )
        self.assertIsNone(direct.fit)
        self.assertFalse(direct.execution.complete)
        self.assertNotIsInstance(direct.fit, ExponentialFitFailure)


if __name__ == "__main__":
    unittest.main()
