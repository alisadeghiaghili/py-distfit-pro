"""CSV-01--CSV-06 executable RED contracts for strict lifetime CSV ingestion."""

from __future__ import annotations

import re
import unittest
from io import BytesIO
from pathlib import Path

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeAdapterError,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    fit_exponential_csv,
    retained_object_graph_bytes,
)
from veridist.engine.errors import FailureCode
from veridist.engine.outcome import FailedOutcome, UnknownMissingRanges
from veridist.engine.provenance import PublicSourceId
from veridist.execution import ExponentialSourceFitResult, fit_exponential_source
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitFailureCode

SOURCE_ID = PublicSourceId("src_0123456789abcdef0123456789abcdef")
SCHEMA = CsvLifetimeSchema(time_column="time", event_observed_column="event_observed")
_CHUNK_ID = re.compile(r"chk_[0-9a-f]{32}")


class TrackingBinaryStream(BytesIO):
    """A binary stream that fails any whole-file read or seek attempt."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.close_count = 0
        self.read_requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > 8192:
            raise AssertionError("unbounded binary read")
        self.read_requests.append(size)
        return super().read(size)

    def readall(self) -> bytes:
        raise AssertionError("readall is forbidden")

    def seek(self, offset: int, whence: int = 0) -> int:
        del offset, whence
        raise AssertionError("seek is forbidden")

    def close(self) -> None:
        self.close_count += 1
        super().close()


class TrackingSource:
    """Replayable test seam with observable opens, identities, and close lifecycle."""

    def __init__(self, payload: bytes, *, changed_after_open: bool = False) -> None:
        self._changed_after_open = changed_after_open
        self._payload = payload
        self._revision = "rev0"
        self.open_count = 0
        self.streams: list[TrackingBinaryStream] = []

    def identity(self, path: Path) -> object:
        del path
        return self._revision

    def open_binary(self, path: Path) -> TrackingBinaryStream:
        del path
        self.open_count += 1
        if self._changed_after_open:
            self._revision = "rev1"
        stream = TrackingBinaryStream(self._payload)
        self.streams.append(stream)
        return stream

    @property
    def close_count(self) -> int:
        return sum(stream.close_count for stream in self.streams)


class UnexpectedSource(TrackingSource):
    """A seam for checking that programming faults are never relabelled."""

    def open_binary(self, path: Path) -> TrackingBinaryStream:
        del path
        raise RuntimeError("unexpected source fault")


class CsvLifetimeAdapterContracts(unittest.TestCase):
    """Strict CSV semantics must be proven through adapter and execution seams."""

    def adapter(
        self,
        payload: bytes,
        *,
        chunk_bytes: int = 2048,
        max_inflight_bytes: int = 4096,
        changed_after_open: bool = False,
    ) -> tuple[CsvLifetimeAdapter, TrackingSource]:
        source = TrackingSource(payload, changed_after_open=changed_after_open)
        return (
            CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(chunk_bytes, max_inflight_bytes),
                opener=source,
            ),
            source,
        )

    def assert_redacted(self, error: CsvLifetimeAdapterError) -> None:
        public = f"{error!r} {error} {error.context!r}"
        self.assertNotIn("private-lifetime-data.csv", public)
        self.assertNotIn("super-secret-cell", public)
        self.assertNotIn("rev1", public)

    def assert_adapter_error(
        self,
        adapter: CsvLifetimeAdapter,
        *,
        code: FailureCode,
        reason: str,
    ) -> None:
        with self.assertRaises(CsvLifetimeAdapterError) as captured:
            tuple(adapter.iter_chunks())
        self.assertEqual(captured.exception.code, code)
        self.assertEqual(captured.exception.context, {"reason": reason})
        self.assert_redacted(captured.exception)

    def test_csv01_exact_schema_tokens_and_ascii_decimal_grammar(self) -> None:
        adapter, _ = self.adapter(b"\xef\xbb\xbftime,event_observed\r\n0,1\r\n1.25e+1,0\r\n")
        chunks = tuple(adapter.iter_chunks())
        observations = tuple(item for chunk in chunks for item in chunk.observations)
        self.assertEqual(len(observations), 2)

        for value in (b" 1", b"+1", b"01", b".5", b"1.", b"NaN", b"Inf", b"1e-999999"):
            with self.subTest(value=value):
                invalid, _ = self.adapter(b"time,event_observed\n" + value + b",1\n")
                self.assert_adapter_error(
                    invalid,
                    code=FailureCode.SOURCE_ROW_INVALID,
                    reason="row_invalid",
                )

    def test_csv02_logical_offsets_are_stable_but_chunk_ids_vary_with_budget(self) -> None:
        payload = b"time,event_observed\n1,1\n2,0\n3,1\n"
        narrow, _ = self.adapter(payload, chunk_bytes=2048, max_inflight_bytes=4096)
        wide, _ = self.adapter(payload, chunk_bytes=8192, max_inflight_bytes=8192)
        narrow_chunks = tuple(narrow.iter_chunks())
        wide_chunks = tuple(wide.iter_chunks())

        narrow_rows = [
            chunk.envelope.row_identity(index)
            for chunk in narrow_chunks
            for index in range(chunk.envelope.row_count)
        ]
        wide_rows = [
            chunk.envelope.row_identity(index)
            for chunk in wide_chunks
            for index in range(chunk.envelope.row_count)
        ]
        self.assertEqual(narrow_rows, wide_rows)
        self.assertEqual(
            narrow_rows, [(SOURCE_ID.value, 0), (SOURCE_ID.value, 1), (SOURCE_ID.value, 2)]
        )
        self.assertTrue(
            all(_CHUNK_ID.fullmatch(chunk.envelope.chunk_id) for chunk in narrow_chunks)
        )
        self.assertTrue(all(_CHUNK_ID.fullmatch(chunk.envelope.chunk_id) for chunk in wide_chunks))

        multiline, _ = self.adapter(b'time,event_observed\n1,1\n"super-secret-cell\n",0\n')
        self.assert_adapter_error(
            multiline,
            code=FailureCode.SOURCE_SCHEMA_INVALID,
            reason="malformed_record",
        )

    def test_csv03_object_graph_accounting_and_guarded_stream_forbid_materialization(self) -> None:
        adapter, source = self.adapter(b"time,event_observed\n1,1\n2,0\n")
        chunks = tuple(adapter.iter_chunks())
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.retained_payload_bytes, retained_object_graph_bytes(chunk))
            self.assertLessEqual(chunk.retained_payload_bytes, adapter.limits.chunk_bytes)
            self.assertEqual(chunk.envelope.byte_size, chunk.retained_payload_bytes)
        self.assertTrue(source.streams)
        self.assertTrue(
            all(
                request >= 0 and request <= 8192
                for stream in source.streams
                for request in stream.read_requests
            )
        )

        oversized, _ = self.adapter(b"time,event_observed\n1" + b"0" * 10000 + b",1\n")
        self.assert_adapter_error(
            oversized,
            code=FailureCode.CHUNK_TOO_LARGE,
            reason="record_too_large",
        )

    def test_csv04_replay_is_two_opens_and_one_pass_execution_is_one_open(self) -> None:
        adapter, source = self.adapter(b"time,event_observed\n1,1\n3,0\n")
        self.assertEqual(tuple(adapter.iter_chunks()), tuple(adapter.iter_chunks()))
        self.assertEqual(source.open_count, 2)

        result = fit_exponential_source(adapter)
        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertEqual(source.open_count, 3)
        self.assertEqual(result.execution.provenance.execution.passes.actual_pass_count, 1)
        self.assertEqual(result.execution.provenance.execution.passes.max_passes, 1)

    def test_csv05_failure_codes_reasons_lifecycle_and_closed_outcome_mapping(self) -> None:
        cases = (
            (b"", FailureCode.SOURCE_SCHEMA_INVALID, "header_missing"),
            (b"\xef\xbb\xbf", FailureCode.SOURCE_SCHEMA_INVALID, "header_missing"),
            (b"\xff", FailureCode.SOURCE_DECODE_FAILED, "decode_failed"),
            (b"time,time\n1,1\n", FailureCode.SOURCE_SCHEMA_INVALID, "header_duplicate"),
            (b"time\n1\n", FailureCode.SOURCE_SCHEMA_INVALID, "header_missing"),
            (
                b"time,\xef\xbb\xbfevent_observed\n1,1\n",
                FailureCode.SOURCE_SCHEMA_INVALID,
                "header_missing",
            ),
            (
                b"time,event_observed,extra\n1,1,x\n",
                FailureCode.SOURCE_SCHEMA_INVALID,
                "extra_column",
            ),
            (b"time,event_observed\n\n", FailureCode.SOURCE_ROW_INVALID, "row_invalid"),
            (
                b"time,event_observed\nsuper-secret-cell,1\n",
                FailureCode.SOURCE_ROW_INVALID,
                "row_invalid",
            ),
        )
        for payload, code, reason in cases:
            with self.subTest(code=code, reason=reason):
                adapter, source = self.adapter(payload)
                self.assert_adapter_error(adapter, code=code, reason=reason)
                self.assertEqual(source.close_count, 1)

        changed, source = self.adapter(
            b"time,event_observed\n1,1\n",
            changed_after_open=True,
        )
        self.assert_adapter_error(
            changed,
            code=FailureCode.SOURCE_REVISION_MISMATCH,
            reason="source_mutated",
        )
        self.assertEqual(source.close_count, 1)

        early, source = self.adapter(b"time,event_observed\n1,1\n")
        iterator = early.iter_chunks()
        iterator.close()
        self.assertEqual(source.close_count, 1)

        unexpected_source = UnexpectedSource(b"time,event_observed\n1,1\n")
        unexpected = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=unexpected_source,
        )
        with self.assertRaises(RuntimeError):
            tuple(unexpected.iter_chunks())

        result = fit_exponential_csv(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
        )
        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertIsNone(result.fit)
        self.assertIsInstance(result.execution.outcome, FailedOutcome)
        assert isinstance(result.execution.outcome, FailedOutcome)
        self.assertEqual(result.execution.outcome.failure.code, FailureCode.SOURCE_OPEN_FAILED)
        self.assertIsInstance(result.execution.outcome.coverage, UnknownMissingRanges)
        self.assertEqual(
            result.execution.outcome.failure.stage.value,
            "preflight",
        )

    def test_csv06_empty_header_and_full_scan_precedence_constructor_and_result_invariants(
        self,
    ) -> None:
        header_only, source = self.adapter(b"time,event_observed\n")
        result = fit_exponential_source(header_only)
        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertIsInstance(result.fit, ExponentialFitFailure)
        assert isinstance(result.fit, ExponentialFitFailure)
        self.assertEqual(result.fit.code, ExponentialFitFailureCode.EMPTY_SAMPLE)
        self.assertTrue(result.execution.outcome.complete)
        self.assertEqual(source.close_count, 1)

        overflow_then_invalid, _ = self.adapter(
            b"time,event_observed\n1e10000,1\nsuper-secret-cell,1\n"
        )
        failed = fit_exponential_source(overflow_then_invalid)
        self.assertIsInstance(failed, ExponentialSourceFitResult)
        assert isinstance(failed, ExponentialSourceFitResult)
        self.assertIsNone(failed.fit)
        self.assertEqual(
            failed.execution.outcome.failure.code,  # type: ignore[union-attr]
            FailureCode.SOURCE_ROW_INVALID,
        )

        for columns in (("", "event_observed"), ("time", "time")):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    CsvLifetimeSchema(*columns)
        for limits in (
            (True, 1),
            (0, 1),
            (1, 0),
            (2, 1),
        ):
            with self.subTest(limits=limits):
                with self.assertRaises((TypeError, ValueError)):
                    CsvLifetimeLimits(*limits)
        with self.assertRaises(TypeError):
            CsvLifetimeAdapter(  # type: ignore[arg-type]
                "not-a-path",
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
            )
        with self.assertRaises(TypeError):
            CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
                opener=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
