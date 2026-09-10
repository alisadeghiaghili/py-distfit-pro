"""RED contracts for the 0.6 resumable CSV execution milestone."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId
from veridist.adapters.csv_lifetimes import CsvLifetimeChunk
from veridist.domain.lifetimes import ExactLifetime
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore
from veridist.engine.delivery import ChunkEnvelope
from veridist.engine.errors import EngineContractError, FailureCode


class V1CheckpointedCsvTests(unittest.TestCase):
    def _store(
        self,
        directory: str,
        revision: str = "revision-a",
        reducer_id: str = "exponential-reduction-v1",
        accumulator_schema: str = "exponential-reduction-v1",
    ) -> SQLiteCheckpointStore:
        state = (
            b'{"compensation":"0x0.0p+0","event_count":0,'
            b'"observation_count":0,"total_time":"0x0.0p+0"}'
        )
        initial = CheckpointRecord.create(
            format_version=1,
            source_id="source",
            source_schema="csv-lifetime-v1",
            source_revision=revision,
            reducer_id=reducer_id,
            accumulator_schema=accumulator_schema,
            plan_digest="plan",
            cursor=0,
            committed_ranges=(),
            generation=0,
            operation_token=None,
            operation_digest=None,
            state=state,
        )
        path = Path(directory) / f"{reducer_id}-{accumulator_schema}.sqlite3"
        return SQLiteCheckpointStore.create(path, initial)

    def test_checkpointed_csv_api_is_keyword_explicit(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        signature = inspect.signature(fit_exponential_checkpointed_csv)
        self.assertEqual(
            tuple(signature.parameters),
            ("path", "schema", "source_id", "limits", "store", "source_revision", "cancel"),
        )
        self.assertTrue(
            all(
                item.kind is inspect.Parameter.KEYWORD_ONLY
                for item in signature.parameters.values()
            )
        )

    def test_resume_does_not_apply_committed_rows_twice(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n2,0\n3,1\n", encoding="utf-8")
            store = self._store(directory)
            first = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=CsvLifetimeLimits(24, 48),
                store=store,
                source_revision="revision-a",
                cancel=lambda cursor: cursor >= 2,
            )
            self.assertEqual(first.code, "CANCELLED")
            committed_cursor = store.read().cursor
            self.assertEqual(committed_cursor, 2)
            resumed = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=CsvLifetimeLimits(24, 48),
                store=SQLiteCheckpointStore(store.path),
                source_revision="revision-a",
                cancel=None,
            )
        self.assertEqual(resumed.fit.observation_count, 3)
        self.assertEqual(resumed.fit.event_count, 2)
        self.assertEqual(resumed.fit.total_time, 6.0)

    def test_source_revision_change_cannot_advance_checkpoint(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            store = self._store(directory)
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision="revision-b",
                cancel=None,
            )
            self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")
            self.assertEqual(store.read().cursor, 0)

    def test_contract_input_types_fail_before_storage_or_source_access(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        common = {
            "path": Path("source.csv"),
            "schema": CsvLifetimeSchema("time", "event_observed"),
            "source_id": PublicSourceId("src_0123456789abcdef0123456789abcdef"),
            "limits": CsvLifetimeLimits(32, 64),
            "store": object(),
            "source_revision": "revision-a",
            "cancel": None,
        }
        for name, value in (
            ("path", "source.csv"),
            ("schema", object()),
            ("source_id", object()),
            ("limits", object()),
            ("cancel", object()),
        ):
            with self.subTest(name=name):
                arguments = dict(common)
                arguments[name] = value
                with self.assertRaises(TypeError):
                    fit_exponential_checkpointed_csv(**arguments)

    def test_incompatible_reducer_metadata_is_rejected_before_reading_csv(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            for reducer_id, schema, expected in (
                ("other-reducer", "exponential-reduction-v1", "REDUCER_MISMATCH"),
                ("exponential-reduction-v1", "other-schema", "ACCUMULATOR_SCHEMA_MISMATCH"),
            ):
                with self.subTest(expected=expected):
                    store = self._store(directory, reducer_id=reducer_id, accumulator_schema=schema)
                    result = fit_exponential_checkpointed_csv(
                        path=source,
                        schema=CsvLifetimeSchema("time", "event_observed"),
                        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                        limits=CsvLifetimeLimits(32, 64),
                        store=store,
                        source_revision="revision-a",
                        cancel=None,
                    )
                    self.assertEqual(result.code, expected)

    def test_checksum_mismatch_is_rejected_before_constructing_the_adapter(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class CorruptStore:
            def read(self) -> CheckpointRecord:
                return replace(self_record, checksum="not-a-valid-checksum")

        with tempfile.TemporaryDirectory() as directory:
            self_record = self._store(directory).read()
            result = fit_exponential_checkpointed_csv(
                path=Path(directory) / "unread.csv",
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=CsvLifetimeLimits(32, 64),
                store=CorruptStore(),
                source_revision="revision-a",
                cancel=None,
            )
        self.assertEqual(result.code, "CHECKPOINT_CHECKSUM_MISMATCH")

    def test_range_gap_from_adapter_boundary_is_not_silently_repaired(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class StaticStore:
            def read(self) -> CheckpointRecord:
                return record

        class GappedAdapter:
            def __init__(self, *unused: object) -> None:
                return None

            def iter_chunks(self) -> object:
                yield CsvLifetimeChunk(
                    ChunkEnvelope(
                        source_id="src_0123456789abcdef0123456789abcdef",
                        chunk_id="chunk-gap",
                        sequence_number=0,
                        row_start=1,
                        row_stop=2,
                        byte_size=1,
                    ),
                    (ExactLifetime(Decimal("1")),),
                    1,
                )

        with tempfile.TemporaryDirectory() as directory:
            record = self._store(directory).read()
            with patch("veridist.execution.CsvLifetimeAdapter", GappedAdapter):
                result = fit_exponential_checkpointed_csv(
                    path=Path(directory) / "unused.csv",
                    schema=CsvLifetimeSchema("time", "event_observed"),
                    source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                    limits=CsvLifetimeLimits(32, 64),
                    store=StaticStore(),
                    source_revision="revision-a",
                    cancel=None,
                )
        self.assertEqual(result.code, "RANGE_MISMATCH")

    def test_engine_contract_error_from_checkpoint_boundary_is_returned_as_a_code(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class FailingStore:
            def read(self) -> CheckpointRecord:
                raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED)

        result = fit_exponential_checkpointed_csv(
            path=Path("unread.csv"),
            schema=CsvLifetimeSchema("time", "event_observed"),
            source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
            limits=CsvLifetimeLimits(32, 64),
            store=FailingStore(),
            source_revision="revision-a",
            cancel=None,
        )
        self.assertEqual(result.code, "CHECKPOINT_STORAGE_FAILED")

    def test_final_checkpoint_revision_is_rechecked_after_an_empty_source(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class RevisionChangingStore:
            def __init__(self, initial: CheckpointRecord) -> None:
                self._initial = initial
                self._reads = 0

            def read(self) -> CheckpointRecord:
                self._reads += 1
                if self._reads == 1:
                    return self._initial
                return replace(self._initial, source_revision="revision-b")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.csv"
            source.write_text("time,event_observed\n", encoding="utf-8")
            initial = self._store(directory).read()
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=CsvLifetimeLimits(32, 64),
                store=RevisionChangingStore(initial),
                source_revision="revision-a",
                cancel=None,
            )
        self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")


if __name__ == "__main__":
    unittest.main()
