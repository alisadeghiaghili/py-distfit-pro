"""RED contracts for the 0.6 resumable CSV execution milestone."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore


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


if __name__ == "__main__":
    unittest.main()
