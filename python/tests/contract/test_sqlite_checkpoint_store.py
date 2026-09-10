"""CKPT-SQL contracts for the local transactional checkpoint backend."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from veridist.engine.checkpoint import (
    CheckpointRecord,
    SQLiteCheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode


def initial_record() -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id="source",
        source_schema="schema",
        source_revision="private-revision",
        reducer_id="reducer",
        accumulator_schema="accumulator",
        plan_digest="plan",
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=b"\x00binary-state",
    )


def next_record(record: CheckpointRecord) -> CheckpointRecord:
    return record.next_generation(
        cursor=1,
        committed_ranges=((0, 1),),
        operation_token="chunk-1",
        operation_digest="digest-1",
        state=b"\xffnext-state",
    )


class SQLiteCheckpointStoreContractTests(unittest.TestCase):
    def test_ckpt_sql01_create_reopen_and_binary_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            self.assertEqual(store.read(), initial)
            candidate = next_record(initial)
            self.assertEqual(store.compare_and_swap(0, candidate), candidate)
            self.assertEqual(SQLiteCheckpointStore(path).read(), candidate)

    def test_ckpt_sql02_create_and_stale_writer_conflicts_are_typed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            first = SQLiteCheckpointStore.create(path, initial)
            with self.assertRaises(EngineContractError) as existing:
                SQLiteCheckpointStore.create(path, initial)
            self.assertIs(existing.exception.code, FailureCode.CHECKPOINT_ALREADY_EXISTS)

            candidate = next_record(initial)
            self.assertEqual(first.compare_and_swap(0, candidate), candidate)
            with self.assertRaises(EngineContractError) as stale:
                SQLiteCheckpointStore(path).compare_and_swap(0, candidate)
            self.assertIs(stale.exception.code, FailureCode.CHECKPOINT_CONFLICT)

    def test_ckpt_sql03_corruption_and_invalid_candidate_fail_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            invalid = replace(next_record(initial), checksum="not-a-checksum")
            with self.assertRaises(EngineContractError) as candidate_error:
                store.compare_and_swap(0, invalid)
            self.assertIs(candidate_error.exception.code, FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
            self.assertEqual(store.read(), initial)

            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE checkpoint SET payload = ? WHERE singleton = 1", ("{}",))
            with self.assertRaises(EngineContractError) as malformed:
                SQLiteCheckpointStore(path).read()
            self.assertIs(malformed.exception.code, FailureCode.CHECKPOINT_DECODE_FAILED)

    def test_ckpt_sql04_missing_store_and_unsupported_format_have_distinct_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            with self.assertRaises(EngineContractError) as missing:
                SQLiteCheckpointStore(path).read()
            self.assertIs(missing.exception.code, FailureCode.CHECKPOINT_NOT_FOUND)

            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE checkpoint SET format_version = 99 WHERE singleton = 1")
            with self.assertRaises(EngineContractError) as unsupported:
                store.read()
            self.assertIs(unsupported.exception.code, FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
