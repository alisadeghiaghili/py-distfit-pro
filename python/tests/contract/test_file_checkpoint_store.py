from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from veridist.engine.checkpoint import CheckpointRecord, FileCheckpointStore
from veridist.engine.errors import EngineContractError


class FileCheckpointStoreTests(unittest.TestCase):
    def test_round_trip_and_atomic_compare_and_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            initial = CheckpointRecord.create(
                format_version=1,
                source_id="source",
                source_schema="schema",
                source_revision="revision",
                reducer_id="reducer",
                accumulator_schema="accumulator",
                plan_digest="plan",
                cursor=0,
                committed_ranges=(),
                generation=0,
                operation_token=None,
                operation_digest=None,
                state=b"0",
            )
            store = FileCheckpointStore.create(path, initial)
            self.assertEqual(store.read(), initial)
            candidate = initial.next_generation(
                cursor=1,
                committed_ranges=((0, 1),),
                operation_token="chunk-1",
                operation_digest="digest-1",
                state=b"1",
            )
            self.assertEqual(store.compare_and_swap(0, candidate), candidate)
            self.assertEqual(FileCheckpointStore(path).read(), candidate)

    def test_tampering_is_rejected_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            initial = CheckpointRecord.create(
                format_version=1, source_id="source", source_schema="schema",
                source_revision="revision", reducer_id="reducer",
                accumulator_schema="accumulator", plan_digest="plan", cursor=0,
                committed_ranges=(), generation=0, operation_token=None,
                operation_digest=None, state=b"0",
            )
            FileCheckpointStore.create(path, initial)
            tampered = path.read_text(encoding="utf-8").replace('"state":"MA=="', '"state":"MQ=="')
            path.write_text(tampered, encoding="utf-8")
            with self.assertRaises(EngineContractError):
                FileCheckpointStore(path).read()


if __name__ == "__main__":
    unittest.main()
