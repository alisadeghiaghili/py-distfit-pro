"""Contracts for evidence-gated legacy migration governance."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
MIGRATION_ROOT = REPOSITORY_ROOT / "docs" / "migration"
SCHEMA_PATH = MIGRATION_ROOT / "legacy-salvage-ledger.schema.json"
LEDGER_PATH = MIGRATION_ROOT / "legacy-salvage-ledger.json"
CHECKER_PATH = PYTHON_ROOT / "tools" / "check_migration_ledger.py"


class MigrationLedgerTests(unittest.TestCase):
    def test_schema_and_ledger_are_checked_by_stdlib_tool(self) -> None:
        self.assertTrue(SCHEMA_PATH.is_file())
        self.assertTrue(LEDGER_PATH.is_file())
        self.assertTrue(CHECKER_PATH.is_file())
        result = subprocess.run(
            [sys.executable, str(CHECKER_PATH)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ledger_is_draft_2020_12_and_declares_only_allowed_dispositions(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(ledger["target_namespace"], "veridist")
        entries = ledger["entries"]
        self.assertGreaterEqual(len(entries), 2)
        self.assertEqual(
            {entry["disposition"] for entry in entries},
            {"modify_port", "rewrite", "archive"},
        )
        self.assertEqual(len({entry["id"] for entry in entries}), len(entries))
        for entry in entries:
            self.assertEqual(set(entry["isolation"].values()), {False})
            self.assertEqual(set(entry["reviews"]), {"license", "statistical", "scale", "i18n"})
        exponential = next(entry for entry in entries if entry["component"] == "exponential")
        self.assertEqual(exponential["disposition"], "rewrite")

    def test_checker_rejects_a_stale_hash_and_cross_field_policy_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_ledger = Path(temporary_directory) / "ledger.json"
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger["entries"][0]["source"]["sha256"] = "0" * 64
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--ledger", str(temporary_ledger)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sha256", result.stderr)

    def test_checker_rejects_cross_field_isolation_violation_without_mutating_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_ledger = Path(temporary_directory) / "ledger.json"
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger["entries"][1]["isolation"]["oracle"] = True
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--ledger", str(temporary_ledger)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("isolation", result.stderr)

    def test_checker_rejects_archive_with_non_archived_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_ledger = Path(temporary_directory) / "ledger.json"
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger["entries"][0]["status"] = "review_pending"
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--ledger", str(temporary_ledger)],
                cwd=REPOSITORY_ROOT, check=False, capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive", result.stderr)


if __name__ == "__main__":
    unittest.main()
