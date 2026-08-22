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
            self.assertEqual(set(entry["license"]), {"spdx", "basis"})
            self.assertEqual(entry["license"]["spdx"], "MIT")
            self.assertEqual(entry["license"]["basis"], "LICENSE")
        exponential = next(entry for entry in entries if entry["component"] == "exponential")
        self.assertEqual(exponential["disposition"], "rewrite")

    def test_schema_closes_and_types_every_nested_ledger_contract(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        entry = schema["properties"]["entries"]["items"]
        self.assertFalse(entry["additionalProperties"])
        for field in ("target", "source", "evidence", "reviews", "isolation", "license"):
            nested = entry["properties"][field]
            self.assertEqual(nested["type"], "object")
            self.assertFalse(nested["additionalProperties"])
            self.assertTrue(nested["required"])
        self.assertEqual(
            entry["properties"]["source"]["properties"]["path"]["pattern"],
            "^(?:distfit_pro|tests|examples)/",
        )
        self.assertEqual(
            entry["properties"]["reviews"]["properties"]["license"]["enum"],
            ["reviewed", "review_pending", "not_applicable"],
        )

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
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive", result.stderr)

    def test_checker_rejects_extra_keys_and_invalid_review_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_ledger = Path(temporary_directory) / "ledger.json"
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger["entries"][0]["unexpected"] = True
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            command = [sys.executable, str(CHECKER_PATH), "--ledger", str(temporary_ledger)]
            extra = subprocess.run(
                command, cwd=REPOSITORY_ROOT, check=False, capture_output=True, text=True
            )
            ledger["entries"][0].pop("unexpected")
            ledger["entries"][0]["reviews"]["license"] = "unknown"
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            review = subprocess.run(
                command, cwd=REPOSITORY_ROOT, check=False, capture_output=True, text=True
            )
        self.assertNotEqual(extra.returncode, 0)
        self.assertNotEqual(review.returncode, 0)

    def test_checker_rejects_schema_contract_drift_and_wrong_nested_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            schema["properties"]["entries"]["items"]["properties"]["reviews"].pop(
                "additionalProperties"
            )
            temporary_schema = directory / "schema.json"
            temporary_schema.write_text(json.dumps(schema), encoding="utf-8")
            drift = subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--schema", str(temporary_schema)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger["entries"][0]["target"]["public_surface"] = "false"
            temporary_ledger = directory / "ledger.json"
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            wrong_type = subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--ledger", str(temporary_ledger)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(drift.returncode, 0)
        self.assertIn("schema", drift.stderr)
        self.assertNotEqual(wrong_type.returncode, 0)
        self.assertIn("target", wrong_type.stderr)

    def test_checker_distinguishes_unavailable_evidence_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_ledger = Path(temporary_directory) / "ledger.json"
            ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
            ledger["entries"][0]["source"]["commit"] = "0" * 40
            temporary_ledger.write_text(json.dumps(ledger), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--ledger", str(temporary_ledger)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source.commit unavailable", result.stderr)


if __name__ == "__main__":
    unittest.main()
