"""RED contracts for fail-closed frozen migration provenance."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.quality.test_migration_ledger import CHECKER_PATH, LEDGER_PATH, REPOSITORY_ROOT

SOURCE_LOCKS_PATH = REPOSITORY_ROOT / "docs" / "migration" / "legacy-source-locks.json"
LM004_COMMIT = "6b021995b25dab81e3fdb9c18410cf974363848a"
OTHER_VALID_LM004_COMMIT = "61e7ab62c1962b95bd012b282e11d0556959a49f"


class MigrationLedgerFailClosedContractTests(unittest.TestCase):
    """Mutation probes for immutable commits, blobs, and structured line ranges."""

    def _ledger(self) -> dict[str, object]:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    def _lm004(self, ledger: dict[str, object]) -> dict[str, object]:
        entries = ledger["entries"]
        assert isinstance(entries, list)
        entry = next(item for item in entries if item["id"] == "LM-004")
        assert isinstance(entry, dict)
        return entry

    def _check(self, ledger: dict[str, object]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ledger.json"
            path.write_text(json.dumps(ledger), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(CHECKER_PATH), "--ledger", str(path)],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_lm004_declares_structured_path_bound_line_ranges(self) -> None:
        ledger = self._ledger()
        source = self._lm004(ledger)["source"]
        assert isinstance(source, dict)
        ranges = source["line_ranges"]
        self.assertEqual(
            ranges,
            [
                {"path": source["path"], "start_line": 34, "end_line": 69},
                {"path": source["path"], "start_line": 153, "end_line": 195},
                {"path": source["path"], "start_line": 245, "end_line": 283},
                {"path": source["path"], "start_line": 286, "end_line": 327},
                {"path": source["path"], "start_line": 367, "end_line": 403},
                {"path": source["path"], "start_line": 1047, "end_line": 1101},
            ],
        )

    def test_lm004_commit_is_separately_frozen_by_the_source_lock(self) -> None:
        locks = json.loads(SOURCE_LOCKS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(locks["schema_version"], "1.0")
        self.assertEqual(
            locks["entries"],
            {
                "LM-001": "6b021995b25dab81e3fdb9c18410cf974363848a",
                "LM-002": "6b021995b25dab81e3fdb9c18410cf974363848a",
                "LM-003": "6b021995b25dab81e3fdb9c18410cf974363848a",
                "LM-004": LM004_COMMIT,
            },
        )

    def test_checker_rejects_a_different_valid_commit_with_the_same_lm004_blob(self) -> None:
        ledger = self._ledger()
        source = self._lm004(ledger)["source"]
        assert isinstance(source, dict)
        self.assertEqual(source["commit"], LM004_COMMIT)
        source["commit"] = OTHER_VALID_LM004_COMMIT
        result = self._check(ledger)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source lock", result.stderr)

    def test_checker_rejects_missing_wrong_and_noncommit_source_revisions(self) -> None:
        for replacement in ("0" * 40, "f" * 40, self._tree_id()):
            with self.subTest(replacement=replacement):
                ledger = self._ledger()
                source = self._lm004(ledger)["source"]
                assert isinstance(source, dict)
                source["commit"] = replacement
                result = self._check(ledger)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("commit", result.stderr)

    def test_checker_rejects_wrong_blob_sha_path_and_invalid_line_ranges(self) -> None:
        mutations = (
            lambda source: source.__setitem__("blob", "0" * 40),
            lambda source: source.__setitem__("sha256", "0" * 64),
            lambda source: source.__setitem__("path", "distfit_pro/core/base.py"),
            lambda source: source.__setitem__("line_ranges", []),
            lambda source: source.__setitem__(
                "line_ranges", [{"path": source["path"], "start_line": 0, "end_line": 1}]
            ),
            lambda source: source.__setitem__(
                "line_ranges", [{"path": source["path"], "start_line": 5, "end_line": 4}]
            ),
            lambda source: source.__setitem__(
                "line_ranges", [{"path": source["path"], "start_line": 1, "end_line": 999999}]
            ),
            lambda source: source.__setitem__(
                "line_ranges",
                [{"path": "distfit_pro/core/base.py", "start_line": 1, "end_line": 1}],
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                ledger = copy.deepcopy(self._ledger())
                source = self._lm004(ledger)["source"]
                assert isinstance(source, dict)
                mutate(source)
                self.assertNotEqual(self._check(ledger).returncode, 0)

    @staticmethod
    def _tree_id() -> str:
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        entries = ledger["entries"]
        assert isinstance(entries, list)
        entry = next(item for item in entries if item["id"] == "LM-004")
        assert isinstance(entry, dict)
        source = entry["source"]
        assert isinstance(source, dict)
        commit = source["commit"]
        assert isinstance(commit, str)
        result = subprocess.run(
            ["git", "rev-parse", f"{commit}^{{tree}}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
