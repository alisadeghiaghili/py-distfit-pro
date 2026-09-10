"""Adversarial unit contracts for the fail-closed v1 release-evidence gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.check_competitive_source_lock import claim_id
from tools.check_competitive_source_lock import validate as validate_source_locks
from tools.check_v1_execution_evidence import validate as validate_execution

SHA = "a" * 40


def _execution_payload() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for platform in ("linux", "macos", "windows"):
        for rows in (10_000, 100_000, 1_000_000):
            for scenario in ("complete", "retry_resume", "cancel"):
                cells.append(
                    {
                        "platform": platform,
                        "rows": rows,
                        "scenario": scenario,
                        "candidate_git_sha": SHA,
                        "actual_passes": 1,
                        "peak_rss_bytes": 1,
                        "max_inflight_bytes": 1,
                        "observed_inflight_bytes": 1,
                        "canonical_result_equal": scenario != "retry_resume" or True,
                        "resources_released": scenario != "cancel" or True,
                    }
                )
    return {"schema_version": 1, "candidate_git_sha": SHA, "cells": cells}


class V1EvidenceCheckerTests(unittest.TestCase):
    def test_execution_evidence_requires_the_complete_candidate_bound_matrix(self) -> None:
        payload = _execution_payload()
        self.assertEqual(validate_execution(payload, SHA), [])
        cells = payload["cells"]
        assert isinstance(cells, list)
        cells.pop()
        self.assertIn("matrix", " ".join(validate_execution(payload, SHA)))

    def test_source_lock_requires_every_publishable_claim_cell(self) -> None:
        row = {
            "project": "fixture",
            "language": "Python",
            "capability": "fit",
            "status": "supported",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "matrix.csv"
            locks = root / "locks.json"
            matrix.write_text(
                "project,language,capability,status\nfixture,Python,fit,supported\n",
                encoding="utf-8",
            )
            locks.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "locks": [
                            {
                                "claim_id": claim_id(row),
                                "source_locator": "https://example.invalid/source",
                                "reviewed_at": "2026-09-10",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_source_locks(matrix, locks), [])
            locks.write_text('{"schema_version": 1, "locks": []}', encoding="utf-8")
            self.assertIn("coverage", " ".join(validate_source_locks(matrix, locks)))


if __name__ == "__main__":
    unittest.main()
