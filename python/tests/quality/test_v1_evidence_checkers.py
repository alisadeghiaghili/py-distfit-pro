"""Adversarial unit contracts for the fail-closed v1 release-evidence gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.assemble_v1_execution_evidence import assemble
from tools.check_v1_execution_evidence import validate as validate_execution
from tools.check_v1_execution_evidence import validate_raw_fragment
from tools.collect_v1_execution_evidence import _run_scenario

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
                        "attempt_count": 2 if scenario == "retry_resume" else 1,
                        "process_peak_rss_bytes": 1,
                        "canonical_result_equal": scenario != "retry_resume" or True,
                        "resources_released": scenario != "cancel" or True,
                    }
                )
    return {"schema_version": 1, "candidate_git_sha": SHA, "cells": cells}


def _raw_payload(platform: str) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for rows in (10_000, 100_000, 1_000_000):
        for scenario in ("complete", "retry_resume", "cancel"):
            cells.append(
                {
                    "platform": platform,
                    "rows": rows,
                    "scenario": scenario,
                    "candidate_git_sha": SHA,
                    "attempt_count": 2 if scenario == "retry_resume" else 1,
                    "process_peak_rss_bytes": 1,
                    "interrupted_cursor": 5 if scenario in {"retry_resume", "cancel"} else 0,
                    "final_cursor": 5 if scenario == "cancel" else rows,
                    "source_bytes": 1,
                    "source_sha256": "b" * 64,
                    "result_sha256": None if scenario == "cancel" else "c" * 64,
                    "result_code": "CANCELLED" if scenario == "cancel" else "COMPLETE",
                    "retry_initial_code": "CANCELLED" if scenario == "retry_resume" else None,
                    "canonical_result_equal": scenario != "retry_resume" or True,
                    "cancellation_observed": scenario in {"retry_resume", "cancel"},
                    "resources_released": scenario == "cancel",
                }
            )
    return {
        "schema_version": 1,
        "artifact_kind": "v1-execution-raw",
        "candidate_git_sha": SHA,
        "platform": platform,
        "host_platform": "fixture",
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "veridist_version": "0.9.0",
        "collected_at": "2026-09-10T00:00:00Z",
        "cells": cells,
    }


class V1EvidenceCheckerTests(unittest.TestCase):
    def test_execution_evidence_requires_the_complete_candidate_bound_matrix(self) -> None:
        payload = _execution_payload()
        self.assertEqual(validate_execution(payload, SHA), [])
        cells = payload["cells"]
        assert isinstance(cells, list)
        cells.pop()
        self.assertIn("matrix", " ".join(validate_execution(payload, SHA)))

    def test_raw_platform_fragments_require_executed_retry_and_cleanup_facts(self) -> None:
        payload = _raw_payload("linux")
        self.assertEqual(validate_raw_fragment(payload, SHA), [])
        cells = payload["cells"]
        assert isinstance(cells, list)
        cancellation = next(cell for cell in cells if cell["scenario"] == "cancel")
        cancellation["resources_released"] = False
        self.assertIn("cleanup", " ".join(validate_raw_fragment(payload, SHA)))

    def test_assembler_accepts_only_all_executed_platform_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: list[Path] = []
            for platform in ("linux", "macos", "windows"):
                path = root / f"{platform}.json"
                path.write_text(json.dumps(_raw_payload(platform)), encoding="utf-8")
                paths.append(path)
            value, errors = assemble(paths, SHA)
            self.assertEqual(errors, [])
            self.assertIsNotNone(value)
            value, errors = assemble(paths[:2], SHA)
            self.assertIsNone(value)
            self.assertIn("matrix", " ".join(errors))

    def test_assembler_rejects_cross_platform_result_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: list[Path] = []
            for platform in ("linux", "macos", "windows"):
                payload = _raw_payload(platform)
                if platform == "windows":
                    cells = payload["cells"]
                    assert isinstance(cells, list)
                    cells[0]["result_sha256"] = "d" * 64
                path = root / f"{platform}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            value, errors = assemble(paths, SHA)
            self.assertIsNone(value)
            self.assertIn("result digests differ", " ".join(errors))

    def test_local_collector_observes_real_retry_and_cancel_paths(self) -> None:
        retry = _run_scenario(3, "retry_resume")
        cancellation = _run_scenario(3, "cancel")
        self.assertTrue(retry["canonical_result_equal"])
        self.assertEqual(retry["retry_initial_code"], "CANCELLED")
        self.assertGreater(retry["interrupted_cursor"], 0)
        self.assertEqual(retry["final_cursor"], 3)
        self.assertEqual(cancellation["result_code"], "CANCELLED")
        self.assertGreater(cancellation["interrupted_cursor"], 0)
        self.assertTrue(cancellation["resources_released"])


if __name__ == "__main__":
    unittest.main()
