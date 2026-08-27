"""SCALE-CSV-EXP retained-evidence contracts.

The checker is deliberately a stdlib command boundary: artifact acceptance must
not depend on the production implementation used to create it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[2] / "tools"
CHECKER = TOOLS / "check_scale_csv_exponential_evidence.py"
RUNNER = TOOLS / "run_scale_csv_exponential_evidence.py"


def _cell(rows: int, chunk_bytes: int) -> dict[str, object]:
    return {
        "rows": rows,
        "chunk_bytes": chunk_bytes,
        "max_inflight_bytes": chunk_bytes,
        "source": {"bytes": 1, "sha256": "b" * 64},
        "observed": {
            "actual_pass_count": 1,
            "max_passes": 1,
            "accepted_chunk_count": 1,
            "processed_row_count": rows,
            "peak_inflight_bytes": 128,
            "largest_retained_chunk_bytes": 128,
            "backpressure_event_count": 0,
        },
        "fit": {
            "observation_count": rows,
            "event_count": rows // 2,
            "total_time": 1.0,
            "rate": rows / 2.0,
            "expected_event_count": rows // 2,
            "expected_total_time": "1",
            "expected_rate": str(rows // 2),
            "absolute_rate_error": 0.0,
            "relative_rate_error": 0.0,
        },
        "memory": {
            "tracemalloc_peak_bytes": 1,
            "rss_peak_bytes": None,
            "rss_delta_bytes": None,
        },
        "elapsed_seconds": 0.01,
    }


def _artifact() -> dict[str, object]:
    cells = [_cell(rows, budget) for rows in (10_000, 100_000, 1_000_000) for budget in (1024, 2048, 4096)]
    value: dict[str, object] = {
        "schema_version": "1",
        "run": {
            "git_sha": "a" * 40,
            "git_dirty": False,
            "utc_started": "2026-08-27T00:00:00Z",
            "python": {"implementation": "CPython", "version": "3.11.0"},
            "platform": "test-platform",
        },
        "generator": {"formula_version": "1", "temporary_root": "redacted"},
        "cells": cells,
        "operation_evidence": {"rows": [10000, 100000, 1000000], "accepted_chunks": [1, 1, 1]},
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    value["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


class ScaleCsvExponentialEvidenceTests(unittest.TestCase):
    def _check(self, artifact: dict[str, object]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(CHECKER), "--artifact", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_scale01_checker_accepts_complete_matrix(self) -> None:
        result = self._check(_artifact())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_scale02_checker_rejects_missing_cell_bound_and_dirty_run(self) -> None:
        artifact = _artifact()
        cells = artifact["cells"]
        assert isinstance(cells, list)
        cells.pop()
        cells[0]["observed"]["peak_inflight_bytes"] = 1025
        artifact["run"]["git_dirty"] = True
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing matrix cells", result.stderr)
        self.assertIn("dirty", result.stderr)
        self.assertIn("inflight", result.stderr)

    def test_scale03_checker_rejects_path_leak_pass_count_and_fit_mismatch(self) -> None:
        artifact = _artifact()
        artifact["generator"]["temporary_root"] = "E:/private/input.csv"
        cells = artifact["cells"]
        assert isinstance(cells, list)
        cells[0]["observed"]["actual_pass_count"] = 2
        cells[0]["fit"]["event_count"] = 1
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("path", result.stderr)
        self.assertIn("pass", result.stderr)
        self.assertIn("fit", result.stderr)

    def test_scale04_runner_produces_checker_accepted_smoke_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--output",
                    str(output),
                    "--rows",
                    "100",
                    "--chunk-bytes",
                    "2048,4096,8192",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            accepted = subprocess.run(
                [sys.executable, str(CHECKER), "--artifact", str(output), "--smoke"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)


if __name__ == "__main__":
    unittest.main()
