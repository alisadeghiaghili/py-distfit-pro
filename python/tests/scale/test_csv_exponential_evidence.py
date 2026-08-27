"""Adversarial contracts for the retained CSV/exponential scale evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).parents[2] / "tools"
CHECKER = TOOLS / "check_scale_csv_exponential_evidence.py"
RUNNER = TOOLS / "run_scale_csv_exponential_evidence.py"
REPO = Path(__file__).parents[3]
CHECKER_SPEC = importlib.util.spec_from_file_location("scale_checker", CHECKER)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
CHECKER_MODULE = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(CHECKER_MODULE)
RUNNER_SPEC = importlib.util.spec_from_file_location("scale_runner", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _seal(value: dict[str, object]) -> None:
    body = dict(value)
    body.pop("artifact_sha256", None)
    value["artifact_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _cell(rows: int, budget: int) -> dict[str, object]:
    events, total, byte_count, source_hash = CHECKER_MODULE._fixture_facts(rows, "1")
    expected_rate = Decimal(events) / total
    rate = float(expected_rate)
    actual_rate = Decimal(str(rate))
    absolute = abs(actual_rate - expected_rate)
    relative = absolute / abs(expected_rate)
    return {
        "rows": rows,
        "chunk_bytes": budget,
        "max_inflight_bytes": budget,
        "source": {"bytes": byte_count, "sha256": source_hash},
        "observed": {
            "actual_pass_count": 1,
            "max_passes": 1,
            "accepted_chunk_count": budget // 1024,
            "processed_row_count": rows,
            "peak_inflight_bytes": min(128, budget),
            "largest_retained_chunk_bytes": min(128, budget),
            "backpressure_event_count": 0,
        },
        "fit": {
            "observation_count": rows,
            "event_count": events,
            "total_time": float(total),
            "rate": rate,
            "expected_event_count": events,
            "expected_total_time": str(total),
            "expected_rate": str(expected_rate),
            "absolute_rate_error": float(absolute),
            "relative_rate_error": float(relative),
        },
        "memory": {"tracemalloc_peak_bytes": 1, "rss_peak_bytes": None, "rss_delta_bytes": None},
        "elapsed_seconds": 0.01,
    }


def _smoke_artifact() -> dict[str, object]:
    row, budgets = 100, (2048, 4096, 8192)
    cells = [_cell(row, budget) for budget in budgets]
    value: dict[str, object] = {
        "schema_version": "1",
        "run": {
            "git_sha": _head(),
            "git_dirty": False,
            "utc_started": "2026-08-27T00:00:00Z",
            "python": {"implementation": "CPython", "version": "3.11.0"},
            "platform": "test-platform",
            "measurement_workers": 1,
        },
        "generator": {"formula_version": "1", "temporary_root": "redacted"},
        "cells": cells,
        "operation_evidence": {"rows": [row], "accepted_chunks": [cells[0]["observed"]["accepted_chunk_count"]]},
    }
    _seal(value)
    return value


class ScaleCsvExponentialEvidenceTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        environment = dict(os.environ)
        source = str(Path(__file__).parents[2] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return environment

    def _check(self, artifact: dict[str, object], *, expected_sha: str | None = None, smoke: bool = True) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            command = [sys.executable, str(CHECKER), "--artifact", str(path), "--expected-git-sha", expected_sha or _head(), "--repo-root", str(REPO)]
            if smoke:
                command.append("--smoke")
            return subprocess.run(command, check=False, capture_output=True, text=True, env=self._environment())

    def test_scale01_checker_accepts_explicitly_locked_smoke_matrix(self) -> None:
        result = self._check(_smoke_artifact())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scale02_rejects_wrong_expected_sha_and_nonancestor(self) -> None:
        artifact = _smoke_artifact()
        result = self._check(artifact, expected_sha="0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen expected SHA", result.stderr)
        self.assertIn("existing ancestor", result.stderr)

    def test_scale03_rejects_matrix_cell_count_duplicate_and_wrong_operation_crosslink(self) -> None:
        artifact = _smoke_artifact()
        cells = artifact["cells"]
        assert isinstance(cells, list)
        cells.append(dict(cells[0]))
        artifact["operation_evidence"]["accepted_chunks"] = [999]
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly three", result.stderr)
        self.assertIn("duplicate", result.stderr)
        self.assertIn("cross-link", result.stderr)

    def test_scale04_rejects_zero_source_sha_and_fake_counts(self) -> None:
        artifact = _smoke_artifact()
        cells = artifact["cells"]
        assert isinstance(cells, list)
        cells[0]["source"]["sha256"] = "0" * 64
        cells[0]["fit"]["event_count"] = 1
        cells[0]["fit"]["expected_event_count"] = 1
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source bytes or SHA", result.stderr)
        self.assertIn("observation/event", result.stderr)

    def test_scale05_rejects_fake_total_and_rate_errors(self) -> None:
        artifact = _smoke_artifact()
        cell = artifact["cells"][0]
        cell["fit"]["total_time"] = 999999
        cell["fit"]["expected_total_time"] = "999999"
        cell["fit"]["absolute_rate_error"] = 999
        cell["fit"]["relative_rate_error"] = 999
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("total time", result.stderr)
        self.assertIn("rate errors", result.stderr)

    def test_scale06_rejects_extra_schema_key_and_path_key(self) -> None:
        artifact = _smoke_artifact()
        artifact["run"]["platform"] = "C:/secret"
        artifact["cells"][0]["fit"]["extra"] = 1
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema keys", result.stderr)
        self.assertIn("path", result.stderr)

    def test_scale07_rejects_bad_worker_and_dirty_run(self) -> None:
        artifact = _smoke_artifact()
        artifact["run"]["measurement_workers"] = 0
        artifact["run"]["git_dirty"] = True
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("workers", result.stderr)
        self.assertIn("dirty", result.stderr)

    def test_scale08_runner_produces_checker_accepted_smoke_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            command = [sys.executable, str(RUNNER), "--output", str(output), "--rows", "100", "--chunk-bytes", "2048,4096,8192", "--workers", "2"]
            generated = subprocess.run(command, check=False, capture_output=True, text=True, env=self._environment())
            self.assertEqual(generated.returncode, 0, generated.stderr)
            result = subprocess.run([sys.executable, str(CHECKER), "--artifact", str(output), "--expected-git-sha", _head(), "--repo-root", str(REPO), "--smoke"], check=False, capture_output=True, text=True, env=self._environment())
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_scale09_runner_smoke_is_concurrent_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / f"artifact-{index}.json" for index in range(2)]
            processes = [subprocess.Popen([sys.executable, str(RUNNER), "--output", str(output), "--rows", "100", "--chunk-bytes", "2048,4096,8192"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self._environment()) for output in outputs]
            results = [process.communicate() for process in processes]
            self.assertTrue(all(process.returncode == 0 for process in processes), results)
            self.assertTrue(all(output.exists() for output in outputs))

    def test_scale10_full_contract_rejects_twelve_cells(self) -> None:
        retained = Path(__file__).parents[2] / "evidence" / "scale-csv-exponential-v1.json"
        artifact = json.loads(retained.read_text(encoding="utf-8"))
        artifact["cells"].extend(artifact["cells"][:3])
        _seal(artifact)
        result = self._check(artifact, expected_sha="4490c9eb08e9ed5e420a2b677d9de843fdf66a5d", smoke=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly nine", result.stderr)
        self.assertIn("duplicate", result.stderr)

    def test_scale11_runner_refuses_output_when_head_changes_mid_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            facts = iter([_head(), "f" * 40])
            command = [str(RUNNER), "--output", str(output), "--rows", "10", "--chunk-bytes", "2048,4096,8192"]
            with patch.object(RUNNER_MODULE, "_clean_checkout_sha", side_effect=lambda _root: next(facts)), patch.object(sys, "argv", command):
                with self.assertRaises(SystemExit) as failure:
                    RUNNER_MODULE.main()
            self.assertIn("HEAD changed", str(failure.exception))
            self.assertFalse(output.exists())

    def test_scale12_retained_artifact_is_locked_and_accepted(self) -> None:
        artifact = Path(__file__).parents[2] / "evidence" / "scale-csv-exponential-v1.json"
        result = subprocess.run([sys.executable, str(CHECKER), "--artifact", str(artifact), "--expected-git-sha", "4490c9eb08e9ed5e420a2b677d9de843fdf66a5d", "--repo-root", str(REPO)], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
