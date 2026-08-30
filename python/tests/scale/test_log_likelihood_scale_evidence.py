"""Fail-closed contracts for exact-state likelihood scale evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).parents[3]
TOOLS = REPO / "python" / "tools"
CHECKER = TOOLS / "check_log_likelihood_scale_evidence.py"
RUNNER = TOOLS / "run_log_likelihood_scale_evidence.py"
SPEC = importlib.util.spec_from_file_location("likelihood_scale_checker", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


class LogLikelihoodScaleEvidenceTests(unittest.TestCase):
    def test_checker_rejects_tampered_exact_state(self) -> None:
        value = {
            "schema_version": "1",
            "run": {"git_sha": _head(), "git_dirty": False, "generator": "normal-zero-v1"},
            "cells": [],
        }
        for rows in MODULE.ROWS:
            for budget in MODULE.BUDGETS:
                units = MODULE._units(rows)
                value["cells"].append(
                    {
                        "rows": rows,
                        "chunk_size": budget,
                        "one_pass": True,
                        "state": {
                            "observation_count": rows,
                            "total_units": units,
                            "total_units_bit_length": abs(units).bit_length(),
                            "bound_bits": 2162,
                        },
                        "elapsed_seconds": 0.0,
                        "memory": {"tracemalloc_peak_bytes": 0},
                    }
                )
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertEqual(MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO), [])
        value["cells"][0]["state"]["total_units"] += 1
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "independent exact state oracle mismatch",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

    def test_runner_output_is_checker_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            generated = subprocess.run(
                [sys.executable, str(RUNNER), "--output", str(output)],
                cwd=REPO / "python",
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            checked = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--artifact",
                    str(output),
                    "--expected-git-sha",
                    _head(),
                    "--repo-root",
                    str(REPO),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))["cells"]), 9)
