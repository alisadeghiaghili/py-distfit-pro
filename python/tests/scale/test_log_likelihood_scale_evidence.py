"""Fail-closed contracts for exact-state likelihood scale evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veridist.families.registry import FamilyId

REPO = Path(__file__).parents[3]
TOOLS = REPO / "python" / "tools"
CHECKER = TOOLS / "check_log_likelihood_scale_evidence.py"
RUNNER = TOOLS / "run_log_likelihood_scale_evidence.py"
SPEC = importlib.util.spec_from_file_location("likelihood_scale_checker", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RUNNER_SPEC = importlib.util.spec_from_file_location("likelihood_scale_runner", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


class LogLikelihoodScaleEvidenceTests(unittest.TestCase):
    def _value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "2",
            "run": {"git_sha": _head(), "git_dirty": False, "generator": "normal-zero-v1"},
            "cells": [],
        }
        for rows in MODULE.ROWS:
            for budget in MODULE.BUDGETS:
                units = MODULE._units(rows)
                expected = float(__import__("fractions").Fraction(units, 1 << 1074))
                value["cells"].append(
                    {
                        "rows": rows,
                        "chunk_size": budget,
                        "one_pass": {"iterator_acquisitions": 1, "observation_yields": rows},
                        "oracle": {
                            "oracle_total_units": units,
                            "oracle_total_units_bit_length": abs(units).bit_length(),
                            "bound_bits": 2162,
                        },
                        "actual": {
                            "observation_count": rows,
                            "total_log_likelihood": expected,
                            "total_log_likelihood_hex": expected.hex(),
                        },
                        "elapsed_seconds": 0.0,
                        "memory": {"tracemalloc_peak_bytes": 0},
                    }
                )
        value["artifact_sha256"] = MODULE._digest(value)
        return value

    def test_runner_rejects_a_correct_count_with_a_wrong_returned_total(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodSuccess

        def wrong_total(*_args: object, **_kwargs: object) -> LogLikelihoodSuccess:
            return LogLikelihoodSuccess(FamilyId.NORMAL, "0" * 64, 10, 0.0)

        with patch.object(RUNNER_MODULE, "reduce_log_likelihood_chunks", side_effect=wrong_total):
            with self.assertRaises(RuntimeError):
                RUNNER_MODULE._cell(10, 1)

    def test_runner_rejects_a_second_outer_iterator_acquisition(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodSuccess

        def second_pass(chunks: object, *_args: object, **_kwargs: object) -> LogLikelihoodSuccess:
            list(chunks)  # type: ignore[arg-type]
            list(chunks)  # type: ignore[arg-type]
            return LogLikelihoodSuccess(FamilyId.NORMAL, "0" * 64, 10, -1.0)

        with patch.object(RUNNER_MODULE, "reduce_log_likelihood_chunks", side_effect=second_pass):
            with self.assertRaises(RuntimeError):
                RUNNER_MODULE._cell(10, 1)

    def test_checker_rejects_tampered_actual_returned_total(self) -> None:
        value = self._value()
        cell = value["cells"][0]
        assert isinstance(cell, dict)
        actual = cell["actual"]
        assert isinstance(actual, dict)
        actual["total_log_likelihood"] = 0.0
        actual["total_log_likelihood_hex"] = (0.0).hex()
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "actual returned total does not match independent exact oracle",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

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
