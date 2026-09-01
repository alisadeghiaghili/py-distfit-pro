"""Red contracts for formal mutation evidence schema v2 fail-closed behavior."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).parents[2]
CHECKER = PYTHON_ROOT / "tools" / "check_mutation_evidence.py"


class MutationEvidenceV2Contracts(unittest.TestCase):
    def test_checker_exposes_duplicate_and_nonfinite_safe_json_loader(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import check_mutation_evidence

        for source in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    check_mutation_evidence.load_json(source)

    def test_schema_v2_is_explicit_and_rejects_legacy_self_consistent_evidence(self) -> None:
        self.assertIn("SCHEMA_VERSION = 2", CHECKER.read_text(encoding="utf-8"))

    def test_official_mutmut_status_mapping_is_complete_and_not_caller_controlled(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        expected = {
            0: "survived",
            1: "killed",
            3: "killed",
            5: "no_tests",
            36: "timeout",
            37: "type_check",
            -11: "segfault",
            None: "not_checked",
        }
        for code, status in expected.items():
            with self.subTest(code=code):
                self.assertEqual(mutation_evidence.official_status(code), status)
        self.assertEqual(mutation_evidence.scoring_status(0), "survived")
        self.assertEqual(mutation_evidence.scoring_status(37), "killed")

    def test_checker_requires_raw_cache_for_non_fixture_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text("[tool.mutmut]\n", encoding="utf-8")
            evidence = root / "evidence.json"
            evidence.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--project-root",
                    str(root),
                    "--evidence",
                    str(evidence),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--mutants-root", result.stderr)

    def test_runner_provenance_contracts_are_present(self) -> None:
        runner = (PYTHON_ROOT / "tools" / "run_mutation.py").read_text(encoding="utf-8")
        required_terms = (
            "--mutmut-wheel",
            "git status --porcelain",
            "input_digest",
            "baseline",
            "mutation",
        )
        for required in required_terms:
            with self.subTest(required=required):
                self.assertIn(required, runner)
