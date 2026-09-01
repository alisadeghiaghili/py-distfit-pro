"""TDD contracts for mutation schema, checker and Windows runner refusal."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PYTHON_ROOT / "tools"))
from mutation_evidence import config_digest, source_tree_digest  # noqa: E402

CHECKER = PYTHON_ROOT / "tools" / "check_mutation_evidence.py"
RUNNER = PYTHON_ROOT / "tools" / "run_mutation.py"
COUNT_KEYS = ("generated", "killed", "survived", "suspicious", "timeout", "error", "unclassified")


def fixture(root: Path) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for module in ("domain", "statistics", "families", "engine"):
        name = f"src/veridist/{module}/sample.py"
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def value():\n    return 1\n", encoding="utf-8")
        reports.append(
            {
                "path": name,
                "generated": 1,
                "killed": 1,
                "survived": 0,
                "suspicious": 0,
                "timeout": 0,
                "error": 0,
                "unclassified": 0,
                "mutants": [{"id": f"{module}.value__mutmut_1", "status": "killed"}],
            }
        )
    (root / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'source_paths = ["src/veridist/domain", "src/veridist/statistics", '
        '"src/veridist/families", "src/veridist/engine"]\n'
        'pytest_add_cli_args_test_selection = ["tests"]\nmutate_only_covered_lines = false\n',
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "source": {"commit": "fixture", "tree_sha256": source_tree_digest(root)},
        "config": {
            "mutmut_version": "3.7.0",
            "config_sha256": config_digest(root),
            "source_paths": [
                f"src/veridist/{name}" for name in ("domain", "statistics", "families", "engine")
            ],
            "pytest_selection": ["tests"],
        },
        "environment": {"python": "fixture", "platform": "fixture"},
        "baseline": {"passed": True, "command": ["pytest", "tests"]},
        "files": reports,
        "modules": [
            {
                "module": module,
                **{key: 1 if key in {"generated", "killed"} else 0 for key in COUNT_KEYS},
            }
            for module in ("domain", "statistics", "families", "engine")
        ],
        "totals": {key: 4 if key in {"generated", "killed"} else 0 for key in COUNT_KEYS},
        "score": 1.0,
    }


def check(root: Path, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    path = root / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(root),
            "--evidence",
            str(path),
            "--fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class MutationEvidenceTests(unittest.TestCase):
    def test_accepts_complete_deterministic_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = check(root, fixture(root))
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_tampering(self) -> None:
        for change in ("missing", "bool", "duplicate", "unresolved"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root)
                reports = payload["files"]
                assert isinstance(reports, list)
                if change == "missing":
                    payload["files"] = reports[:-1]
                elif change == "bool":
                    reports[0]["killed"] = True
                elif change == "duplicate":
                    reports[1]["mutants"][0]["id"] = reports[0]["mutants"][0]["id"]
                else:
                    reports[0]["mutants"][0]["status"] = "timeout"
                    reports[0]["killed"] = 0
                    reports[0]["timeout"] = 1
                    payload["totals"]["killed"] = 3
                    payload["totals"]["timeout"] = 1
                self.assertNotEqual(check(root, payload).returncode, 0)

    def test_runner_refuses_native_windows_before_mutmut_execution(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--project-root",
                str(PYTHON_ROOT),
                "--output",
                "ignored.json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires POSIX/fork", result.stderr)
