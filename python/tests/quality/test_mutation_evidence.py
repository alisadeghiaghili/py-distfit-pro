"""TDD contracts for cache-bound mutation evidence v2."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PYTHON_ROOT / "tools"))
from mutation_evidence import (  # noqa: E402
    CRITICAL_MODULES,
    MUTMUT_WHEEL_SHA256,
    config_digest,
    source_tree_digest,
)

CHECKER = PYTHON_ROOT / "tools" / "check_mutation_evidence.py"
RUNNER = PYTHON_ROOT / "tools" / "run_mutation.py"
COUNT_KEYS = ("generated", "killed", "survived", "unresolved")


def command(exit_code: int = 0) -> dict[str, object]:
    return {
        "state": "passed" if exit_code == 0 else "failed",
        "command": ["python", "-m", "pytest", "tests"],
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "exit_code": exit_code,
        "log_path": "fixture.log",
        "log_sha256": "0" * 64,
    }


def fixture(root: Path) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for module in CRITICAL_MODULES:
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
                "unresolved": 0,
                "mutants": [
                    {
                        "id": f"{name}::value__mutmut_1",
                        "cache_key": "value__mutmut_1",
                        "exit_code": 1,
                        "official_status": "killed",
                        "scoring_status": "killed",
                    }
                ],
                "function_hashes": {},
                "type_check_errors": {},
                "durations": {},
                "estimated_durations": {},
            }
        )
    (root / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'source_paths = ["src/veridist/domain", "src/veridist/statistics", '
        '"src/veridist/families", "src/veridist/engine"]\n'
        'pytest_add_cli_args_test_selection = ["tests"]\n'
        'also_copy = ["tools"]\nmutate_only_covered_lines = false\n',
        encoding="utf-8",
    )
    (root / "quality").mkdir()
    (root / "quality" / "mutation-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "production_root": "src/veridist",
                "critical_modules": list(CRITICAL_MODULES),
                "mutmut_version": "3.7.0",
                "minimum_score": 0.8,
                "pytest_selection": ["tests"],
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": 2,
        "source": {
            "commit": "fixture",
            "tree_sha256": source_tree_digest(root),
            "cache_sha256": "fixture",
        },
        "config": {
            "mutmut_version": "3.7.0",
            "wheel_sha256": MUTMUT_WHEEL_SHA256,
            "config_sha256": config_digest(root),
            "source_paths": [f"src/veridist/{name}" for name in CRITICAL_MODULES],
            "pytest_selection": ["tests"],
            "also_copy": ["tools"],
        },
        "environment": {"python": "fixture", "platform": "fixture"},
        "provenance": {
            "inputs": {},
            "input_digest": "fixture",
            "pre_input_digest": "fixture",
            "post_input_digest": "fixture",
        },
        "baseline": command(),
        "mutation": {**command(), "command": ["mutmut", "run"]},
        "files": reports,
        "modules": [
            {
                "module": module,
                **{key: 1 if key in {"generated", "killed"} else 0 for key in COUNT_KEYS},
            }
            for module in CRITICAL_MODULES
        ],
        "totals": {key: 4 if key in {"generated", "killed"} else 0 for key in COUNT_KEYS},
        "score": 1.0,
    }


def check(root: Path, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    evidence = root / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(root),
            "--evidence",
            str(evidence),
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

    def test_rejects_status_score_and_schema_tampering(self) -> None:
        for change in ("bool", "wrong-status", "extra-key", "unresolved", "score"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root)
                report = payload["files"][0]  # type: ignore[index]
                if change == "bool":
                    report["killed"] = True
                elif change == "wrong-status":
                    report["mutants"][0]["official_status"] = "survived"
                elif change == "extra-key":
                    report["extra"] = 1
                elif change == "unresolved":
                    report["mutants"][0]["exit_code"] = None
                    report["mutants"][0]["official_status"] = "not_checked"
                    report["mutants"][0]["scoring_status"] = "unresolved"
                    report["killed"], report["unresolved"] = 0, 1
                    payload["totals"]["killed"], payload["totals"]["unresolved"] = 3, 1
                    payload["modules"][0]["killed"], payload["modules"][0]["unresolved"] = 0, 1
                else:
                    payload["score"] = True
                self.assertNotEqual(check(root, payload).returncode, 0)

    def test_runner_refuses_native_windows_before_mutmut_execution(self) -> None:
        launcher = (
            "import platform, runpy, sys; "
            "platform.system = lambda: 'Windows'; "
            f"sys.path.insert(0, {str(PYTHON_ROOT / 'tools')!r}); "
            f"sys.argv = [{str(RUNNER)!r}, '--project-root', {str(PYTHON_ROOT)!r}, "
            "'--output', 'ignored.json']; "
            f"runpy.run_path({str(RUNNER)!r}, run_name='__main__')"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                launcher,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires POSIX/fork", result.stderr)
