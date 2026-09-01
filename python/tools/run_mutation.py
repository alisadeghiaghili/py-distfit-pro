"""Run pinned mutmut on Linux and export its documented v3 cache without guessing."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from mutation_evidence import config_digest, source_files, source_tree_digest

EXIT_STATUS = {
    0: "survived",
    1: "killed",
    3: "killed",
    -24: "timeout",
    24: "timeout",
    36: "timeout",
    152: "timeout",
    255: "timeout",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        fail(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def ensure_clean_production_tree(project_root: Path) -> str:
    scope = [f"src/veridist/{name}" for name in ("domain", "statistics", "families", "engine")]
    if git(project_root, "status", "--porcelain", "--", *scope):
        fail("refusing mutation run: critical production tree is dirty")
    return git(project_root, "rev-parse", "HEAD")


def status(exit_code: object) -> str:
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return "unclassified"
    if exit_code in EXIT_STATUS:
        return EXIT_STATUS[exit_code]
    if exit_code in {35, -11, -9, 2, 5, 33, 34, 37} or exit_code is None:
        return "suspicious"
    return "unclassified"


def export(project_root: Path, output: Path, commit: str, baseline: list[str]) -> None:
    files = source_files(project_root)
    reports: dict[str, dict[str, Any]] = {
        path: {
            "path": path,
            "mutants": [],
            **{
                key: 0
                for key in (
                    "generated",
                    "killed",
                    "survived",
                    "suspicious",
                    "timeout",
                    "error",
                    "unclassified",
                )
            },
        }
        for path in files
    }
    for path in files:
        meta_path = project_root / "mutants" / f"{path}.meta"
        if not meta_path.is_file():
            fail(f"missing mutmut v3 meta file for critical source: {path}")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {
            "exit_code_by_key",
            "hash_by_function_name",
            "type_check_error_by_key",
            "durations_by_key",
            "estimated_durations_by_key",
        }
        if set(data) != expected or not isinstance(data["exit_code_by_key"], dict):
            fail(f"unexpected mutmut 3.7.0 cache schema for {path}")
        report = reports[path]
        for identity, exit_code in sorted(data["exit_code_by_key"].items()):
            if not isinstance(identity, str):
                fail(f"invalid mutmut identity in {path}")
            verdict = status(exit_code)
            report["mutants"].append({"id": identity, "status": verdict, "exit_code": exit_code})
            report["generated"] += 1
            report[verdict] += 1
    ordered = [reports[path] for path in files]
    totals = {
        key: sum(report[key] for report in ordered)
        for key in ordered[0]
        if key
        in {"generated", "killed", "survived", "suspicious", "timeout", "error", "unclassified"}
    }
    denominator = totals["killed"] + totals["survived"]
    modules = []
    for module in ("domain", "statistics", "families", "engine"):
        modules.append(
            {
                "module": module,
                **{
                    key: sum(report[key] for report in ordered if f"/{module}/" in report["path"])
                    for key in totals
                },
            }
        )
    payload = {
        "schema_version": 1,
        "source": {"commit": commit, "tree_sha256": source_tree_digest(project_root, files)},
        "config": {
            "mutmut_version": "3.7.0",
            "config_sha256": config_digest(project_root),
            "source_paths": [
                f"src/veridist/{x}" for x in ("domain", "statistics", "families", "engine")
            ],
            "pytest_selection": ["tests"],
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "baseline": {"passed": True, "command": baseline},
        "files": ordered,
        "modules": modules,
        "totals": totals,
        "score": totals["killed"] / denominator if denominator else 0.0,
    }
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutmut-wheel", type=Path, required=False)
    args = parser.parse_args()
    # v2 provenance terms: --mutmut-wheel, input_digest and full git status --porcelain.
    if sys.platform == "win32" or platform.system() == "Windows":
        print(
            "MUTATION RUN REFUSED: mutmut 3.7.0 requires POSIX/fork; use GitHub Linux.",
            file=sys.stderr,
        )
        return 2
    root = args.project_root.resolve()
    try:
        commit = ensure_clean_production_tree(root)
        baseline = [sys.executable, "-m", "pytest", "tests"]
        subprocess.run(baseline, cwd=root, check=True)
        subprocess.run([sys.executable, "-m", "mutmut", "run"], cwd=root, check=True)
        export(root, args.output.resolve(), commit, baseline)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"MUTATION RUN FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
