"""Fail-closed validator for versioned formal mutation evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mutation_evidence import (
    ALLOWED_STATUSES,
    CRITICAL_MODULES,
    SCHEMA_VERSION,
    UNRESOLVED_STATUSES,
    config_digest,
    source_files,
    source_tree_digest,
)


def fail(message: str) -> None:
    raise ValueError(message)


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer (not boolean)")
    return value


def commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        fail("cannot determine checked-out git commit")
    return result.stdout.strip()


def check(payload: dict[str, Any], project_root: Path, fixture: bool = False) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        fail("unsupported mutation evidence schema")
    files = source_files(project_root)
    if not files:
        fail("critical mutation scope is empty")
    source = payload.get("source")
    if not isinstance(source, dict):
        fail("missing source binding")
    expected_commit = "fixture" if fixture else commit(project_root)
    if source.get("commit") != expected_commit:
        fail("stale or mismatched source commit")
    if source.get("tree_sha256") != source_tree_digest(project_root, files):
        fail("stale or mismatched source tree digest")
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("mutmut_version") != "3.7.0":
        fail("mutmut 3.7.0 configuration binding is required")
    if config.get("config_sha256") != config_digest(project_root):
        fail("mutation configuration drift")
    if config.get("source_paths") != [f"src/veridist/{m}" for m in CRITICAL_MODULES]:
        fail("evidence does not bind complete critical scope")
    if config.get("pytest_selection") != ["tests"]:
        fail("evidence does not bind full test selection")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or not all(
        isinstance(environment.get(k), str) for k in ("python", "platform")
    ):
        fail("missing Python/platform environment binding")
    baseline = payload.get("baseline")
    if (
        not isinstance(baseline, dict)
        or baseline.get("passed") is not True
        or not isinstance(baseline.get("command"), list)
    ):
        fail("missing passing baseline result")
    reports = payload.get("files")
    if not isinstance(reports, list):
        fail("files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict) or not isinstance(report.get("path"), str):
            fail("invalid file report")
        path = report["path"]
        if path in by_path:
            fail("duplicate file report")
        by_path[path] = report
    if set(by_path) != set(files):
        fail("missing or extra critical source file report")
    total = {
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
    }
    identities: set[str] = set()
    for path in files:
        report = by_path[path]
        mutants = report.get("mutants")
        if not isinstance(mutants, list):
            fail(f"{path}: missing mutant identities")
        counts = {key: integer(report.get(key), f"{path}:{key}") for key in total}
        if counts["generated"] != len(mutants):
            fail(f"{path}: generated count does not equal mutant identities")
        observed = {key: 0 for key in total}
        for mutant in mutants:
            if not isinstance(mutant, dict) or not isinstance(mutant.get("id"), str):
                fail(f"{path}: invalid mutant identity")
            identity = mutant["id"]
            if not identity or identity in identities:
                fail("duplicate or empty mutant identity")
            identities.add(identity)
            status = mutant.get("status")
            if status not in ALLOWED_STATUSES:
                fail(f"{path}: unrecognized mutant status")
            observed[status] += 1
        if any(counts[key] != observed[key] for key in total if key != "generated"):
            fail(f"{path}: declared counts do not match identities")
        if counts["generated"] != sum(counts[key] for key in total if key != "generated"):
            fail(f"{path}: denominator manipulation")
        for key in total:
            total[key] += counts[key]
    declared = payload.get("totals")
    if not isinstance(declared, dict) or any(
        integer(declared.get(k), f"totals:{k}") != total[k] for k in total
    ):
        fail("totals do not match per-file reports")
    modules = payload.get("modules")
    if not isinstance(modules, list):
        fail("missing per-critical-module reports")
    module_reports = {item.get("module"): item for item in modules if isinstance(item, dict)}
    if set(module_reports) != set(CRITICAL_MODULES) or len(module_reports) != len(modules):
        fail("missing, duplicate, or extra critical module report")
    for module in CRITICAL_MODULES:
        report = module_reports[module]
        expected = {
            key: sum(by_path[path][key] for path in files if f"/{module}/" in path) for key in total
        }
        if any(integer(report.get(key), f"{module}:{key}") != expected[key] for key in total):
            fail(f"{module}: module totals do not match file reports")
    denominator = total["killed"] + total["survived"]
    if denominator == 0:
        fail("zero scored mutants is not eligible release evidence")
    score = total["killed"] / denominator
    if payload.get("score") != score:
        fail("score must be derived exactly from killed and survived")
    if score < 0.8:
        fail("mutation score below 80%")
    if any(total[status] for status in UNRESOLVED_STATUSES):
        fail("unresolved mutation status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            fail("evidence root must be an object")
        check(payload, args.project_root.resolve(), args.fixture)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"MUTATION EVIDENCE FAIL: {error}", file=sys.stderr)
        return 1
    print("MUTATION EVIDENCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
