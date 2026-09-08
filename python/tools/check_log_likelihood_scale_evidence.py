"""Fail closed on retained exact-state log-likelihood scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROWS = (10_000, 100_000, 1_000_000)
BUDGETS = (1_024, 8_192, 65_536)
SHA = re.compile(r"[0-9a-f]{40}")
KEYS = {"schema_version", "run", "cells", "artifact_sha256"}
CELL_KEYS = {"rows", "chunk_size", "one_pass", "oracle", "actual", "elapsed_seconds", "memory"}
ONE_PASS_KEYS = {"iterator_acquisitions", "observation_yields"}
ORACLE_KEYS = {"oracle_total_units", "oracle_total_units_bit_length", "bound_bits"}
ACTUAL_KEYS = {"observation_count", "total_log_likelihood", "total_log_likelihood_hex"}
MEMORY_KEYS = {"tracemalloc_peak_bytes"}


def _digest(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _units(rows: int) -> int:
    contribution = -0.5 * math.log(2.0 * math.pi)
    numerator, denominator = contribution.as_integer_ratio()
    return rows * numerator * ((1 << 1074) // denominator)


def validate(value: object, *, expected_git_sha: str, repo_root: Path) -> list[str]:
    """Return all violations; malformed or non-reproducible evidence fails closed."""

    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != KEYS:
        return ["artifact schema keys invalid"]
    if value["schema_version"] != "2" or value["artifact_sha256"] != _digest(value):
        errors.append("artifact version or digest invalid")
    run = value["run"]
    if not isinstance(run, dict) or set(run) != {"git_sha", "git_dirty", "generator"}:
        return [*errors, "run schema invalid"]
    if (
        run["git_sha"] != expected_git_sha
        or not isinstance(run["git_sha"], str)
        or SHA.fullmatch(run["git_sha"]) is None
    ):
        errors.append("frozen Git SHA mismatch")
    if run["git_dirty"] is not False or run["generator"] != "normal-zero-v1":
        errors.append("run metadata invalid")
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", expected_git_sha, "HEAD"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        errors.append("frozen Git SHA is not an ancestor")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != len(ROWS) * len(BUDGETS):
        return [*errors, "full 10k/100k/1m by three-chunk matrix required"]
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != CELL_KEYS:
            errors.append("cell schema invalid")
            continue
        rows, chunk = cell["rows"], cell["chunk_size"]
        if type(rows) is not int or type(chunk) is not int or (rows, chunk) in seen:
            errors.append("cell key invalid or duplicate")
            continue
        seen.add((rows, chunk))
        if rows not in ROWS or chunk not in BUDGETS:
            errors.append("cell scale or pass facts invalid")
        one_pass, oracle, actual, memory = (
            cell["one_pass"],
            cell["oracle"],
            cell["actual"],
            cell["memory"],
        )
        if not isinstance(one_pass, dict) or set(one_pass) != ONE_PASS_KEYS:
            errors.append("one-pass measurement schema invalid")
            continue
        if one_pass != {"iterator_acquisitions": 1, "observation_yields": rows}:
            errors.append("actual one-pass traversal mismatch")
        if not isinstance(oracle, dict) or set(oracle) != ORACLE_KEYS:
            errors.append("oracle schema invalid")
            continue
        if oracle != {
            "oracle_total_units": _units(rows),
            "oracle_total_units_bit_length": abs(_units(rows)).bit_length(),
            "bound_bits": 2162,
        }:
            errors.append("independent exact oracle mismatch")
        expected = float(Fraction(_units(rows), 1 << 1074))
        if not isinstance(actual, dict) or set(actual) != ACTUAL_KEYS:
            errors.append("actual result schema invalid")
            continue
        if (
            actual["observation_count"] != rows
            or type(actual["total_log_likelihood"]) is not float
            or actual["total_log_likelihood_hex"] != expected.hex()
            or actual["total_log_likelihood"].hex() != expected.hex()
        ):
            errors.append("actual returned total does not match independent exact oracle")
        if (
            not isinstance(memory, dict)
            or set(memory) != MEMORY_KEYS
            or type(memory["tracemalloc_peak_bytes"]) is not int
            or memory["tracemalloc_peak_bytes"] < 0
        ):
            errors.append("descriptive memory fact invalid")
        if (
            not isinstance(cell["elapsed_seconds"], (int, float))
            or isinstance(cell["elapsed_seconds"], bool)
            or not math.isfinite(cell["elapsed_seconds"])
            or cell["elapsed_seconds"] < 0
        ):
            errors.append("descriptive elapsed fact invalid")
    if seen != {(row, budget) for row in ROWS for budget in BUDGETS}:
        errors.append("matrix coverage invalid")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        value: object = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    errors = validate(value, expected_git_sha=args.expected_git_sha, repo_root=args.repo_root)
    if errors:
        print("FAIL: log-likelihood scale evidence rejected", file=sys.stderr)
        print(*[f"- {error}" for error in errors], sep="\n", file=sys.stderr)
        return 1
    print("PASS: log-likelihood scale evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
