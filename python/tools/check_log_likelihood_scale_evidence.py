"""Fail closed on retained exact-state log-likelihood scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROWS = (10_000, 100_000, 1_000_000)
BUDGETS = (1_024, 8_192, 65_536)
SHA = re.compile(r"[0-9a-f]{40}")
KEYS = {"schema_version", "run", "cells", "artifact_sha256"}
CELL_KEYS = {"rows", "chunk_size", "one_pass", "state", "elapsed_seconds", "memory"}
STATE_KEYS = {"observation_count", "total_units", "total_units_bit_length", "bound_bits"}
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
    if value["schema_version"] != "1" or value["artifact_sha256"] != _digest(value):
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
        if rows not in ROWS or chunk not in BUDGETS or cell["one_pass"] is not True:
            errors.append("cell scale or pass facts invalid")
        state, memory = cell["state"], cell["memory"]
        if not isinstance(state, dict) or set(state) != STATE_KEYS:
            errors.append("state schema invalid")
            continue
        if state != {
            "observation_count": rows,
            "total_units": _units(rows),
            "total_units_bit_length": abs(_units(rows)).bit_length(),
            "bound_bits": 2162,
        }:
            errors.append("independent exact state oracle mismatch")
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
