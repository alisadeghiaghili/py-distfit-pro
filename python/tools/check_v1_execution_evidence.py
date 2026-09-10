"""Fail closed for retained v1 retry/cancel/resume scale evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{40}$")
_PLATFORMS = frozenset({"linux", "macos", "windows"})
_ROWS = frozenset({10_000, 100_000, 1_000_000})
_SCENARIOS = frozenset({"complete", "retry_resume", "cancel"})


def _valid_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate(payload: object, expected_sha: str) -> list[str]:
    """Return schema and candidate-binding violations without accepting partial evidence."""

    errors: list[str] = []
    if _SHA.fullmatch(expected_sha) is None:
        return ["expected candidate SHA is invalid"]
    if not isinstance(payload, dict):
        return ["evidence root must be an object"]
    if payload.get("schema_version") != 1:
        errors.append("unsupported execution evidence schema")
    if payload.get("candidate_git_sha") != expected_sha:
        errors.append("evidence candidate SHA does not match the reviewed candidate")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return [*errors, "evidence cells must be a list"]
    expected = {
        (platform, rows, scenario)
        for platform in _PLATFORMS
        for rows in _ROWS
        for scenario in _SCENARIOS
    }
    observed: set[tuple[object, object, object]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} must be an object")
            continue
        key = (cell.get("platform"), cell.get("rows"), cell.get("scenario"))
        if key in observed:
            errors.append(f"duplicate evidence cell: {key!r}")
        observed.add(key)
        if cell.get("candidate_git_sha") != expected_sha:
            errors.append(f"cell {index} is not bound to the reviewed candidate")
        if not _valid_positive_int(cell.get("actual_passes")):
            errors.append(f"cell {index} has no measured pass count")
        if not _valid_positive_int(cell.get("peak_rss_bytes")):
            errors.append(f"cell {index} has no measured RSS")
        maximum, observed_bytes = (
            cell.get("max_inflight_bytes"),
            cell.get("observed_inflight_bytes"),
        )
        if not (
            isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and maximum > 0
            and isinstance(observed_bytes, int)
            and not isinstance(observed_bytes, bool)
            and 0 <= observed_bytes <= maximum
        ):
            errors.append(f"cell {index} has invalid inflight-byte observation")
        if (
            cell.get("scenario") == "retry_resume"
            and cell.get("canonical_result_equal") is not True
        ):
            errors.append(f"cell {index} lacks retry/resume equality evidence")
        if cell.get("scenario") == "cancel" and cell.get("resources_released") is not True:
            errors.append(f"cell {index} lacks cancellation cleanup evidence")
    if observed != expected:
        errors.append("execution evidence matrix is incomplete or contains unknown cells")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    args = parser.parse_args()
    try:
        payload: Any = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("FAIL: execution evidence artifact is unreadable", file=sys.stderr)
        return 1
    errors = validate(payload, args.expected_git_sha)
    if errors:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS: v1 execution evidence is complete and candidate-bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
