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
_RAW_KIND = "v1-execution-raw"


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


def validate_raw_fragment(payload: object, expected_sha: str) -> list[str]:
    """Validate one actually executed platform fragment before matrix assembly.

    This stricter schema keeps a collector from turning an arbitrary JSON
    object into retained-looking evidence.  It deliberately accepts only one
    platform's nine real cells; :func:`validate` remains responsible for the
    final all-platform matrix.
    """

    errors: list[str] = []
    if _SHA.fullmatch(expected_sha) is None:
        return ["expected candidate SHA is invalid"]
    if not isinstance(payload, dict):
        return ["raw evidence root must be an object"]
    if payload.get("schema_version") != 1 or payload.get("artifact_kind") != _RAW_KIND:
        errors.append("unsupported raw execution evidence schema")
    platform = payload.get("platform")
    if platform not in _PLATFORMS:
        errors.append("raw evidence platform is invalid")
    if payload.get("candidate_git_sha") != expected_sha:
        errors.append("raw evidence candidate SHA does not match the reviewed candidate")
    if not isinstance(payload.get("host_platform"), str) or not payload["host_platform"].strip():
        errors.append("raw evidence lacks host-platform provenance")
    collected_at = payload.get("collected_at")
    if not isinstance(collected_at, str) or not collected_at.endswith("Z"):
        errors.append("raw evidence lacks UTC collection provenance")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return [*errors, "raw evidence cells must be a list"]
    expected = (
        {(platform, rows, scenario) for rows in _ROWS for scenario in _SCENARIOS}
        if isinstance(platform, str)
        else set()
    )
    observed: set[tuple[object, object, object]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"raw cell {index} must be an object")
            continue
        key = (cell.get("platform"), cell.get("rows"), cell.get("scenario"))
        if not (
            isinstance(key[0], str) and isinstance(key[1], int) and isinstance(key[2], str)
        ):
            errors.append(f"raw cell {index} has an invalid matrix key")
            continue
        if key in observed:
            errors.append(f"duplicate raw evidence cell: {key!r}")
        observed.add(key)
        if cell.get("candidate_git_sha") != expected_sha:
            errors.append(f"raw cell {index} is not bound to the reviewed candidate")
        if not _valid_positive_int(cell.get("source_bytes")):
            errors.append(f"raw cell {index} lacks a measured source size")
        source_sha = cell.get("source_sha256")
        if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha) is None:
            errors.append(f"raw cell {index} lacks a deterministic source digest")
        actual_passes = cell.get("actual_passes")
        scenario = cell.get("scenario")
        required_passes = 2 if scenario == "retry_resume" else 1
        if actual_passes != required_passes:
            errors.append(f"raw cell {index} has an invalid observed pass count")
        if not _valid_positive_int(cell.get("peak_rss_bytes")):
            errors.append(f"raw cell {index} has no measured RSS")
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
            errors.append(f"raw cell {index} has invalid inflight-byte observation")
        if scenario == "complete" and cell.get("result_code") != "COMPLETE":
            errors.append(f"raw cell {index} lacks a complete result")
        if scenario == "retry_resume" and not (
            cell.get("result_code") == "COMPLETE"
            and cell.get("retry_initial_code") == "CANCELLED"
            and cell.get("cancellation_observed") is True
            and cell.get("canonical_result_equal") is True
        ):
            errors.append(f"raw cell {index} lacks retry/resume facts")
        if scenario == "cancel" and not (
            cell.get("result_code") == "CANCELLED"
            and cell.get("cancellation_observed") is True
            and cell.get("resources_released") is True
        ):
            errors.append(f"raw cell {index} lacks cancellation cleanup facts")
    if observed != expected:
        errors.append("raw execution evidence matrix is incomplete or contains unknown cells")
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
