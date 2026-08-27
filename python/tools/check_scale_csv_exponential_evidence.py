"""Fail closed on retained SCALE-CSV-EXP evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
FULL_ROWS = (10_000, 100_000, 1_000_000)
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _contains_path(value: object) -> bool:
    if isinstance(value, str):
        return "/" in value or "\\" in value or re.search(r"[A-Za-z]:", value) is not None
    if isinstance(value, list):
        return any(_contains_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_path(item) for item in value.values())
    return False


def _digest(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _cell_key(cell: dict[str, Any]) -> tuple[int, int] | None:
    rows, chunk = cell.get("rows"), cell.get("chunk_bytes")
    if not _integer(rows) or not _integer(chunk) or chunk == 0:
        return None
    return rows, chunk


def validate(value: object, *, smoke: bool = False) -> list[str]:
    """Return all invariant violations; malformed artifacts never partly pass."""

    errors: list[str] = []
    if not isinstance(value, dict):
        return ["artifact must be an object"]
    required = {
        "schema_version",
        "run",
        "generator",
        "cells",
        "operation_evidence",
        "artifact_sha256",
    }
    missing = required.difference(value)
    if missing:
        return [f"artifact missing required fields: {', '.join(sorted(missing))}"]
    if value["schema_version"] != SCHEMA_VERSION:
        errors.append("unsupported schema version")
    if not isinstance(value["artifact_sha256"], str) or value["artifact_sha256"] != _digest(value):
        errors.append("artifact digest mismatch")
    if _contains_path(value):
        errors.append("path leaked into evidence artifact")
    run = value["run"]
    if not isinstance(run, dict):
        return [*errors, "run must be an object"]
    if not isinstance(run.get("git_sha"), str) or GIT_SHA.fullmatch(run["git_sha"]) is None:
        errors.append("run git SHA is invalid")
    if run.get("git_dirty") is not False:
        errors.append("run is dirty")
    python = run.get("python")
    if (
        not isinstance(python, dict)
        or python.get("implementation") != "CPython"
        or not isinstance(python.get("version"), str)
    ):
        errors.append("run Python metadata is invalid")
    if not isinstance(run.get("utc_started"), str) or not run["utc_started"].endswith("Z"):
        errors.append("run UTC timestamp is invalid")
    generator = value["generator"]
    if (
        not isinstance(generator, dict)
        or generator.get("formula_version") != "1"
        or generator.get("temporary_root") != "redacted"
    ):
        errors.append("generator contract is invalid")
    cells = value["cells"]
    if not isinstance(cells, list) or not cells:
        return [*errors, "cells must be a non-empty list"]
    keys: list[tuple[int, int]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} must be an object")
            continue
        key = _cell_key(cell)
        if key is None:
            errors.append(f"cell {index} has invalid rows/chunk")
            continue
        keys.append(key)
        rows, chunk = key
        if cell.get("max_inflight_bytes") != chunk:
            errors.append(f"cell {key} max inflight must equal configured chunk")
        source = cell.get("source")
        if (
            not isinstance(source, dict)
            or not _integer(source.get("bytes"))
            or not isinstance(source.get("sha256"), str)
            or SHA256.fullmatch(source["sha256"]) is None
        ):
            errors.append(f"cell {key} source hash facts invalid")
        observed = cell.get("observed")
        if not isinstance(observed, dict):
            errors.append(f"cell {key} observed facts invalid")
            continue
        if observed.get("actual_pass_count") != 1 or observed.get("max_passes") != 1:
            errors.append(f"cell {key} pass count must be exactly one")
        if observed.get("processed_row_count") != rows:
            errors.append(f"cell {key} observed rows mismatch")
        for metric in (
            "accepted_chunk_count",
            "peak_inflight_bytes",
            "largest_retained_chunk_bytes",
            "backpressure_event_count",
        ):
            if not _integer(observed.get(metric)):
                errors.append(f"cell {key} {metric} invalid")
        if (
            isinstance(observed.get("peak_inflight_bytes"), int)
            and observed["peak_inflight_bytes"] > chunk
        ):
            errors.append(f"cell {key} inflight bound exceeded")
        if (
            isinstance(observed.get("largest_retained_chunk_bytes"), int)
            and observed["largest_retained_chunk_bytes"] > chunk
        ):
            errors.append(f"cell {key} retained chunk bound exceeded")
        fit = cell.get("fit")
        if not isinstance(fit, dict):
            errors.append(f"cell {key} fit facts invalid")
            continue
        if fit.get("observation_count") != rows or not _integer(fit.get("event_count")):
            errors.append(f"cell {key} fit count mismatch")
        if fit.get("event_count") != fit.get("expected_event_count"):
            errors.append(f"cell {key} fit event mismatch")
        try:
            expected = Decimal(str(fit.get("expected_rate")))
            actual = Decimal(str(fit.get("rate")))
            expected_total = Decimal(str(fit.get("expected_total_time")))
            total = Decimal(str(fit.get("total_time")))
            if expected <= 0 or actual <= 0 or expected_total <= 0 or total <= 0:
                raise InvalidOperation
            if Decimal(fit["event_count"]) / expected_total != expected:
                errors.append(f"cell {key} expected fit facts inconsistent")
            if abs(actual - expected) > Decimal("1e-12") * max(Decimal(1), abs(expected)):
                errors.append(f"cell {key} fit rate mismatch")
        except (InvalidOperation, ValueError, TypeError, KeyError):
            errors.append(f"cell {key} fit numeric facts invalid")
        for metric in ("absolute_rate_error", "relative_rate_error"):
            if not _finite_nonnegative(fit.get(metric)):
                errors.append(f"cell {key} {metric} invalid")
        memory = cell.get("memory")
        if not isinstance(memory, dict) or not _integer(memory.get("tracemalloc_peak_bytes")):
            errors.append(f"cell {key} memory facts invalid")
        elif memory.get("rss_peak_bytes") is not None and not _integer(
            memory.get("rss_peak_bytes")
        ):
            errors.append(f"cell {key} RSS peak invalid")
        if (
            memory is not None
            and isinstance(memory, dict)
            and memory.get("rss_delta_bytes") is not None
            and not _integer(memory.get("rss_delta_bytes"))
        ):
            errors.append(f"cell {key} RSS delta invalid")
        if not _finite_nonnegative(cell.get("elapsed_seconds")):
            errors.append(f"cell {key} elapsed time invalid")
    if len(keys) != len(set(keys)):
        errors.append("duplicate matrix cells")
    row_set = {key[0] for key in keys}
    if smoke:
        if len(keys) != 3 or len(row_set) != 1 or len({key[1] for key in keys}) < 3:
            errors.append("smoke matrix must have one row size and three chunk budgets")
    else:
        budgets = {key[1] for key in keys}
        expected_keys = {(rows, budget) for rows in FULL_ROWS for budget in budgets}
        if row_set != set(FULL_ROWS) or len(budgets) < 3 or set(keys) != expected_keys:
            errors.append("missing matrix cells")
    operation = value["operation_evidence"]
    expected_operation_rows = sorted(row_set) if smoke else list(FULL_ROWS)
    if not isinstance(operation, dict) or operation.get("rows") != expected_operation_rows:
        errors.append("operation evidence rows invalid")
    else:
        counts = operation.get("accepted_chunks")
        if (
            not isinstance(counts, list)
            or len(counts) != len(expected_operation_rows)
            or not all(_integer(item) and item > 0 for item in counts)
        ):
            errors.append("operation evidence counts invalid")
        elif not smoke:
            densities = [count / row for count, row in zip(counts, FULL_ROWS, strict=True)]
            if max(densities) / min(densities) > 2.0:
                errors.append("operation evidence is not structurally near-linear")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: cannot read artifact: {error}", file=sys.stderr)
        return 1
    errors = validate(value, smoke=args.smoke)
    if errors:
        print("FAIL: scale evidence rejected", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("PASS: scale evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
