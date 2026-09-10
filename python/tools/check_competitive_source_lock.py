"""Fail closed until every publishable competitive claim has an auditable lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def claim_id(row: dict[str, str]) -> str:
    """Derive a stable claim-cell ID from the public comparison fields."""

    canonical = "\x1f".join(
        row.get(field, "").strip() for field in ("project", "language", "capability", "status")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate(matrix: Path, lock_file: Path) -> list[str]:
    errors: list[str] = []
    try:
        with matrix.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        payload: Any = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, csv.Error, json.JSONDecodeError):
        return ["source-lock inputs are unreadable"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return ["unsupported source-lock schema"]
    locks = payload.get("locks")
    if not isinstance(locks, list):
        return ["source-lock records must be a list"]
    required = {
        claim_id(row) for row in rows if row.get("status") in {"supported", "not_supported"}
    }
    observed: set[str] = set()
    for record in locks:
        if not isinstance(record, dict):
            errors.append("source-lock record must be an object")
            continue
        identifier = record.get("claim_id")
        locator = record.get("source_locator")
        review = record.get("reviewed_at")
        if not isinstance(identifier, str) or len(identifier) != 64:
            errors.append("source-lock claim_id is invalid")
            continue
        if identifier in observed:
            errors.append("source-lock claim_id is duplicated")
        observed.add(identifier)
        if not isinstance(locator, str) or not locator:
            errors.append("source-lock locator is missing")
        if not isinstance(review, str) or not review:
            errors.append("source-lock review date is missing")
    if observed != required:
        errors.append("source-lock coverage is below 100 percent")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--locks", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.matrix, args.locks)
    if errors:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS: competitive claim source locks cover every publishable cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
