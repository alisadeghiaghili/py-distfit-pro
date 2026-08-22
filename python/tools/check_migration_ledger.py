"""Validate the evidence-gated legacy migration ledger without dependencies."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
MIGRATION_ROOT = REPOSITORY_ROOT / "docs" / "migration"
SCHEMA_PATH = MIGRATION_ROOT / "legacy-salvage-ledger.schema.json"
LEDGER_PATH = MIGRATION_ROOT / "legacy-salvage-ledger.json"
ALLOWED_DISPOSITIONS = frozenset({"modify_port", "rewrite", "archive"})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LedgerError(ValueError):
    """Raised when a ledger claim cannot be independently verified."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LedgerError(f"cannot read {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise LedgerError(f"{path.name} must contain a JSON object")
    return data


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise LedgerError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _require_strings(entry: dict[str, Any], key: str) -> list[str]:
    value = entry.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise LedgerError(f"{entry.get('component', '<unknown>')}: {key} must be a non-empty string list")
    return value


def validate() -> None:
    schema = _read_json(SCHEMA_PATH)
    ledger = _read_json(LEDGER_PATH)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise LedgerError("schema must declare JSON Schema Draft 2020-12")
    if ledger.get("schema_version") != "1.0":
        raise LedgerError("unsupported ledger schema_version")
    if ledger.get("target_namespace") != "veridist" or ledger.get("legacy_package") != "distfit_pro":
        raise LedgerError("only veridist target and distfit_pro legacy package are allowed")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LedgerError("entries must be a non-empty list")
    expected_commit = _git("rev-parse", "origin/main")
    seen_components: set[str] = set()
    seen_dispositions: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise LedgerError("each entry must be an object")
        component = entry.get("component")
        if not isinstance(component, str) or not component or component in seen_components:
            raise LedgerError("components must be unique non-empty strings")
        seen_components.add(component)
        disposition = entry.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            raise LedgerError(f"{component}: unsupported disposition")
        seen_dispositions.add(disposition)
        if entry.get("license") != "MIT":
            raise LedgerError(f"{component}: license must be independently recorded as MIT")
        for key in ("evidence", "reviews", "limits"):
            _require_strings(entry, key)
        source = entry.get("source")
        if not isinstance(source, dict):
            raise LedgerError(f"{component}: source must be an object")
        commit, path, blob, sha256 = (source.get(key) for key in ("commit", "path", "blob", "sha256"))
        if not isinstance(commit, str) or not HEX40.fullmatch(commit):
            raise LedgerError(f"{component}: source.commit must be a 40-character hash")
        if commit != expected_commit:
            raise LedgerError(f"{component}: source.commit is stale versus origin/main")
        if not isinstance(path, str) or not path.startswith("distfit_pro/") or ".." in Path(path).parts:
            raise LedgerError(f"{component}: source.path must stay within distfit_pro")
        if not isinstance(blob, str) or not HEX40.fullmatch(blob):
            raise LedgerError(f"{component}: source.blob must be a 40-character hash")
        if not isinstance(sha256, str) or not HEX64.fullmatch(sha256):
            raise LedgerError(f"{component}: source.sha256 must be a 64-character hash")
        if _git("rev-parse", f"{commit}:{path}") != blob:
            raise LedgerError(f"{component}: source.blob does not match source.commit:path")
        payload = (REPOSITORY_ROOT / path).read_bytes()
        observed_hash = hashlib.sha256(payload).hexdigest()
        if observed_hash != sha256:
            raise LedgerError(f"{component}: source.sha256 is stale")
        if disposition == "modify_port" and "Candidate only" not in " ".join(entry["limits"]):
            raise LedgerError(f"{component}: modify_port must remain a candidate, not an approval")
        if component == "exponential" and disposition != "rewrite":
            raise LedgerError("exponential must be a rewrite")
    if seen_dispositions != ALLOWED_DISPOSITIONS:
        raise LedgerError("ledger must demonstrate every allowed disposition")


def main() -> int:
    try:
        validate()
    except LedgerError as error:
        print(f"migration ledger check failed: {error}", file=sys.stderr)
        return 1
    print("migration ledger check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
