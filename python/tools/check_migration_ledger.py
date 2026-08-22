"""Validate the evidence-gated legacy migration ledger without dependencies."""

from __future__ import annotations

import argparse
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
RELEASE = re.compile(r"^(?:unplanned|0\.[0-9]+\.[0-9]+(?:a[0-9]+)?)$")
REVIEW_VALUES = ["reviewed", "review_pending", "not_applicable"]


class LedgerError(ValueError):
    """Raised when a ledger claim cannot be independently verified."""


def _schema_value(schema: dict[str, Any], path: str) -> Any:
    value: Any = schema
    if not path:
        return value
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise LedgerError(f"schema contract missing {path}")
        value = value[part]
    return value


def _require_schema(schema: dict[str, Any], path: str, **expected: Any) -> None:
    value = _schema_value(schema, path)
    if not isinstance(value, dict) or any(
        value.get(key) != wanted for key, wanted in expected.items()
    ):
        raise LedgerError(f"schema contract drift at {path}")


def _validate_schema_contract(schema: dict[str, Any]) -> None:
    """Check the enforced Draft 2020-12 contract using only the standard library."""
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise LedgerError("schema must declare JSON Schema Draft 2020-12")
    _require_schema(schema, "", type="object", additionalProperties=False)
    properties = _schema_value(schema, "properties")
    required_root = {"schema_version", "target_namespace", "legacy_package", "entries"}
    if not isinstance(properties, dict) or set(properties) != required_root:
        raise LedgerError("schema contract drift at root properties")
    for name, constant in (
        ("schema_version", "1.0"),
        ("target_namespace", "veridist"),
        ("legacy_package", "distfit_pro"),
    ):
        _require_schema(schema, f"properties.{name}", type="string", const=constant)
    _require_schema(schema, "properties.entries", type="array", minItems=1)
    entry_path = "properties.entries.items"
    _require_schema(schema, entry_path, type="object", additionalProperties=False)
    entry = _schema_value(schema, entry_path)
    expected_fields = {
        "id",
        "component",
        "component_kind",
        "target",
        "status",
        "disposition",
        "source",
        "license",
        "evidence",
        "reviews",
        "isolation",
        "limits",
    }
    if (
        set(entry.get("required", [])) != expected_fields
        or set(entry.get("properties", {})) != expected_fields
    ):
        raise LedgerError("schema contract drift at entry fields")
    for name in ("target", "source", "license", "evidence", "reviews", "isolation"):
        nested_path = f"{entry_path}.properties.{name}"
        _require_schema(schema, nested_path, type="object", additionalProperties=False)
        nested = _schema_value(schema, nested_path)
        if set(nested.get("required", [])) != set(nested.get("properties", {})):
            raise LedgerError(f"schema contract drift at {name} fields")
    _require_schema(
        schema,
        f"{entry_path}.properties.source.properties.path",
        type="string",
        pattern="^(?:distfit_pro|tests|examples)/",
        minLength=1,
    )
    _require_schema(
        schema,
        f"{entry_path}.properties.target.properties.path",
        type="string",
        pattern="^python/(?:src/veridist|docs)/",
        minLength=1,
    )
    _require_schema(
        schema, f"{entry_path}.properties.target.properties.public_surface", type="boolean"
    )
    _require_schema(
        schema,
        f"{entry_path}.properties.target.properties.release",
        type="string",
        pattern=RELEASE.pattern,
    )
    for name in ("commit", "blob"):
        _require_schema(
            schema,
            f"{entry_path}.properties.source.properties.{name}",
            type="string",
            pattern="^[0-9a-f]{40}$",
        )
    _require_schema(
        schema,
        f"{entry_path}.properties.source.properties.sha256",
        type="string",
        pattern="^[0-9a-f]{64}$",
    )
    for name, constant in (("spdx", "MIT"), ("basis", "LICENSE")):
        _require_schema(
            schema,
            f"{entry_path}.properties.license.properties.{name}",
            type="string",
            const=constant,
        )
    for name in ("license", "statistical", "scale", "i18n"):
        _require_schema(
            schema,
            f"{entry_path}.properties.reviews.properties.{name}",
            type="string",
            enum=REVIEW_VALUES,
        )
    for name in (
        "runtime_import",
        "runtime_dependency",
        "dynamic_fallback",
        "package_content",
        "oracle",
    ):
        _require_schema(
            schema,
            f"{entry_path}.properties.isolation.properties.{name}",
            type="boolean",
            const=False,
        )
    for name in ("limits", "evidence.scenario_ids", "evidence.test_ids", "evidence.reference_ids"):
        schema_path = f"{entry_path}.properties.{name.replace('.', '.properties.')}"
        _require_schema(schema, schema_path, type="array", minItems=1)
        _require_schema(schema, f"{schema_path}.items", type="string", minLength=1)


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


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, check=False, capture_output=True
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise LedgerError(message or "git command failed")
    return result.stdout


def _require_available_commit(commit: str, component: str) -> None:
    try:
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    except LedgerError as error:
        raise LedgerError(f"{component}: source.commit unavailable: {commit}") from error


def _require_strings(entry: dict[str, Any], key: str) -> list[str]:
    value = entry.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise LedgerError(
            f"{entry.get('component', '<unknown>')}: {key} must be a non-empty string list"
        )
    return value


def validate(schema_path: Path = SCHEMA_PATH, ledger_path: Path = LEDGER_PATH) -> None:
    schema = _read_json(schema_path)
    ledger = _read_json(ledger_path)
    _validate_schema_contract(schema)
    if ledger.get("schema_version") != "1.0":
        raise LedgerError("unsupported ledger schema_version")
    if (
        ledger.get("target_namespace") != "veridist"
        or ledger.get("legacy_package") != "distfit_pro"
    ):
        raise LedgerError("only veridist target and distfit_pro legacy package are allowed")
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LedgerError("entries must be a non-empty list")
    seen_components: set[str] = set()
    seen_ids: set[str] = set()
    seen_dispositions: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise LedgerError("each entry must be an object")
        required = {
            "id",
            "component",
            "component_kind",
            "target",
            "status",
            "disposition",
            "source",
            "license",
            "evidence",
            "reviews",
            "isolation",
            "limits",
        }
        if set(entry) != required:
            raise LedgerError("entry keys must exactly match the ledger contract")
        identifier = entry.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"LM-[0-9]{3}", identifier)
            or identifier in seen_ids
        ):
            raise LedgerError("entries require unique stable LM-xxx ids")
        seen_ids.add(identifier)
        component = entry.get("component")
        if not isinstance(component, str) or not component or component in seen_components:
            raise LedgerError("components must be unique non-empty strings")
        seen_components.add(component)
        disposition = entry.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            raise LedgerError(f"{component}: unsupported disposition")
        seen_dispositions.add(disposition)
        status = entry.get("status")
        if disposition == "archive" and status != "archived":
            raise LedgerError(f"{component}: archive disposition requires archived status")
        if disposition in {"rewrite", "modify_port"} and status != "review_pending":
            raise LedgerError(f"{component}: active disposition requires review_pending status")
        if entry.get("license") != {"spdx": "MIT", "basis": "LICENSE"}:
            raise LedgerError(f"{component}: license requires SPDX MIT and LICENSE basis")
        _require_strings(entry, "limits")
        target = entry.get("target")
        target_keys = {"path", "public_surface", "release"}
        if not isinstance(target, dict) or set(target) != target_keys:
            raise LedgerError(f"{component}: target must declare path/public_surface/release")
        if (
            not isinstance(target["path"], str)
            or not target["path"].startswith(("python/src/veridist/", "python/docs/"))
            or Path(target["path"]).is_absolute()
            or ".." in Path(target["path"]).parts
            or not isinstance(target["public_surface"], bool)
            or not isinstance(target["release"], str)
            or not RELEASE.fullmatch(target["release"])
        ):
            raise LedgerError(f"{component}: target types are invalid")
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != {
            "scenario_ids",
            "test_ids",
            "reference_ids",
        }:
            raise LedgerError(f"{component}: evidence must declare scenario/test/reference IDs")
        for value in evidence.values():
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item for item in value)
            ):
                raise LedgerError(f"{component}: evidence IDs must be non-empty string lists")
        reviews = entry.get("reviews")
        if not isinstance(reviews, dict) or set(reviews) != {
            "license",
            "statistical",
            "scale",
            "i18n",
        }:
            raise LedgerError(f"{component}: reviews must be categorized")
        review_values = set(REVIEW_VALUES)
        if any(value not in review_values for value in reviews.values()):
            raise LedgerError(f"{component}: review values are invalid")
        isolation = entry.get("isolation")
        isolation_keys = {
            "runtime_import",
            "runtime_dependency",
            "dynamic_fallback",
            "package_content",
            "oracle",
        }
        if (
            not isinstance(isolation, dict)
            or set(isolation) != isolation_keys
            or any(value is not False for value in isolation.values())
        ):
            raise LedgerError(f"{component}: all legacy isolation booleans must be false")
        source = entry.get("source")
        if not isinstance(source, dict):
            raise LedgerError(f"{component}: source must be an object")
        commit, path, blob, sha256 = (
            source.get(key) for key in ("commit", "path", "blob", "sha256")
        )
        if not isinstance(commit, str) or not HEX40.fullmatch(commit):
            raise LedgerError(f"{component}: source.commit must be a 40-character hash")
        allowed_roots = ("distfit_pro/", "tests/", "examples/")
        if (
            not isinstance(path, str)
            or not path.startswith(allowed_roots)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise LedgerError(f"{component}: source.path must stay in an allowed legacy root")
        if not isinstance(blob, str) or not HEX40.fullmatch(blob):
            raise LedgerError(f"{component}: source.blob must be a 40-character hash")
        if not isinstance(sha256, str) or not HEX64.fullmatch(sha256):
            raise LedgerError(f"{component}: source.sha256 must be a 64-character hash")
        _require_available_commit(commit, component)
        if _git("rev-parse", f"{commit}:{path}") != blob:
            raise LedgerError(f"{component}: source.blob does not match source.commit:path")
        payload = _git_bytes("show", f"{commit}:{path}")
        observed_hash = hashlib.sha256(payload).hexdigest()
        if observed_hash != sha256:
            raise LedgerError(f"{component}: source.sha256 is stale")
        if disposition == "modify_port" and entry.get("status") != "review_pending":
            raise LedgerError(f"{component}: modify_port must remain review_pending")
        if component == "exponential" and disposition != "rewrite":
            raise LedgerError("exponential must be a rewrite")
    if seen_dispositions != ALLOWED_DISPOSITIONS:
        raise LedgerError("ledger must demonstrate every allowed disposition")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parsed = parser.parse_args(arguments)
    try:
        validate(parsed.schema, parsed.ledger)
    except LedgerError as error:
        print(f"migration ledger check failed: {error}", file=sys.stderr)
        return 1
    print("migration ledger check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
