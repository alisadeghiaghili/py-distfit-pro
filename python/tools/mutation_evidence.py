"""Shared deterministic rules for formal mutmut evidence (schema v1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
CRITICAL_MODULES = ("domain", "statistics", "families", "engine")
UNRESOLVED_STATUSES = {"suspicious", "timeout", "error", "unclassified"}
ALLOWED_STATUSES = {"killed", "survived", *UNRESOLVED_STATUSES}


def official_status(exit_code: object) -> str:
    if exit_code is None:
        return "not_checked"
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return "unknown"
    return {
        0: "survived",
        1: "killed",
        3: "killed",
        5: "no_tests",
        33: "no_tests",
        2: "interrupted",
        34: "skipped",
        35: "suspicious",
        36: "timeout",
        37: "type_check",
        -24: "timeout",
        24: "timeout",
        152: "timeout",
        255: "timeout",
        -11: "segfault",
        -9: "segfault",
    }.get(exit_code, "unknown")


def scoring_status(exit_code: object) -> str:
    value = official_status(exit_code)
    if value in {"killed", "type_check"}:
        return "killed"
    if value == "survived":
        return "survived"
    return "unresolved"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_files(project_root: Path) -> list[str]:
    root = project_root / "src" / "veridist"
    return sorted(
        path.relative_to(project_root).as_posix()
        for module in CRITICAL_MODULES
        for path in (root / module).rglob("*.py")
        if path.is_file()
    )


def source_tree_digest(project_root: Path, files: list[str] | None = None) -> str:
    files = source_files(project_root) if files is None else files
    digest = hashlib.sha256()
    for name in files:
        payload = (project_root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def mutation_config(project_root: Path) -> dict[str, Any]:
    import tomllib

    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    config = project.get("tool", {}).get("mutmut")
    if not isinstance(config, dict):
        raise ValueError("[tool.mutmut] is required")
    forbidden = {"do_not_mutate", "do_not_mutate_patterns", "only_mutate"} & set(config)
    if forbidden:
        raise ValueError(f"forbidden mutmut exclusion configuration: {sorted(forbidden)}")
    paths = config.get("source_paths")
    if paths != [f"src/veridist/{module}" for module in CRITICAL_MODULES]:
        raise ValueError("mutmut source_paths must list every critical module exactly once")
    selection = config.get("pytest_add_cli_args_test_selection")
    if selection != ["tests"]:
        raise ValueError("mutmut must select the full tests suite")
    if config.get("mutate_only_covered_lines") is not False:
        raise ValueError("mutmut must not reduce its denominator to covered lines")
    return config


def config_digest(project_root: Path) -> str:
    return sha256_text(canonical_json(mutation_config(project_root)))
