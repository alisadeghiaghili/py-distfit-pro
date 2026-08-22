"""Enforce that veridist never imports or ships legacy distfit_pro code."""

from __future__ import annotations

import argparse
import ast
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

LEGACY_ROOT = "distfit_pro"


class IsolationError(ValueError):
    """Raised when a source or distribution violates the legacy boundary."""


def _constant_string(node: ast.AST, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left, names)
        right = _constant_string(node.right, names)
        return None if left is None or right is None else left + right
    return None


def _is_legacy_module(name: str) -> bool:
    return name == LEGACY_ROOT or name.startswith(f"{LEGACY_ROOT}.")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan_tree(tree: ast.AST, source: Path) -> list[str]:
    violations: list[str] = []
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _constant_string(node.value, names)
            if value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _constant_string(node.value, names) if node.value is not None else None
            if value is not None:
                names[node.target.id] = value
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_legacy_module(alias.name):
                    violations.append(f"legacy import in {source}: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and _is_legacy_module(node.module):
            violations.append(f"legacy import in {source}: {node.module}")
        elif isinstance(node, ast.Call) and node.args and _call_name(node.func) in {"__import__", "import_module"}:
            module_name = _constant_string(node.args[0], names)
            if module_name is not None and _is_legacy_module(module_name):
                violations.append(f"legacy import in {source}: {module_name}")
    return violations


def scan_source(source_root: Path) -> list[str]:
    if not source_root.is_dir():
        raise IsolationError(f"source directory does not exist: {source_root}")
    violations: list[str] = []
    for source in sorted(source_root.rglob("*.py")):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as error:
            raise IsolationError(f"cannot parse {source}: {error}") from error
        violations.extend(_scan_tree(tree, source))
    return violations


def _is_legacy_payload(member_name: str) -> bool:
    parts = Path(member_name).parts
    return LEGACY_ROOT in parts or Path(member_name).name == f"{LEGACY_ROOT}.py"


def _artifact_members(artifact: Path) -> Iterable[str]:
    if artifact.suffix == ".whl" or zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as archive:
            yield from archive.namelist()
        return
    if tarfile.is_tarfile(artifact):
        with tarfile.open(artifact) as archive:
            yield from (member.name for member in archive.getmembers())
        return
    raise IsolationError(f"unsupported artifact: {artifact}")


def scan_artifact(artifact: Path) -> list[str]:
    if not artifact.is_file():
        raise IsolationError(f"artifact does not exist: {artifact}")
    return [
        f"legacy payload in {artifact}: {member}"
        for member in _artifact_members(artifact)
        if _is_legacy_payload(member)
    ]


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parsed = parser.parse_args(arguments)
    if not parsed.source and not parsed.artifact:
        parser.error("at least one --source or --artifact is required")
    try:
        violations = [violation for source in parsed.source for violation in scan_source(source)]
        violations.extend(
            violation for artifact in parsed.artifact for violation in scan_artifact(artifact)
        )
    except IsolationError as error:
        print(f"legacy isolation check failed: {error}", file=sys.stderr)
        return 1
    if violations:
        print("legacy isolation check failed: " + "; ".join(violations), file=sys.stderr)
        return 1
    print("legacy isolation check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
