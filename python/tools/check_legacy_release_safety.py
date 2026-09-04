"""Fail closed if the legacy GitHub workflow can acquire publication capability."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

_PUBLICATION_JOB = re.compile(r"(?:publish|release|deploy)", re.IGNORECASE)
_RELEASE_ENVIRONMENT = re.compile(r"(?:pypi|production|prod|release|deploy)", re.IGNORECASE)
_SECRET_REFERENCE = re.compile(r"\$\{\{\s*secrets\.", re.IGNORECASE)
_PUBLISH_ACTION = re.compile(r"(?:^|[-_/])publish(?:$|[-_/])", re.IGNORECASE)
_PYPI_ACTION = re.compile(r"(?:pypi|upload[-_]?to[-_]?pypi)", re.IGNORECASE)
_PUBLISH_COMMAND = re.compile(
    r"\b(?:python\s+-m\s+)?(?:twine\s+upload|uv\s+publish|"
    r"hatch\s+publish|poetry\s+publish|flit\s+publish)\b|"
    r"\b(?:publish|upload[-_ ]?to[-_ ]?pypi)\b",
    re.IGNORECASE,
)


def _walk_mappings(value: object) -> tuple[dict[str, object], ...]:
    if isinstance(value, dict):
        nested = tuple(
            mapping for child in value.values() for mapping in _walk_mappings(child)
        )
        return (value, *nested)
    if isinstance(value, list):
        return tuple(mapping for child in value for mapping in _walk_mappings(child))
    return ()


def _walk_scalars(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(scalar for child in value.values() for scalar in _walk_scalars(child))
    if isinstance(value, list):
        return tuple(scalar for child in value for scalar in _walk_scalars(child))
    return ()


def _contains_release(value: object) -> bool:
    if isinstance(value, str):
        return value.casefold() == "release"
    if isinstance(value, dict):
        return any(
            str(key).casefold() == "release" or _contains_release(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_release(child) for child in value)
    return False


def _grants_write_permission(value: object) -> bool:
    if isinstance(value, str):
        return value.casefold() == "write-all"
    if not isinstance(value, dict):
        return False
    return any(
        str(key).casefold() in {"id-token", "packages"}
        and isinstance(permission, str)
        and permission.casefold() == "write"
        for key, permission in value.items()
    )


def _is_release_environment(value: object) -> bool:
    if isinstance(value, str):
        return _RELEASE_ENVIRONMENT.search(value) is not None
    if isinstance(value, dict):
        name = value.get("name")
        return isinstance(name, str) and _RELEASE_ENVIRONMENT.search(name) is not None
    return False


def _steps(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(step for step in value if isinstance(step, dict))


def _is_publisher_action(value: object) -> bool:
    if not isinstance(value, str):
        return False
    action = value.casefold()
    if action.startswith("actions/upload-artifact@"):
        return False
    return _PYPI_ACTION.search(action) is not None or _PUBLISH_ACTION.search(action) is not None


def _strip_shell_comment(line: str) -> str:
    quote: str | None = None
    for index, character in enumerate(line):
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif character == "#" and quote is None:
            return line[:index]
    return line


def _shell_segments(line: str) -> tuple[str, ...]:
    segments: list[str] = []
    start = 0
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
        elif quote is None and character in {";", "|", "&"}:
            segments.append(line[start:index])
            if character in {"|", "&"} and index + 1 < len(line):
                if line[index + 1] == character:
                    index += 1
            start = index + 1
        index += 1
    segments.append(line[start:])
    return tuple(segments)


def _is_publisher_command(value: object) -> bool:
    if not isinstance(value, str):
        return False
    for raw_line in value.splitlines():
        for segment in _shell_segments(_strip_shell_comment(raw_line)):
            command = segment.strip()
            if not command or re.match(r"(?:command\s+)?(?:echo|printf)\b", command):
                continue
            if _PUBLISH_COMMAND.search(command) is not None:
                return True
    return False


def find_violations(workflow: str) -> tuple[str, ...]:
    """Return every publication-capability violation in a legacy workflow text.

    The parser uses PyYAML's ``BaseLoader`` so that GitHub Actions' bare ``on``
    key remains a string instead of YAML 1.1's boolean.  Comments and harmless
    prose do not become YAML values; publication checks inspect only workflow
    triggers, permissions, job environments, secrets, and executable steps.
    """

    if type(workflow) is not str:
        raise TypeError("workflow must be a built-in string")

    try:
        document = yaml.load(workflow, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return ("invalid YAML",)
    if not isinstance(document, dict):
        return ("workflow root is not a mapping",)

    violations: list[str] = []
    if _contains_release(document.get("on")):
        violations.append("release trigger")
    if any(
        _grants_write_permission(mapping.get("permissions"))
        for mapping in _walk_mappings(document)
    ):
        violations.append("trusted or package write permission")
    if any(_SECRET_REFERENCE.search(scalar) for scalar in _walk_scalars(document)):
        violations.append("secret reference")

    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return tuple(violations)
    for job_name, job in jobs.items():
        if _PUBLICATION_JOB.search(str(job_name)) is not None:
            violations.append("publication-like job")
        if not isinstance(job, dict):
            continue
        if _is_release_environment(job.get("environment")):
            violations.append("release credential environment")
        for mapping in _walk_mappings(job):
            for step in _steps(mapping.get("steps")):
                if _is_publisher_action(step.get("uses")):
                    violations.append("publication action")
                if _is_publisher_command(step.get("run")):
                    violations.append("publication command")
    return tuple(dict.fromkeys(violations))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    arguments = parser.parse_args()
    violations = find_violations(arguments.workflow.read_text(encoding="utf-8"))
    if not violations:
        return 0
    for violation in violations:
        print(f"legacy release safety violation: {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
