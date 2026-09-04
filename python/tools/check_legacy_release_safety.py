"""Fail closed if the legacy GitHub workflow can acquire publication capability."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_RELEASE_TRIGGER = re.compile(r"(?m)^ {2}release\s*:")
_PUBLICATION_JOB = re.compile(r"(?im)^ {2}[a-z0-9_-]*(?:publish|release|deploy)[a-z0-9_-]*\s*:")
_WRITE_PERMISSION = re.compile(r"(?im)^\s*(?:id-token|packages)\s*:\s*[\"']?write\b")
_DEPLOYMENT_ENVIRONMENT = re.compile(r"(?im)^\s*environment\s*:")
_SECRET_REFERENCE = re.compile(r"\$\{\{\s*secrets\.", re.IGNORECASE)
_PUBLISH_ACTION = re.compile(
    r"(?im)^\s*[-]?\s*uses\s*:\s*[^\n]*(?:publish|upload[-_ ]?to[-_ ]?pypi)"
)
_PUBLISH_COMMAND = re.compile(
    r"(?im)\b(?:python\s+-m\s+)?(?:twine\s+upload|uv\s+publish|"
    r"hatch\s+publish|poetry\s+publish|flit\s+publish)\b|"
    r"\b(?:publish|upload[-_ ]?to[-_ ]?pypi)\b"
)


def find_violations(workflow: str) -> tuple[str, ...]:
    """Return every publication-capability violation in a legacy workflow text.

    GitHub Actions accepts YAML 1.2, while common local YAML parsers disagree
    about the bare ``on`` key.  These structural checks deliberately inspect
    the workflow vocabulary that grants a legacy workflow publication power;
    they do not attempt to implement a partial YAML parser.
    """

    if type(workflow) is not str:
        raise TypeError("workflow must be a built-in string")

    checks: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("release trigger", _RELEASE_TRIGGER),
        ("publication-like job", _PUBLICATION_JOB),
        ("trusted or package write permission", _WRITE_PERMISSION),
        ("deployment environment", _DEPLOYMENT_ENVIRONMENT),
        ("secret reference", _SECRET_REFERENCE),
        ("publication action", _PUBLISH_ACTION),
        ("publication command", _PUBLISH_COMMAND),
    )
    return tuple(description for description, pattern in checks if pattern.search(workflow))


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
