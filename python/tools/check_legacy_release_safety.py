"""Fail closed unless archived legacy CI matches its reviewed structural manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "quality" / "legacy-ci-manifest.json"
_TOP_LEVEL_KEYS = frozenset({"name", "on", "permissions", "jobs"})
_TRIGGERS = frozenset({"push", "pull_request", "workflow_dispatch"})
_JOB_KEYS = {
    "legacy-scope": frozenset({"name", "runs-on", "outputs", "steps"}),
    "legacy-test": frozenset({"name", "needs", "if", "runs-on", "strategy", "steps"}),
    "legacy-gate": frozenset({"name", "needs", "if", "runs-on", "steps"}),
}
_FORBIDDEN_JOB_KEYS = frozenset(
    {"uses", "secrets", "environment", "permissions", "container", "services"}
)


def _load_document(workflow: str) -> dict[str, object] | None:
    try:
        document = yaml.load(workflow, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return None
    return document if isinstance(document, dict) else None


def _canonical_sha256(document: dict[str, object]) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _load_manifest(path: Path = _MANIFEST_PATH) -> dict[str, object] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _structural_violations(document: dict[str, object]) -> tuple[str, ...]:
    violations: list[str] = []
    if frozenset(document) != _TOP_LEVEL_KEYS:
        violations.append("unknown top-level workflow structure")
    triggers = document.get("on")
    if not isinstance(triggers, dict) or frozenset(triggers) != _TRIGGERS:
        violations.append("unapproved workflow triggers")
    if document.get("permissions") != {"contents": "read"}:
        violations.append("workflow permissions are not read-only")
    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or frozenset(jobs) != frozenset(_JOB_KEYS):
        return tuple([*violations, "unapproved job inventory"])
    for job_id, allowed_keys in _JOB_KEYS.items():
        job = jobs.get(job_id)
        if not isinstance(job, dict) or not frozenset(job).issubset(allowed_keys):
            violations.append(f"unapproved structure in {job_id}")
            continue
        if any(key in job for key in _FORBIDDEN_JOB_KEYS):
            violations.append(f"forbidden capability in {job_id}")
        steps = job.get("steps")
        if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
            violations.append(f"invalid steps in {job_id}")
            continue
        for step in steps:
            if not frozenset(step).issubset({"id", "name", "uses", "with", "shell", "env", "run"}):
                violations.append(f"unapproved step structure in {job_id}")
            for key in ("with", "env"):
                if key in step and not isinstance(step[key], dict):
                    violations.append(f"invalid {key} in {job_id}")
            if "uses" in step and not isinstance(step["uses"], str):
                violations.append(f"invalid action in {job_id}")
            if "run" in step and not isinstance(step["run"], str):
                violations.append(f"invalid run block in {job_id}")
    return tuple(dict.fromkeys(violations))


def find_violations(workflow: str) -> tuple[str, ...]:
    """Return fail-closed violations for an archived legacy workflow."""

    if type(workflow) is not str:
        raise TypeError("workflow must be a built-in string")
    document = _load_document(workflow)
    if document is None:
        return ("invalid YAML",)
    violations = list(_structural_violations(document))
    manifest = _load_manifest()
    if manifest is None:
        return tuple([*violations, "missing or invalid legacy manifest"])
    if manifest.get("schema_version") != 1:
        violations.append("unsupported legacy manifest schema")
    if manifest.get("workflow_sha256") != _canonical_sha256(document):
        violations.append("legacy workflow fingerprint differs from manifest")
    if manifest.get("job_ids") != sorted(_JOB_KEYS):
        violations.append("legacy manifest job inventory differs")
    if manifest.get("approved_actions") != ["actions/checkout@v4", "actions/setup-python@v5"]:
        violations.append("legacy manifest action inventory differs")
    return tuple(dict.fromkeys(violations))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    arguments = parser.parse_args()
    try:
        workflow = arguments.workflow.read_text(encoding="utf-8")
    except OSError as error:
        parser.error(str(error))
    violations = find_violations(workflow)
    if not violations:
        return 0
    for violation in violations:
        print(f"legacy release safety violation: {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
