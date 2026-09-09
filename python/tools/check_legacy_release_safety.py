"""Fail closed unless archived legacy CI matches its reviewed structural manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
_SECRET_EXPRESSION = re.compile(r"(?i)\bsecrets\s*(?:\.|\[)")
_APPROVED_ACTIONS = frozenset({"actions/checkout@v4", "actions/setup-python@v5"})
_APPROVED_RUN_SHA256 = frozenset(
    {
        "b3a953c025bc5de29e8d6e6f640dc51c2a8352f88ceb72c11df70dfa4627c33e",
        "8d9b8aa7a497a8a5be0112e49ddebd3a4031ca70f67a37eb7beb420615233423",
        "1a4b784d5261462dc46d2b14ac1b29e8c8988d7c066818b091e72c00490ad799",
        "92423a744a3db8b5dd9d22b9861c50faa7b2122583bdb19172bb4a8bfa6ff93d",
        "6c3aed6819e5a743ca17c67aff7946dfb82018f71a9abed003a37d81e4df67fb",
    }
)
# Environment and action inputs are capabilities, so their approved values are
# intentionally bound to an exact step position.  A global allow-list would let
# a reviewed payload be transplanted into a different, more privileged step.
_APPROVED_STEP_ENV_SHA256 = {
    ("legacy-scope", 2): "c4adfa89e0d1094d8ef1f649948e1b479c7c597bc1fa003c92525dfebf4d4155",
    ("legacy-gate", 2): "0bea57b7d274c74287d2ae3c79ca98ab278ba80150e3d76c535d39c36c60c133",
}
_APPROVED_STEP_WITH_SHA256 = {
    ("legacy-scope", 0): "183d260e8b678c373a566ae833d8d36277a19f369c38897d04524715cf579002",
    ("legacy-scope", 1): "936963ee6f7504ff953519faacf98865febff4eb6cd471ca903ca35188633e9f",
    ("legacy-test", 1): "16105d964abbe3f77f9498112614313cdab360afb7c7176c20b93d1a064d55e9",
    ("legacy-gate", 1): "936963ee6f7504ff953519faacf98865febff4eb6cd471ca903ca35188633e9f",
}


def _load_document(workflow: str) -> dict[str, object] | None:
    try:
        document = yaml.load(workflow, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return None
    return document if isinstance(document, dict) else None


def _canonical_sha256(document: dict[str, object]) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _load_manifest(path: Path | None = None) -> dict[str, object] | None:
    if path is None:
        path = _MANIFEST_PATH
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _semantic_inventory(document: dict[str, object]) -> tuple[list[str], dict[str, str]]:
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return [], {}
    actions: list[str] = []
    step_sha256: dict[str, str] = {}
    for job_id in sorted(jobs):
        job = jobs[job_id]
        if not isinstance(job, dict) or not isinstance(job.get("steps"), list):
            continue
        for index, step in enumerate(job["steps"]):
            if not isinstance(step, dict):
                continue
            step_sha256[f"{job_id}:{index}"] = _canonical_sha256(step)
            action = step.get("uses")
            if isinstance(action, str):
                actions.append(action)
    return sorted(actions), step_sha256


def _contains_secret(value: object) -> bool:
    if isinstance(value, str):
        return _SECRET_EXPRESSION.search(value) is not None
    if isinstance(value, dict):
        return any(
            "secret" in str(key).casefold() or _contains_secret(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret(child) for child in value)
    return False


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
    if _contains_secret(document):
        violations.append("secret capability")
    actions, step_sha256 = _semantic_inventory(document)
    if not set(actions).issubset(_APPROVED_ACTIONS):
        violations.append("unapproved action")
    jobs = document.get("jobs")
    if isinstance(jobs, dict):
        for job_id, job in jobs.items():
            if not isinstance(job, dict) or not isinstance(job.get("steps"), list):
                continue
            for index, step in enumerate(job["steps"]):
                if isinstance(step, dict) and isinstance(step.get("run"), str):
                    digest = hashlib.sha256(step["run"].encode("utf-8")).hexdigest()
                    if digest not in _APPROVED_RUN_SHA256:
                        violations.append("unapproved run block")
                if not isinstance(step, dict):
                    continue
                position = (job_id, index)
                expected_env_sha256 = _APPROVED_STEP_ENV_SHA256.get(position)
                environment = step.get("env")
                if expected_env_sha256 is None:
                    if "env" in step:
                        violations.append("unapproved step environment")
                elif (
                    not isinstance(environment, dict)
                    or _canonical_sha256(environment) != expected_env_sha256
                ):
                    violations.append("unapproved step environment")

                expected_with_sha256 = _APPROVED_STEP_WITH_SHA256.get(position)
                inputs = step.get("with")
                if expected_with_sha256 is None:
                    if "with" in step:
                        violations.append("unapproved step inputs")
                elif (
                    not isinstance(inputs, dict)
                    or _canonical_sha256(inputs) != expected_with_sha256
                ):
                    violations.append("unapproved step inputs")
    manifest = _load_manifest()
    if manifest is None:
        return tuple([*violations, "missing or invalid legacy manifest"])
    expected_manifest_keys = {
        "schema_version",
        "workflow_sha256",
        "job_ids",
        "approved_actions",
        "step_sha256",
    }
    if (
        frozenset(manifest) != expected_manifest_keys
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
    ):
        violations.append("unsupported legacy manifest schema")
    if manifest.get("workflow_sha256") != _canonical_sha256(document):
        violations.append("legacy workflow fingerprint differs from manifest")
    if manifest.get("job_ids") != sorted(_JOB_KEYS):
        violations.append("legacy manifest job inventory differs")
    if (
        not isinstance(manifest.get("approved_actions"), list)
        or manifest.get("approved_actions") != actions
    ):
        violations.append("legacy manifest action inventory differs")
    if (
        not isinstance(manifest.get("step_sha256"), dict)
        or manifest.get("step_sha256") != step_sha256
    ):
        violations.append("legacy manifest step inventory differs")
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
