"""Validate release metadata shared by Python, CFF, Zenodo, and conda-forge."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

import yaml

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_LINE = re.compile(r'^\{% set version = "([^"]+)" %\}$', re.MULTILINE)


def _date(value: object) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None
    return None


def validate(repository: Path) -> list[str]:
    """Return every cross-file release-metadata violation."""

    errors: list[str] = []
    try:
        project = tomllib.loads((repository / "python/pyproject.toml").read_text("utf-8"))[
            "project"
        ]
        version = project["version"]
        license_id = project["license"]
        citation = yaml.safe_load((repository / "CITATION.cff").read_text("utf-8"))
        zenodo = json.loads((repository / ".zenodo.json").read_text("utf-8"))
        recipe = (repository / "conda-forge-recipe/meta.yaml").read_text("utf-8")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        return [f"release metadata is unreadable: {error}"]
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return ["project version is not a stable semantic version"]
    if not isinstance(citation, dict):
        return ["CITATION.cff root must be a mapping"]
    if citation.get("cff-version") != "1.2.0":
        errors.append("CITATION.cff must use CFF 1.2.0")
    if citation.get("version") != version:
        errors.append("CITATION.cff version differs from the package")
    released = _date(citation.get("date-released"))
    if released is None:
        errors.append("CITATION.cff lacks a valid release date")
    if citation.get("license") != license_id:
        errors.append("CITATION.cff license differs from the package")
    if not isinstance(citation.get("authors"), list) or not citation["authors"]:
        errors.append("CITATION.cff lacks authors")
    if citation.get("repository-code") != "https://github.com/alisadeghiaghili/veridist":
        errors.append("CITATION.cff repository is not canonical")
    if not isinstance(zenodo, dict):
        return [*errors, ".zenodo.json root must be an object"]
    if zenodo.get("version") != version:
        errors.append("Zenodo version differs from the package")
    if zenodo.get("publication_date") != released:
        errors.append("Zenodo and CFF release dates differ")
    if zenodo.get("upload_type") != "software":
        errors.append("Zenodo upload type must be software")
    if zenodo.get("license") != license_id:
        errors.append("Zenodo license differs from the package")
    if not isinstance(zenodo.get("creators"), list) or not zenodo["creators"]:
        errors.append("Zenodo metadata lacks creators")
    recipe_version = _VERSION_LINE.search(recipe)
    if recipe_version is None or recipe_version.group(1) != version:
        errors.append("conda-forge recipe version differs from the package")
    digest = re.search(r"^\s*sha256:\s*([0-9a-f]+)\s*$", recipe, re.MULTILINE)
    if digest is None or _SHA256.fullmatch(digest.group(1)) is None:
        errors.append("conda-forge recipe lacks an immutable SHA-256")
    required_recipe_text = (
        "releases/download/v{{ version }}/veridist-{{ version }}.tar.gz",
        "--no-deps",
        "--no-build-isolation",
        "license_file: LICENSE",
    )
    for required in required_recipe_text:
        if required not in recipe:
            errors.append(f"conda-forge recipe lacks required text: {required}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.repository_root.resolve())
    if errors:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS: release metadata is complete and version-aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
