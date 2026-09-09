"""Adversarial contracts for Veridist release distributions."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.check_release_artifacts import ReleaseArtifactError, validate_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.0.0.dev0"
METADATA = (
    f"Metadata-Version: 2.4\nName: veridist\nVersion: {VERSION}\nLicense-Expression: BUSL-1.1\n\n"
).encode()


def _wheel(path: Path, *, metadata: bytes = METADATA, legacy: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("veridist/__init__.py", f'__version__ = "{VERSION}"\n')
        archive.writestr("veridist-0.0.0.dev0.dist-info/METADATA", metadata)
        archive.writestr(
            "veridist-0.0.0.dev0.dist-info/licenses/LICENSE",
            (PROJECT_ROOT / "LICENSE").read_bytes(),
        )
        if legacy:
            archive.writestr("distfit_pro/__init__.py", b"")


def _sdist(path: Path, *, modified_known_limits: bool = False) -> None:
    members = {
        "veridist-0.0.0.dev0/PKG-INFO": METADATA,
        "veridist-0.0.0.dev0/src/veridist.egg-info/PKG-INFO": METADATA,
        "veridist-0.0.0.dev0/LICENSE": (PROJECT_ROOT / "LICENSE").read_bytes(),
        "veridist-0.0.0.dev0/src/veridist/__init__.py": b"",
        **{
            f"veridist-0.0.0.dev0/{name}": (PROJECT_ROOT / name).read_bytes()
            for name in (
                "CHANGELOG.md",
                "KNOWN_LIMITS.md",
                "KNOWN_LIMITS.fa.md",
                "KNOWN_LIMITS.de.md",
            )
        },
    }
    if modified_known_limits:
        members["veridist-0.0.0.dev0/KNOWN_LIMITS.md"] = b"modified\n"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class ReleaseArtifactContractTests(unittest.TestCase):
    def test_valid_wheel_and_sdist_match_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "veridist-0.0.0.dev0-py3-none-any.whl"
            sdist = root / "veridist-0.0.0.dev0.tar.gz"
            _wheel(wheel)
            _sdist(sdist)
            for artifact in (wheel, sdist):
                with self.subTest(artifact=artifact.name):
                    validate_artifact(
                        artifact,
                        project_root=PROJECT_ROOT,
                        release_tag=f"v{VERSION}",
                    )

    def test_rejects_mismatched_tag_metadata_and_legacy_payload(self) -> None:
        cases = {
            "tag": (METADATA, False, "v9.9.9", "does not match package version"),
            "version": (
                METADATA.replace(VERSION.encode(), b"9.9.9"),
                False,
                f"v{VERSION}",
                "metadata Version",
            ),
            "license": (
                METADATA.replace(b"BUSL-1.1", b"MIT"),
                False,
                f"v{VERSION}",
                "metadata License-Expression",
            ),
            "legacy": (METADATA, True, f"v{VERSION}", "legacy distfit_pro"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (metadata, legacy, tag, message) in cases.items():
                artifact = Path(directory) / f"{name}.whl"
                _wheel(artifact, metadata=metadata, legacy=legacy)
                with self.subTest(case=name), self.assertRaisesRegex(ReleaseArtifactError, message):
                    validate_artifact(artifact, project_root=PROJECT_ROOT, release_tag=tag)

    def test_rejects_modified_release_document_in_sdist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "veridist-0.0.0.dev0.tar.gz"
            _sdist(artifact, modified_known_limits=True)
            with self.assertRaisesRegex(ReleaseArtifactError, "modified KNOWN_LIMITS.md"):
                validate_artifact(
                    artifact,
                    project_root=PROJECT_ROOT,
                    release_tag=f"v{VERSION}",
                )


if __name__ == "__main__":
    unittest.main()
