"""Contracts for the fail-closed Veridist release validation workflow."""

import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "veridist-release.yml"


class ReleaseWorkflowTests(unittest.TestCase):
    def test_workflow_validates_only_and_binds_tag_version_and_artifacts(self) -> None:
        content = WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "name: veridist-release-validation",
            "workflow_dispatch:",
            "release_tag:",
            "working-directory: python",
            "python -m build --sdist --wheel",
            "python -m twine check dist/*",
            "python tools/check_release_artifacts.py",
            '--release-tag "${{ inputs.release_tag }}"',
            "--artifact dist/*.whl",
            "--artifact dist/*.tar.gz",
            "importlib.metadata",
            "veridist",
            "release tag does not match package version",
        ):
            with self.subTest(required=required):
                self.assertIn(required, content)
        self.assertNotIn("twine upload", content)
        self.assertNotIn("pypa/gh-action-pypi-publish", content)


if __name__ == "__main__":
    unittest.main()
