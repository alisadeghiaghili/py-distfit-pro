"""Contracts for the isolated trusted-publishing workflow."""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pypi-publish.yml"


def test_pypi_publish_workflow_is_release_bound_and_oidc_only() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "types: [published]",
        "environment: pypi",
        "id-token: write",
        "attestations: write",
        "pypa/gh-action-pypi-publish@release/v1",
        "packages-dir: dist",
        "gh release download",
    ):
        assert required in content
    assert "PYPI_TOKEN" not in content
