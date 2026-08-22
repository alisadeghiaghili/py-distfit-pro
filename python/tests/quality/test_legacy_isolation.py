"""Prevent legacy implementation material from crossing the veridist boundary."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
PACKAGE_ROOT = PYTHON_ROOT / "src" / "veridist"
CHECKER_PATH = PYTHON_ROOT / "tools" / "check_legacy_isolation.py"


class LegacyIsolationTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER_PATH), *arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_current_veridist_source_is_isolated_from_legacy(self) -> None:
        self.assertTrue(CHECKER_PATH.is_file())
        result = self._run("--source", str(PACKAGE_ROOT))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checker_rejects_static_dynamic_and_composed_legacy_imports(self) -> None:
        cases = (
            "from distfit_pro import Fitter\n",
            "import importlib\nimportlib.import_module('distfit_pro.core')\n",
            "from importlib import import_module as legacy_loader\nlegacy_loader('distfit_pro')\n",
            "module = 'distfit' + '_pro'\n__import__(module)\n",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source"
            source.mkdir()
            for index, content in enumerate(cases):
                (source / f"case_{index}.py").write_text(content, encoding="utf-8")
            result = self._run("--source", str(source))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("legacy import", result.stderr)

    def test_checker_rejects_wheel_and_sdist_with_legacy_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            wheel = directory / "veridist-0.0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("distfit_pro/__init__.py", "")
                archive.writestr(
                    "veridist-0.0.0.dist-info/METADATA", "Requires-Dist: distfit-pro\n"
                )
            sdist = directory / "veridist-0.0.0.tar.gz"
            payload = directory / "distfit_pro.py"
            payload.write_text("", encoding="utf-8")
            with tarfile.open(sdist, "w:gz") as archive:
                archive.add(payload, arcname="veridist-0.0.0/distfit_pro.py")
            for artifact in (wheel, sdist):
                with self.subTest(artifact=artifact.name):
                    result = self._run("--artifact", str(artifact))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("legacy payload", result.stderr)

    def test_checker_rejects_legacy_distribution_dependency_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel = Path(temporary_directory) / "veridist-0.0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "veridist-0.0.0.dist-info/METADATA", "Requires-Dist: distfit-pro\n"
                )
            result = self._run("--artifact", str(wheel))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("legacy dependency", result.stderr)

    def test_checker_normalizes_pep503_legacy_dependency_names(self) -> None:
        for name in ("distfit_pro[docs]; python_version >= '3.11'", "distfit.pro>=1"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                wheel = Path(temporary_directory) / "veridist-0.0.0-py3-none-any.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("veridist-0.0.0.dist-info/METADATA", f"Requires-Dist: {name}\n")
                result = self._run("--artifact", str(wheel))
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
