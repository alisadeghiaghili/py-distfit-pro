"""Release-metadata contracts for the nested Veridist package."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
PYPROJECT = PYTHON_ROOT / "pyproject.toml"
LANGUAGE_NAVIGATION = (
    "[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md)"
    " | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md)"
    " | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)"
)
README_PATHS = {
    "en": PYTHON_ROOT / "README.md",
    "fa": PYTHON_ROOT / "README.fa.md",
    "de": PYTHON_ROOT / "README.de.md",
}


class PackageLandingContractTests(unittest.TestCase):
    def test_project_metadata_points_to_packaged_human_facing_material(self) -> None:
        configuration = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        project = configuration["project"]
        self.assertIn("setuptools>=77", configuration["build-system"]["requires"])
        self.assertEqual(
            project["description"],
            "Evidence-first distribution fitting with explicit execution contracts",
        )
        self.assertNotIn("greenfield", project["description"].casefold())
        self.assertEqual(
            project["readme"], {"file": "README.md", "content-type": "text/markdown"}
        )
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(
            project["urls"],
            {
                "Homepage": "https://github.com/alisadeghiaghili/py-distfit-pro",
                "Documentation": (
                    "https://github.com/alisadeghiaghili/py-distfit-pro/tree/main/python/docs"
                ),
                "Repository": "https://github.com/alisadeghiaghili/py-distfit-pro",
                "Issues": "https://github.com/alisadeghiaghili/py-distfit-pro/issues",
            },
        )

    def test_nested_license_is_an_exact_copy_of_the_repository_license(self) -> None:
        self.assertEqual(
            (PYTHON_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines(),
            (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines(),
        )

    def test_all_three_package_readmes_exist_link_each_other_and_match_version(self) -> None:
        for locale, path in README_PATHS.items():
            with self.subTest(locale=locale):
                content = path.read_text(encoding="utf-8")
                self.assertIn(LANGUAGE_NAVIGATION, content)
                self.assertIn("veridist", content.casefold())
                self.assertIn("0.0.0.dev0", content)
                self.assertIn("cd py-distfit-pro/python", content)
                self.assertIn("python -m pip install .", content)
                self.assertIn("python -m pip install /path/to/veridist-", content)
                self.assertIsNone(
                    re.search(r"(?m)^python -m pip install veridist\s*$", content)
                )

    def test_each_locale_states_the_same_pre_alpha_non_claims(self) -> None:
        required = {
            "en": (
                "pre-alpha contract kernel",
                "does not provide a distribution-fitting API",
                "does not ship production data adapters",
                "does not claim persistent checkpoint durability",
            ),
            "fa": (
                "هستهٔ قراردادی پیش‌آلفا",
                "API برازش توزیع ارائه نمی‌کند",
                "آداپتور دادهٔ عملیاتی عرضه نمی‌کند",
                "دوام پایدار checkpoint را ادعا نمی‌کند",
            ),
            "de": (
                "Pre-Alpha-Vertragskern",
                "keine API zur Verteilungsanpassung",
                "keine produktionsreifen Datenadapter",
                "keine dauerhafte Checkpoint-Persistenz",
            ),
        }
        for locale, phrases in required.items():
            content = README_PATHS[locale].read_text(encoding="utf-8")
            normalized_content = " ".join(content.split())
            for phrase in phrases:
                with self.subTest(locale=locale, phrase=phrase):
                    self.assertIn(phrase, normalized_content)

    def test_source_manifest_includes_all_package_landing_files(self) -> None:
        manifest = (PYTHON_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            manifest,
            [
                "include LICENSE",
                "include README.md",
                "include README.fa.md",
                "include README.de.md",
                "include src/veridist/py.typed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
