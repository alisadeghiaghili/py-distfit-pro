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

    def test_each_locale_states_the_same_experimental_vertical_and_limits(self) -> None:
        required = {
            "en": (
                "pre-alpha contract kernel",
                "experimental rate-only exponential MLE",
                "exact and independently right-censored lifetimes",
                "Inference is not provided",
                "typed failures",
                "does not ship production data adapters",
                "does not claim production out-of-core execution",
            ),
            "fa": (
                "هستهٔ قراردادی پیش‌آلفا",
                "برآوردگر آزمایشی MLE نمایی فقط برای پارامتر نرخ",
                "طول عمرهای دقیق و راست‌سانسورشدهٔ مستقل",
                "استنباط ارائه نمی‌شود",
                "شکست‌های نوع‌دار",
                "آداپتور دادهٔ عملیاتی عرضه نمی‌کند",
                "اجرای برون‌حافظه‌ای عملیاتی را ادعا نمی‌کند",
            ),
            "de": (
                "Pre-Alpha-Vertragskern",
                "experimentellen, rein ratenparametrisierten exponentiellen MLE",
                "exakte und unabhängig rechtszensierte Lebensdauern",
                "Inferenz wird nicht bereitgestellt",
                "typisierte Fehlschläge",
                "keine produktionsreifen Datenadapter",
                "keine produktive Out-of-Core-Ausführung",
            ),
        }
        for locale, phrases in required.items():
            content = README_PATHS[locale].read_text(encoding="utf-8")
            normalized_content = " ".join(content.split())
            for phrase in phrases:
                with self.subTest(locale=locale, phrase=phrase):
                    self.assertIn(phrase, normalized_content)

    def test_all_locales_publish_the_same_executable_quickstart(self) -> None:
        snippets: dict[str, str] = {}
        for locale, path in README_PATHS.items():
            content = path.read_text(encoding="utf-8")
            matches = re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)
            with self.subTest(locale=locale):
                self.assertEqual(len(matches), 1)
            snippets[locale] = matches[0].strip()

        self.assertEqual(len(set(snippets.values())), 1)
        quickstart = snippets["en"]
        for required in (
            "ExactLifetime",
            "RightCensoredLifetime",
            "fit_exponential",
            "render_exponential_report",
            "ReportLocale.FA",
            "assert fit.rate == 0.5",
            'assert fit.inference == "not_provided"',
            'assert fit.censoring_assumption == "independent_right_censoring"',
            'assert \'lang="fa" dir="rtl"\' in report',
        ):
            with self.subTest(required=required):
                self.assertIn(required, quickstart)

    def test_persian_rtl_wrapper_never_contains_ltr_code_fences(self) -> None:
        content = README_PATHS["fa"].read_text(encoding="utf-8")
        self.assertIn('<div lang="fa" dir="rtl">', content)
        for fence in re.finditer(r"```(?:console|python)?", content):
            prefix = content[: fence.start()]
            with self.subTest(fence=fence.group(0), offset=fence.start()):
                self.assertEqual(prefix.count('<div lang="fa" dir="rtl">'), prefix.count("</div>"))

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
