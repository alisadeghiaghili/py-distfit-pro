"""Release-metadata contracts for the nested Veridist package."""

from __future__ import annotations

import re
import tomllib
import unicodedata
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
                "public CSV path is strict",
                "one iterator pass",
                "not a generic CSV reader",
            ),
            "fa": (
                "هستهٔ قراردادی پیش‌آلفا",
                "برآوردگر آزمایشی MLE نمایی فقط برای پارامتر نرخ",
                "طول عمرهای دقیق و راست‌سانسورشدهٔ مستقل",
                "استنباط ارائه نمی‌شود",
                "شکست‌های نوع‌دار",
                "مسیر CSV عمومی آن سخت‌گیرانه است",
                "یک گذر از iterator",
                "CSV عمومی",
            ),
            "de": (
                "Pre-Alpha-Vertragskern",
                "experimentellen, rein ratenparametrisierten exponentiellen MLE",
                "exakte und unabhängig rechtszensierte Lebensdauern",
                "Inferenz wird nicht bereitgestellt",
                "typisierte Fehlschläge",
                "öffentliche CSV-Pfad ist strikt",
                "einen Iterator-Durchlauf",
                "allgemeines CSV",
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
            "fit_exponential_csv",
            "CsvLifetimeSchema",
            "CsvLifetimeLimits",
            "PublicSourceId",
            "assert fit.rate == 0.5",
            'assert fit.inference == "not_provided"',
            'assert fit.censoring_assumption == "independent_right_censoring"',
            'path.write_text("time,event_observed\\n1,1\\n1,0\\n", encoding="utf-8")',
        ):
            with self.subTest(required=required):
                self.assertIn(required, quickstart)

        namespace: dict[str, object] = {}
        exec(compile(quickstart, "package-landing-quickstart", "exec"), namespace)

    def test_scale_evidence_claim_is_equivalent_and_limited_in_every_locale(self) -> None:
        expected = {
            "en": "measured 10k/100k/1m by 32KiB/64KiB/128KiB matrix",
            "fa": "ماتریس اندازه‌گیری‌شدهٔ 10k/100k/1m ردیف و بودجه‌های 32KiB/64KiB/128KiB",
            "de": "gemessene Matrix aus 10k/100k/1m Zeilen und 32KiB/64KiB/128KiB",
        }
        for locale, phrase in expected.items():
            with self.subTest(locale=locale):
                content = " ".join(README_PATHS[locale].read_text(encoding="utf-8").split())
                self.assertIn(phrase, content)

    def test_all_landing_pages_are_clean_nfc_without_retired_claims_or_bidi_controls(self) -> None:
        retired_claims = (
            "does not provide a distribution-fitting API",
            "API برازش توزیع ارائه نمی‌کند",
            "keine API zur Verteilungsanpassung",
        )
        mojibake_markers = ("Ã", "Â", "Ø", "Ù", "�")
        bidi_controls = tuple(chr(codepoint) for codepoint in range(0x202A, 0x202F)) + tuple(
            chr(codepoint) for codepoint in range(0x2066, 0x206A)
        )
        for locale, path in README_PATHS.items():
            content = path.read_text(encoding="utf-8")
            with self.subTest(locale=locale, contract="nfc"):
                self.assertEqual(content, unicodedata.normalize("NFC", content))
            with self.subTest(locale=locale, contract="mojibake"):
                self.assertFalse(any(marker in content for marker in mojibake_markers))
            with self.subTest(locale=locale, contract="retired"):
                self.assertFalse(any(claim in content for claim in retired_claims))
            with self.subTest(locale=locale, contract="bidi-controls"):
                self.assertFalse(any(control in content for control in bidi_controls))

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
