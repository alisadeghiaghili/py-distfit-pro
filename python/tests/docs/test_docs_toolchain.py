from __future__ import annotations

import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PYTHON_ROOT / "docs"
SOURCE_ROOT = DOCS_ROOT / "source"
MANIFEST_PATH = DOCS_ROOT / "i18n" / "parity-manifest.json"
TOOLCHAIN_PATH = DOCS_ROOT / "toolchain.py"


def load_toolchain():
    spec = importlib.util.spec_from_file_location("veridist_docs_toolchain", TOOLCHAIN_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("docs/toolchain.py cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocsToolchainContractTests(unittest.TestCase):
    def test_docs_extra_declares_the_accepted_toolchain(self) -> None:
        pyproject = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = pyproject["project"]["optional-dependencies"]["docs"]
        self.assertTrue(any(item.startswith("Sphinx") for item in dependencies))
        self.assertTrue(any(item.startswith("myst-parser") for item in dependencies))
        self.assertTrue(any(item.startswith("sphinx-intl") for item in dependencies))

    def test_manifest_has_stable_pages_messages_and_required_locales(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["canonical_locale"], "en")
        self.assertEqual(manifest["required_locales"], ["en", "fa", "de"])

        page_ids = [page["id"] for page in manifest["pages"]]
        message_ids = [message["id"] for message in manifest["messages"]]
        self.assertEqual(len(page_ids), len(set(page_ids)))
        self.assertEqual(len(message_ids), len(set(message_ids)))
        self.assertTrue(all(re.fullmatch(r"DOC-PAGE-[A-Z0-9-]+", item) for item in page_ids))
        self.assertTrue(all(re.fullmatch(r"DOC-MSG-[A-Z0-9-]+", item) for item in message_ids))

        for page in manifest["pages"]:
            source = (SOURCE_ROOT / page["source"]).read_text(encoding="utf-8")
            self.assertIn(f"({page['anchor']})=", source)

    def test_sphinx_configuration_uses_myst_gettext_and_stable_uuids(self) -> None:
        configuration = runpy.run_path(str(SOURCE_ROOT / "conf.py"))
        self.assertEqual(configuration["extensions"], ["myst_parser"])
        self.assertEqual(configuration["locale_dirs"], ["../locales"])
        self.assertTrue(configuration["gettext_uuid"])
        self.assertFalse(configuration["gettext_compact"])
        self.assertEqual(configuration["html_css_files"], ["rtl.css"])

    def test_translations_are_complete_and_never_silent_fallbacks(self) -> None:
        toolchain = load_toolchain()
        report = toolchain.validate_parity(DOCS_ROOT)
        self.assertEqual(report["locales"], ["fa", "de"])
        self.assertGreater(report["message_count"], 0)
        self.assertEqual(report["missing"], {})
        self.assertEqual(report["fallbacks"], {})

    def test_manifest_matches_every_real_gettext_message_for_tracked_pages(self) -> None:
        toolchain = load_toolchain()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=os.environ.get("VERIDIST_SCRATCH")) as temporary_directory:
            output = Path(temporary_directory) / "gettext"
            result = subprocess.run(
                [sys.executable, "-m", "sphinx", "-b", "gettext", "-W", "-n", str(SOURCE_ROOT), str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = {
                source
                for page in manifest["pages"]
                for source in toolchain._parse_po(output / f"{Path(page['source']).stem}.pot")
            }
        declared = {message["source"] for message in manifest["messages"]}
        self.assertEqual(declared, actual)

    def test_persian_is_rtl_german_is_ltr_and_render_checker_is_strict(self) -> None:
        toolchain = load_toolchain()
        self.assertEqual(toolchain.direction_for("fa"), "rtl")
        self.assertEqual(toolchain.direction_for("de"), "ltr")
        self.assertEqual(toolchain.direction_for("en"), "ltr")
        css = (SOURCE_ROOT / "_static" / "rtl.css").read_text(encoding="utf-8")
        self.assertIn('html[dir="rtl"]', css)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "index.html").write_text(
                '<html lang="fa" dir="rtl"><head><link href="rtl.css"></head>'
                '<body><code>x = 1</code></body></html>',
                encoding="utf-8",
            )
            toolchain.assert_rendered_direction(output, "fa")
            toolchain.assert_rtl_smoke(output)
            (output / "index.html").write_text(
                '<html lang="fa"><body>missing direction</body></html>', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "dir=rtl"):
                toolchain.assert_rendered_direction(output, "fa")

    def test_first_vertical_docs_are_callable_narrow_and_honest(self) -> None:
        api_page = (SOURCE_ROOT / "api.md").read_text(encoding="utf-8")
        tutorial_page = (SOURCE_ROOT / "exponential-right-censoring.md").read_text(encoding="utf-8")
        self.assertIn("fit_exponential", api_page)
        self.assertIn("ReportLocale", api_page)
        self.assertIn("not_provided", api_page)
        self.assertIn("independent right censoring", tutorial_page)
        self.assertIn("no confidence interval", tutorial_page)
        self.assertIn("fixed O(1) accumulator state", tutorial_page)
        self.assertNotIn("autofunction", api_page)
        self.assertNotIn("automodule", api_page)

    def test_example_has_one_registered_executable_source(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["examples"]), 1)
        example = manifest["examples"][0]
        example_path = SOURCE_ROOT / example["source"]
        result = runpy.run_path(str(example_path))
        self.assertEqual(result["EXAMPLE_RESULT"], {
            "rate": "0.5",
            "observation_count": "3",
            "event_count": "2",
            "censored_count": "1",
        })
        for page_name in example["pages"]:
            page = (SOURCE_ROOT / page_name).read_text(encoding="utf-8")
            self.assertIn(example["source"], page)


if __name__ == "__main__":
    unittest.main()
