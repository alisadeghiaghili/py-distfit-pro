"""Real-browser contracts for rendered Sphinx locale pages."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PYTHON_ROOT / "docs" / "source"


@unittest.skipUnless(
    os.environ.get("VERIDIST_BROWSER_TESTS") == "1",
    "Sphinx browser evidence runs only in the dedicated Chromium job",
)
class SphinxRtlBrowserContracts(unittest.TestCase):
    def test_rtl_doc01_built_farsi_and_german_pages_have_computed_direction_contracts(self) -> None:
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {locale: root / locale for locale in ("fa", "de")}
            for locale, output in outputs.items():
                result = subprocess.run(
                    [sys.executable, "-m", "sphinx", "-b", "html", "-W", "-n", str(SOURCE_ROOT), str(output), "-D", f"language={locale}"],
                    check=False, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            with sync_playwright() as playwright:
                executable = os.environ.get("VERIDIST_BROWSER_EXECUTABLE")
                browser = playwright.chromium.launch(
                    **({} if executable is None else {"executable_path": executable})
                )
                try:
                    page = browser.new_page()
                    page.goto((outputs["fa"] / "exponential-right-censoring.html").as_uri())
                    fa = page.evaluate("""() => ({
                      lang: document.documentElement.lang, dir: document.documentElement.dir,
                      body: {direction:getComputedStyle(document.body).direction, align:getComputedStyle(document.body).textAlign},
                      code: getComputedStyle(document.querySelector('code')).direction,
                      pre: getComputedStyle(document.querySelector('pre') || document.querySelector('code')).direction,
                      table: getComputedStyle(document.querySelector('table')).direction,
                      math: getComputedStyle(document.querySelector('[class*=math]') || document.querySelector('table')).direction,
                      isolate: getComputedStyle(document.querySelector('code')).unicodeBidi,
                    })""")
                    self.assertEqual(fa["lang"], "fa")
                    self.assertEqual(fa["dir"], "rtl")
                    self.assertEqual(fa["body"], {"direction": "rtl", "align": "right"})
                    self.assertEqual([fa[key] for key in ("code", "pre", "table", "math")], ["ltr"] * 4)
                    self.assertIn(fa["isolate"], {"isolate", "isolate-override"})
                    page.goto((outputs["de"] / "exponential-right-censoring.html").as_uri())
                    self.assertEqual(page.evaluate("document.documentElement.lang"), "de")
                    self.assertEqual(page.evaluate("getComputedStyle(document.body).direction"), "ltr")
                finally:
                    browser.close()
