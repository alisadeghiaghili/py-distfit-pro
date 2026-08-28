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
                    [
                        sys.executable,
                        "-m",
                        "sphinx",
                        "-b",
                        "html",
                        "-W",
                        "-n",
                        str(SOURCE_ROOT),
                        str(output),
                        "-D",
                        f"language={locale}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            with sync_playwright() as playwright:
                executable = os.environ.get("VERIDIST_BROWSER_EXECUTABLE")
                browser = playwright.chromium.launch(
                    **({} if executable is None else {"executable_path": executable})
                )
                try:
                    page = browser.new_page()
                    page.goto(
                        (outputs["fa"] / "exponential-right-censoring.html").as_uri(),
                        wait_until="load",
                    )
                    fa = page.evaluate("""() => ({
                      lang: document.documentElement.lang, dir: document.documentElement.dir,
                      body: {
                        direction: getComputedStyle(document.body).direction,
                        align: getComputedStyle(document.body).textAlign,
                      },
                      networkScripts: [...document.scripts]
                        .map((script) => script.src)
                        .filter((source) => /^https?:/i.test(source)),
                      exemplars: Object.fromEntries([
                        ['code', 'code.literal'],
                        ['pre', '.highlight pre'],
                        ['table', 'table.docutils'],
                        ['math', '.math'],
                      ].map(([name, selector]) => {
                        const element = document.querySelector(selector);
                        if (element === null) {
                          throw new Error(`missing required exemplar: ${selector}`);
                        }
                        const style = getComputedStyle(element);
                        return [name, {direction: style.direction, unicodeBidi: style.unicodeBidi}];
                      })),
                    })""")
                    self.assertEqual(fa["lang"], "fa")
                    self.assertEqual(fa["dir"], "rtl")
                    self.assertEqual(fa["body"], {"direction": "rtl", "align": "right"})
                    self.assertEqual(fa["networkScripts"], [])
                    self.assertEqual(
                        fa["exemplars"],
                        {
                            name: {"direction": "ltr", "unicodeBidi": "isolate"}
                            for name in ("code", "pre", "table", "math")
                        },
                    )
                    page.goto(
                        (outputs["de"] / "exponential-right-censoring.html").as_uri(),
                        wait_until="load",
                    )
                    de = page.evaluate("""() => ({
                      lang: document.documentElement.lang, dir: document.documentElement.dir,
                      body: getComputedStyle(document.body).direction,
                      exemplarCount: ['code.literal', '.highlight pre', 'table.docutils', '.math']
                        .map((selector) => document.querySelector(selector)).filter(Boolean).length,
                    })""")
                    self.assertEqual(
                        de,
                        {"lang": "de", "dir": "ltr", "body": "ltr", "exemplarCount": 4},
                    )
                finally:
                    browser.close()
