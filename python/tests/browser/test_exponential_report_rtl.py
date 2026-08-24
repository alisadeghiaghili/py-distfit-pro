"""Browser contracts for rendered Persian exponential reports."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.exponential import ExponentialFitFailure, fit_exponential
from veridist.reporting.exponential import ReportLocale, render_exponential_report


class ExponentialReportBrowserContracts(unittest.TestCase):
    """Computed style, bidi isolation, and screenshot evidence are release contracts."""

    @unittest.skipUnless(
        os.environ.get("VERIDIST_BROWSER_TESTS") == "1",
        "browser evidence runs only in the dedicated Chromium job",
    )
    def test_rtl_exp01_success_and_failure_have_rendered_rtl_and_ltr_machine_values(self) -> None:
        from playwright.sync_api import sync_playwright

        success = fit_exponential((ExactLifetime(Decimal("2")),))
        failure = fit_exponential((RightCensoredLifetime(Decimal("2")),))
        assert isinstance(failure, ExponentialFitFailure)
        configured_directory = os.environ.get("VERIDIST_BROWSER_ARTIFACT_DIR")
        with ExitStack() as resources:
            if configured_directory is None:
                artifact_directory = Path(resources.enter_context(tempfile.TemporaryDirectory()))
            else:
                artifact_directory = Path(configured_directory)
                artifact_directory.mkdir(parents=True, exist_ok=True)
            with sync_playwright() as playwright:
                executable = os.environ.get("VERIDIST_BROWSER_EXECUTABLE")
                launch_options = {} if executable is None else {"executable_path": executable}
                browser = playwright.chromium.launch(**launch_options)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 900})
                    for name, result in (("success", success), ("failure", failure)):
                        page.set_content(
                            render_exponential_report(result, ReportLocale.FA), wait_until="load"
                        )
                        computed = page.evaluate(
                            """() => ({
                                documentDirection: document.documentElement.dir,
                                reportDirection: getComputedStyle(
                                    document.querySelector('.report')
                                ).direction,
                                reportAlignment: getComputedStyle(
                                    document.querySelector('.report')
                                ).textAlign,
                                latin: [...document.querySelectorAll('.latin')].map((item) => ({
                                    direction: getComputedStyle(item).direction,
                                    unicodeBidi: getComputedStyle(item).unicodeBidi,
                                })),
                            })"""
                        )
                        self.assertEqual(computed["documentDirection"], "rtl")
                        self.assertEqual(computed["reportDirection"], "rtl")
                        self.assertEqual(computed["reportAlignment"], "right")
                        self.assertTrue(computed["latin"])
                        self.assertTrue(
                            all(
                                item == {"direction": "ltr", "unicodeBidi": "isolate"}
                                for item in computed["latin"]
                            )
                        )
                        page.evaluate("window.scrollTo(0, 0)")
                        screenshot = artifact_directory / f"exponential-report-fa-{name}.png"
                        page.screenshot(path=str(screenshot), full_page=True)
                        self.assertGreater(screenshot.stat().st_size, 0)
                finally:
                    browser.close()


if __name__ == "__main__":
    unittest.main()
