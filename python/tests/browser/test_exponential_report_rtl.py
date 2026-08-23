"""Browser contracts for rendered Persian exponential reports."""

from __future__ import annotations

import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from playwright.sync_api import sync_playwright

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.exponential import ExponentialFitFailure, fit_exponential
from veridist.reporting.exponential import ReportLocale, render_exponential_report


class ExponentialReportBrowserContracts(unittest.TestCase):
    """Computed style, bidi isolation, and screenshot evidence are release contracts."""

    def test_rtl_exp01_success_and_failure_have_rendered_rtl_and_ltr_machine_values(self) -> None:
        success = fit_exponential((ExactLifetime(Decimal("2")),))
        failure = fit_exponential((RightCensoredLifetime(Decimal("2")),))
        assert isinstance(failure, ExponentialFitFailure)
        artifact_directory = Path(os.environ.get("VERIDIST_BROWSER_ARTIFACT_DIR", tempfile.mkdtemp()))
        artifact_directory.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                for name, result in (("success", success), ("failure", failure)):
                    page.set_content(render_exponential_report(result, ReportLocale.FA), wait_until="load")
                    computed = page.evaluate(
                        """() => ({
                            documentDirection: document.documentElement.dir,
                            reportDirection: getComputedStyle(document.querySelector('.report')).direction,
                            reportAlignment: getComputedStyle(document.querySelector('.report')).textAlign,
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
                        all(item == {"direction": "ltr", "unicodeBidi": "isolate"} for item in computed["latin"])
                    )
                    page.screenshot(path=str(artifact_directory / f"exponential-report-fa-{name}.png"), full_page=True)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
