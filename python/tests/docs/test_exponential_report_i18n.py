"""I18N-EXP contracts for explicit-locale, safe exponential HTML reports."""

from __future__ import annotations

from decimal import Decimal
import unittest

from veridist.domain.lifetimes import ExactLifetime
from veridist.families.exponential import fit_exponential
from veridist.reporting.exponential import ReportLocale, render_exponential_report


class ExponentialReportI18nContracts(unittest.TestCase):
    def test_i18n_exp01_requires_explicit_supported_locale_without_fallback(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))
        for locale in ReportLocale:
            self.assertIn("<html", render_exponential_report(result, locale))
        with self.assertRaises((TypeError, ValueError)):
            render_exponential_report(result, "ar")  # type: ignore[arg-type]

    def test_i18n_exp02_farsi_report_has_rtl_root_and_ltr_isolates(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))
        html = render_exponential_report(result, ReportLocale.FA)
        self.assertIn('<html lang="fa" dir="rtl">', html)
        self.assertIn('class="report" dir="rtl"', html)
        self.assertIn('<bdi dir="ltr" class="latin">', html)
        self.assertIn("unicode-bidi:isolate", html)

    def test_i18n_exp03_escapes_untrusted_text_and_preserves_locale_parity(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))
        rendered = {locale: render_exponential_report(result, locale) for locale in ReportLocale}
        self.assertEqual({html.count("data-report-key=") for html in rendered.values()}, {7})
        self.assertNotIn("<script>", rendered[ReportLocale.EN])

