"""I18N-EXP contracts for explicit-locale, safe exponential HTML reports."""

from __future__ import annotations

from decimal import Decimal
import unicodedata
import unittest

from veridist.domain.lifetimes import ExactLifetime
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitFailureCode, fit_exponential
from veridist.reporting.exponential import REPORT_KEYS, ReportLocale, render_exponential_report


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

    def test_i18n_exp04_has_exact_stable_keys_and_semantic_facts_for_every_result(self) -> None:
        results = [fit_exponential((ExactLifetime(Decimal("2")),))] + [
            ExponentialFitFailure(code, 0, 0, 0.0)
            if code is ExponentialFitFailureCode.EMPTY_SAMPLE
            else ExponentialFitFailure(code, 2, 0, 3.0)
            if code is ExponentialFitFailureCode.NO_OBSERVED_EVENTS
            else ExponentialFitFailure(code, 1, 1, 0.0)
            if code is ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD
            else ExponentialFitFailure(code, 2, 0, None)
            for code in ExponentialFitFailureCode
        ]
        self.assertEqual(REPORT_KEYS, ("status", "family", "parameterization", "location", "rate", "mean", "observation_count", "event_count", "censored_count", "total_time", "log_likelihood", "inference", "censoring_assumption", "failure_code"))
        for result in results:
            rendered = {locale: render_exponential_report(result, locale) for locale in ReportLocale}
            for html in rendered.values():
                for key in REPORT_KEYS:
                    self.assertIn(f'data-report-key="{key}"', html)
                    self.assertIn("data-machine-value=", html)
            self.assertEqual({html.count("data-report-key=") for html in rendered.values()}, {len(REPORT_KEYS)})

    def test_i18n_exp05_direction_unicode_and_latin_isolation_are_explicit(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))
        fa = render_exponential_report(result, ReportLocale.FA)
        self.assertEqual(unicodedata.normalize("NFC", fa), fa)
        self.assertIn("برازش", fa)
        self.assertIn('<html lang="en" dir="ltr">', render_exponential_report(result, ReportLocale.EN))
        self.assertIn('<html lang="de" dir="ltr">', render_exponential_report(result, ReportLocale.DE))
        for token in ("exponential", "rate", "2.0", "not_provided"):
            self.assertIn(f'<bdi dir="ltr" class="latin">{token}</bdi>', fa)
