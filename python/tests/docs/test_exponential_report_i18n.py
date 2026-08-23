"""I18N-EXP contracts for explicit-locale, safe exponential HTML reports."""

from __future__ import annotations

from decimal import Decimal
from html.parser import HTMLParser
import re
import unicodedata
import unittest

from veridist.domain.lifetimes import ExactLifetime
from veridist.families.exponential import (
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitSuccess,
    fit_exponential,
)
from veridist.reporting.exponential import (
    FAILURE_MESSAGE_CODES,
    REPORT_CATALOGS,
    REPORT_KEYS,
    REPORT_LABEL_KEYS,
    REPORT_HEADINGS,
    REPORT_TITLES,
    ReportLocale,
    render_exponential_report,
)


class _ReportFactParser(HTMLParser):
    """Extract machine facts without coupling tests to HTML layout."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.machine_values: dict[str, str] = {}
        self.label_keys: dict[str, str] = {}
        self.failure_message_codes: list[str] = []
        self.failure_messages: dict[str, str] = {}
        self._active_failure_code: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "div":
            return
        attributes = dict(attrs)
        key = attributes.get("data-report-key")
        if key is not None:
            machine_value = attributes.get("data-machine-value")
            label_key = attributes.get("data-label-key")
            if machine_value is None or label_key is None:
                raise AssertionError("report fact has incomplete machine metadata")
            if key in self.machine_values:
                raise AssertionError(f"duplicate report fact: {key}")
            self.machine_values[key] = machine_value
            self.label_keys[key] = label_key
        failure_message_code = attributes.get("data-failure-message-code")
        if failure_message_code is not None:
            self.failure_message_codes.append(failure_message_code)
            self._active_failure_code = failure_message_code

    def handle_data(self, data: str) -> None:
        if self._active_failure_code is not None:
            self.failure_messages[self._active_failure_code] = (
                self.failure_messages.get(self._active_failure_code, "") + data
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._active_failure_code is not None:
            self._active_failure_code = None


def _parse_report(html: str) -> _ReportFactParser:
    parser = _ReportFactParser()
    parser.feed(html)
    parser.close()
    return parser


def _all_results() -> list[ExponentialFitSuccess | ExponentialFitFailure]:
    return [fit_exponential((ExactLifetime(Decimal("2")),))] + [
        ExponentialFitFailure(code, 0, 0, 0.0)
        if code is ExponentialFitFailureCode.EMPTY_SAMPLE
        else ExponentialFitFailure(code, 2, 0, 3.0)
        if code is ExponentialFitFailureCode.NO_OBSERVED_EVENTS
        else ExponentialFitFailure(code, 1, 1, 0.0)
        if code is ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD
        else ExponentialFitFailure(code, 2, 0, None)
        for code in ExponentialFitFailureCode
    ]


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
        self.assertEqual({html.count("data-report-key=") for html in rendered.values()}, {14})
        self.assertNotIn("<script>", rendered[ReportLocale.EN])

    def test_i18n_exp04_has_exact_stable_keys_and_semantic_facts_for_every_result(self) -> None:
        self.assertEqual(
            REPORT_KEYS,
            (
                "status", "family", "parameterization", "location", "rate", "mean",
                "observation_count", "event_count", "censored_count", "total_time",
                "log_likelihood", "inference", "censoring_assumption", "failure_code",
            ),
        )
        for result in _all_results():
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
        self.assertIn("برازش نمایی", fa)
        self.assertIn('<html lang="en" dir="ltr">', render_exponential_report(result, ReportLocale.EN))
        self.assertIn('<html lang="de" dir="ltr">', render_exponential_report(result, ReportLocale.DE))
        for token in ("exponential", "rate", "2.0", "not_provided"):
            self.assertIn(f'<bdi dir="ltr" class="latin">{token}</bdi>', fa)

    def test_i18n_exp06_parser_has_exact_machine_parity_for_success_and_each_failure(self) -> None:
        for result in _all_results():
            parsed = {
                locale: _parse_report(render_exponential_report(result, locale)) for locale in ReportLocale
            }
            self.assertEqual({tuple(item.machine_values) for item in parsed.values()}, {REPORT_KEYS})
            self.assertEqual(
                {tuple(item.machine_values.items()) for item in parsed.values()},
                {tuple(next(iter(parsed.values())).machine_values.items())},
            )
            self.assertEqual(
                {tuple(item.label_keys.items()) for item in parsed.values()},
                {tuple(REPORT_LABEL_KEYS.items())},
            )
            expected_failure = (
                "failure.NONE"
                if not isinstance(result, ExponentialFitFailure)
                else FAILURE_MESSAGE_CODES[result.code]
            )
            self.assertEqual(
                {tuple(item.failure_message_codes) for item in parsed.values()}, {(expected_failure,)},
            )

    def test_i18n_exp07_catalogs_are_closed_nfc_translated_and_immutable(self) -> None:
        self.assertEqual(tuple(REPORT_LABEL_KEYS), REPORT_KEYS)
        self.assertEqual(set(REPORT_CATALOGS), set(ReportLocale))
        for catalog in REPORT_CATALOGS.values():
            self.assertEqual(tuple(catalog), REPORT_KEYS)
            for text in catalog.values():
                self.assertTrue(text)
                self.assertEqual(unicodedata.normalize("NFC", text), text)
        self.assertIn("برازش نمایی", " ".join(REPORT_CATALOGS[ReportLocale.FA].values()))
        self.assertIn("Exponentialanpassung", " ".join(REPORT_CATALOGS[ReportLocale.DE].values()))
        with self.assertRaises(TypeError):
            REPORT_CATALOGS[ReportLocale.EN] = {}  # type: ignore[index]
        with self.assertRaises(TypeError):
            REPORT_CATALOGS[ReportLocale.EN]["status"] = "mutated"  # type: ignore[index]
        baseline = render_exponential_report(fit_exponential((ExactLifetime(Decimal("2")),)), ReportLocale.EN)
        self.assertIn("Exponential fit", baseline)

    def test_i18n_exp08_rejects_subclass_tampered_and_unknown_results_without_sentinel_echo(self) -> None:
        class SuccessSubclass(ExponentialFitSuccess):
            pass

        valid = fit_exponential((ExactLifetime(Decimal("2")),))
        assert isinstance(valid, ExponentialFitSuccess)
        subclass = SuccessSubclass(
            valid.rate, valid.observation_count, valid.event_count, valid.total_time,
            valid.mean, valid.log_likelihood, valid.censored_count, valid.provenance,
        )
        sentinel = "<script>malicious-sentinel</script>"
        object.__setattr__(valid, "family", sentinel)
        for result in (object(), subclass, valid):
            with self.assertRaises((TypeError, ValueError)) as captured:
                render_exponential_report(result, ReportLocale.EN)  # type: ignore[arg-type]
            self.assertNotIn(sentinel, str(captured.exception))

    def test_i18n_exp09_all_machine_values_formula_and_latin_identifiers_are_ltr_isolated(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))
        html = render_exponential_report(result, ReportLocale.FA)
        parsed = _parse_report(html)
        for token in (*REPORT_KEYS, *parsed.machine_values.values(), "r*log(rate)-rate*tau"):
            self.assertIn(f'<bdi dir="ltr" class="latin">{token}</bdi>', html)

    def test_i18n_exp10_titles_headings_and_failure_messages_are_localized_not_catalog_codes(self) -> None:
        self.assertEqual(set(REPORT_TITLES), set(ReportLocale))
        self.assertEqual(set(REPORT_HEADINGS), set(ReportLocale))
        self.assertEqual(len(set(REPORT_TITLES.values())), len(ReportLocale))
        self.assertEqual(len(set(REPORT_HEADINGS.values())), len(ReportLocale))
        for locale in ReportLocale:
            self.assertTrue(REPORT_TITLES[locale])
            self.assertTrue(REPORT_HEADINGS[locale])
            self.assertEqual(unicodedata.normalize("NFC", REPORT_TITLES[locale]), REPORT_TITLES[locale])
            self.assertEqual(unicodedata.normalize("NFC", REPORT_HEADINGS[locale]), REPORT_HEADINGS[locale])
        for result in _all_results():
            parsed = {
                locale: _parse_report(render_exponential_report(result, locale)) for locale in ReportLocale
            }
            messages = {locale: next(iter(value.failure_messages.values())) for locale, value in parsed.items()}
            self.assertEqual(len(messages), len(ReportLocale))
            self.assertEqual(len(set(messages.values())), len(ReportLocale))
            for locale, message in messages.items():
                self.assertTrue(message.strip())
                self.assertEqual(unicodedata.normalize("NFC", message), message)
                self.assertNotIn("failure.", message)
                if locale is ReportLocale.FA:
                    self.assertIsNone(re.search(r"[A-Za-z]", message))

    def test_i18n_exp11_has_explicit_start_alignment_and_persian_right_alignment(self) -> None:
        html = render_exponential_report(fit_exponential((ExactLifetime(Decimal("2")),)), ReportLocale.FA)
        self.assertIn("text-align:start", html)
        self.assertIn('[dir="rtl"] .report{text-align:right}', html)
