"""Contracts for the complete Veridist pull-request validation workflow."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "v1-ci.yml"
PYPROJECT_PATH = REPOSITORY_ROOT / "python" / "pyproject.toml"
BROWSER_TEST_PATH = (
    REPOSITORY_ROOT / "python" / "tests" / "browser" / "test_exponential_report_rtl.py"
)


class VeridistWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_workflow_runs_for_every_push_and_main_pull_request(self) -> None:
        self.assertIn("name: veridist-ci", self.workflow)
        self.assertIn("  push:", self.workflow)
        self.assertIn("  pull_request:\n    branches: [main]", self.workflow)
        self.assertIn("  workflow_dispatch:", self.workflow)
        self.assertNotIn("v1-foundation", self.workflow)
        self.assertNotIn("paths:", self.workflow)

    def test_static_job_runs_declared_lint_and_strict_type_checks(self) -> None:
        self.assertIn("  static:", self.workflow)
        self.assertIn("name: veridist / static", self.workflow)
        self.assertIn('python-version: "3.11"', self.workflow)
        self.assertIn('python -m pip install -e ".[lint]"', self.workflow)
        self.assertIn("python -m ruff check src tests docs tools", self.workflow)
        self.assertIn("python -m mypy src", self.workflow)

    def test_test_matrix_enforces_branch_coverage_on_all_supported_pythons(self) -> None:
        self.assertIn("  tests:", self.workflow)
        self.assertIn("name: veridist / tests (${{ matrix.python-version }})", self.workflow)
        self.assertIn(
            'python-version: ["3.11", "3.12", "3.13", "3.14"]', self.workflow
        )
        self.assertIn('python -m pip install -e ".[test]"', self.workflow)
        self.assertIn("python -m pytest --cov=veridist --cov-branch", self.workflow)
        self.assertIn("--ignore=tests/docs/test_docs_toolchain.py", self.workflow)
        self.assertIn("--cov-report=json:coverage.json", self.workflow)
        coverage_step = self.workflow.split(
            "      - name: Test with branch coverage", maxsplit=1
        )[1].split("      - name: Enforce coverage gates", maxsplit=1)[0]
        self.assertEqual(
            re.findall(r"--ignore=\S+", coverage_step),
            ["--ignore=tests/docs/test_docs_toolchain.py"],
        )
        self.assertEqual(re.findall(r"--cov=\S+", coverage_step), ["--cov=veridist"])
        for forbidden_ignore in (
            "--ignore=tests/docs",
            "--ignore=tests/docs/test_exponential_report_i18n.py",
            "--ignore=tests/reference",
            "--ignore=tests/contract",
        ):
            with self.subTest(forbidden_ignore=forbidden_ignore):
                self.assertNotIn(forbidden_ignore, re.findall(r"--ignore=\S+", coverage_step))
        self.assertIn("python tools/check_coverage.py --project-root .", self.workflow)
        self.assertIn("--manifest quality/coverage-manifest.json", self.workflow)
        self.assertIn("--coverage-json coverage.json", self.workflow)

    def test_package_job_builds_checks_and_installs_the_wheel_outside_checkout(self) -> None:
        self.assertIn("  package:", self.workflow)
        self.assertIn("name: veridist / package", self.workflow)
        self.assertIn(
            'python -m pip install "build>=1.2,<2" "twine>=6,<7"', self.workflow
        )
        self.assertIn("python -m build --sdist --wheel", self.workflow)
        self.assertIn("python -m twine check dist/*", self.workflow)
        self.assertIn("Verify built distributions remain legacy-isolated", self.workflow)
        self.assertIn("python tools/check_legacy_isolation.py", self.workflow)
        self.assertIn("--artifact", self.workflow)
        self.assertIn("artifacts=(dist/*.whl dist/*.tar.gz)", self.workflow)
        self.assertIn('python -m venv "$RUNNER_TEMP/veridist-wheel"', self.workflow)
        self.assertIn(
            '"$RUNNER_TEMP/veridist-wheel/bin/python" -m pip install dist/*.whl',
            self.workflow,
        )
        self.assertIn('"$RUNNER_TEMP/veridist-wheel/bin/python" -m pip check', self.workflow)
        self.assertIn('cd "$RUNNER_TEMP"', self.workflow)
        self.assertIn("importlib.metadata", self.workflow)
        self.assertIn("py.typed", self.workflow)
        for smoke_contract in (
            "CsvLifetimeLimits",
            "CsvLifetimeSchema",
            "PublicSourceId",
            "fit_exponential_csv",
            "render_exponential_report",
            "ReportLocale.FA",
            "time,event_observed",
            "assert fit.rate == 0.5",
            "assert fit.inference == 'not_provided'",
            "assert fit.censoring_assumption == 'independent_right_censoring'",
            'assert \'lang="fa" dir="rtl"\' in report',
        ):
            with self.subTest(smoke_contract=smoke_contract):
                self.assertIn(smoke_contract, self.workflow)

    def test_docs_job_runs_the_actual_three_locale_toolchain(self) -> None:
        self.assertIn("  docs:", self.workflow)
        self.assertIn("name: veridist / docs", self.workflow)
        self.assertIn('python -m pip install -e ".[docs,test]"', self.workflow)
        commands = (
            'python -m unittest discover -s tests/docs -p "test_*.py" -v',
            "python docs/toolchain.py check",
            "sphinx-build -b gettext -W -n docs/source docs/_build/gettext",
            "sphinx-build -b html -W -n docs/source docs/_build/en/html -D language=en",
            "sphinx-build -b html -W -n docs/source docs/_build/fa/html -D language=fa",
            "sphinx-build -b html -W -n docs/source docs/_build/de/html -D language=de",
            "sphinx-build -b linkcheck -W -n docs/source docs/_build/linkcheck -D language=en",
            "python docs/toolchain.py render docs/_build/en/html en",
            "python docs/toolchain.py render docs/_build/fa/html fa",
            "python docs/toolchain.py render docs/_build/de/html de",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(command, self.workflow)
        self.assertNotIn("sphinx-intl update", self.workflow)
        self.assertNotIn("-b doctest", self.workflow)
        upload_start = self.workflow.index("      - name: Retain rendered documentation evidence")
        upload_block = self.workflow[upload_start : self.workflow.index("  browser-rtl:")]
        self.assertIn("if: always()", upload_block)
        self.assertIn("if-no-files-found: warn", upload_block)
        self.assertLess(
            self.workflow.index("python docs/toolchain.py render docs/_build/de/html de"),
            upload_start,
        )

    def test_browser_job_uses_pinned_extra_and_requires_exact_screenshot_evidence(self) -> None:
        project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        extras = project["project"]["optional-dependencies"]
        self.assertEqual(extras["browser"], ["playwright==1.62.0"])
        self.assertFalse(any("playwright" in dependency for dependency in extras["test"]))

        self.assertIn("  browser-rtl:", self.workflow)
        self.assertIn("name: veridist / browser rtl", self.workflow)
        self.assertIn("key: playwright-${{ runner.os }}-1.62.0", self.workflow)
        self.assertIn('python -m pip install -e ".[test,browser]"', self.workflow)
        self.assertIn("python -m playwright install --with-deps chromium", self.workflow)
        self.assertIn('VERIDIST_BROWSER_TESTS: "1"', self.workflow)
        self.assertIn("tests.browser.test_sphinx_rtl_pages", self.workflow)
        self.assertIn("find artifacts/browser-rtl", self.workflow)
        self.assertIn("-type f -name '*.png' -size +0c", self.workflow)
        self.assertIn('test "${#screenshots[@]}" -eq 2', self.workflow)
        self.assertIn("exponential-report-fa-failure.png", self.workflow)
        self.assertIn("exponential-report-fa-success.png", self.workflow)
        upload_start = self.workflow.index("      - name: Retain browser screenshots")
        upload_block = self.workflow[upload_start : self.workflow.index("  veridist-gate:")]
        self.assertIn("if: always()", upload_block)
        self.assertIn("if-no-files-found: warn", upload_block)
        self.assertLess(
            self.workflow.index("      - name: Verify exact screenshot evidence set"),
            upload_start,
        )

    def test_browser_contract_is_opt_in_and_cleans_default_temporary_artifacts(self) -> None:
        browser_test = BROWSER_TEST_PATH.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("VERIDIST_BROWSER_TESTS") == "1"', browser_test)
        self.assertIn("tempfile.TemporaryDirectory()", browser_test)
        self.assertNotIn("tempfile.mkdtemp()", browser_test)
        for property_name in ("documentDirection", "reportDirection", "reportAlignment"):
            with self.subTest(property_name=property_name):
                self.assertIn(property_name, browser_test)
        self.assertIn('"unicodeBidi": "isolate"', browser_test)
        self.assertIn("self.assertGreater(screenshot.stat().st_size, 0)", browser_test)
        sphinx_browser_test = (
            REPOSITORY_ROOT / "python" / "tests" / "browser" / "test_sphinx_rtl_pages.py"
        ).read_text(encoding="utf-8")
        for required in (
            "sphinx",
            "exponential-right-censoring.html",
            "unicodeBidi",
            "code.literal",
            ".highlight pre",
            "table.docutils",
            ".math",
            'wait_until="load"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, sphinx_browser_test)
        self.assertNotIn(" || ", sphinx_browser_test)

    def test_aggregate_gate_fails_if_any_required_job_does_not_succeed(self) -> None:
        self.assertIn("  veridist-gate:", self.workflow)
        self.assertIn("name: veridist / gate", self.workflow)
        self.assertIn("needs: [static, tests, package, docs, browser-rtl]", self.workflow)
        self.assertIn("if: always()", self.workflow)
        for result in (
            "needs.static.result",
            "needs.tests.result",
            "needs.package.result",
            "needs.docs.result",
            "needs.browser-rtl.result",
        ):
            with self.subTest(result=result):
                self.assertIn(result, self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)
        self.assertNotIn("|| true", self.workflow)

    def test_aggregate_gate_runs_from_the_checkout_root_without_a_checkout(self) -> None:
        self.assertNotIn("defaults:\n  run:\n    working-directory: python", self.workflow)
        for job in ("static", "tests", "package", "docs", "browser-rtl"):
            with self.subTest(job=job):
                job_block = re.search(
                    rf"(?ms)^  {job}:$(.*?)(?=^  \S|\Z)", self.workflow
                )
                self.assertIsNotNone(job_block)
                self.assertIn("working-directory: python", job_block.group(0))

        gate_start = self.workflow.index("  veridist-gate:")
        self.assertNotIn("working-directory:", self.workflow[gate_start:])


if __name__ == "__main__":
    unittest.main()
