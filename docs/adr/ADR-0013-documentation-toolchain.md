# ADR-0013: Documentation toolchain

Status: Accepted

Owner: documentation/community lead (TBD)

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- use
Sphinx + MyST + gettext/`sphinx-intl` for the EN/FA/DE v1 toolchain.

## Context

ADR-0011 requires complete EN/FA/DE documentation, executable examples, stable
anchors and Persian RTL QA. The current v1 toolchain is NOT IMPLEMENTED; a
command such as `python -m docs.build` would be fictitious.

## Decision

Recommend Sphinx with MyST Markdown, gettext extraction and `sphinx-intl`.
English is canonical; Persian/German are generated from one source tree with
stable anchors/API identifiers. Configure RTL assets and screenshot capture for
Persian. An alternative needs a superseding ADR proving equivalent gettext,
doctest, linkcheck and RTL capabilities.

## Scope, dependencies and tests

Dependencies: a Sphinx config under `docs/source`, MyST, gettext catalogs,
locale owners, screenshot runner, example harness and CI artifacts. This ADR is
Accepted, but its configuration and the following planned commands remain
**NOT IMPLEMENTED**:

```text
sphinx-build -b html -W -n docs/source docs/_build/en/html -D language=en
sphinx-build -b html -W -n docs/source docs/_build/fa/html -D language=fa
sphinx-build -b html -W -n docs/source docs/_build/de/html -D language=de
sphinx-build -b linkcheck -W docs/source docs/_build/linkcheck -D language=en
sphinx-build -b gettext -W docs/source docs/_build/gettext
pytest docs/tests -m 'docs or examples or rtl'
```

CI must test translation-key parity, fallback, Unicode/formatting, executable
examples, `-W` builds, linkcheck and committed RTL screenshots from the first
vertical slice.

## Consequences

Documentation/i18n/examples become continuous gates, not late release work.

## Implementation evidence addendum -- 2026-08-22

The Sphinx/MyST/gettext structure, stable parity manifest, EN/FA/DE catalogs,
RTL stylesheet, canonical executable example and strict rendered-direction
checker are now present. Structural documentation tests and locale parity pass
locally. The pushed CI configuration installs the declared documentation extra,
builds gettext and all three HTML locales with warnings fatal, runs linkcheck,
validates rendered direction and retains HTML artifacts.

The complete local Sphinx run remains **UNVERIFIED**: on 2026-08-22, the
available Sphinx 8.2.3 environment failed before the gettext build because it
lacked the declared `myst_parser` dependency. The remote CI result is also
unverified. Browser-rendered Persian screenshot evidence is not
implemented, so this addendum does not claim complete visual RTL QA or complete
v1 documentation.

## Implementation evidence addendum -- 2026-08-25

The earlier 2026-08-22 limitation is historical. In an isolated E-drive
environment with the declared documentation extra installed, gettext, English,
Persian and German HTML builds completed with `-W -n`; English linkcheck and
the rendered semantic/direction checks also passed. Exact real POT message IDs
match the complete FA/DE catalogs, and rendered pages are checked against silent
English fallback.

The opt-in Playwright test produced exactly two nonempty Persian HTML report
screenshots locally with Edge 151 and verified computed RTL/right alignment plus
LTR/isolate handling for Latin fragments on both success and failure reports.
The workflow pins Playwright 1.62.0 and requires its matched Chromium, but that
remote CI execution is still **UNVERIFIED**. This evidence covers report HTML,
not PDF; no bundled-font or pixel-baseline claim is made.
