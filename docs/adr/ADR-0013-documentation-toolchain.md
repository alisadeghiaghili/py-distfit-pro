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
