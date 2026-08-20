# ADR-0011: i18n and multilingual documentation

Status: Accepted

Owner: Ali Sadeghi Aghili

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- complete
English, Persian and German documentation is a v1 release requirement.

Scope: English, Persian and German public documentation and localized reports;
not translated API identifiers, statistical-core strings or unbounded locales.

## Context

v1 must ship complete English, Persian and German documentation and localized
user-facing reports.  Translation must not contaminate statistical computation
or create three diverging API descriptions.

## Decision

The statistical core returns stable structured codes, fields and evidence, not
translated strings or logs.  API/report adapters localize those codes.  English
is the canonical documentation source; a translation-parity manifest tracks
every English page, stable anchor, message key, example and locale equivalent.
Fallback locale behaviour is explicit in the API and missing required keys fail
CI rather than silently displaying English.

Examples have one executable source or a parity-tested shared harness.  API
reference is generated once from canonical signatures with localized narrative.
Persian pages declare RTL direction and receive visual rendering smoke tests
covering mixed LTR identifiers, code, tables, equations and numerals.

## Consequences

Localization becomes a release gate, with warning-as-error builds and link
checks for `en`, `fa` and `de`.  New public keys/pages require all three locale
entries in the same change or an explicitly approved, visibly surfaced fallback.

## Evidence, tests and dependencies

Evidence is the direct product requirement, not a claim that competitors lack
multilingual documentation. Required tests are locale-key parity, explicit
fallback, Unicode/formatting, RTL screenshots, warning-as-error builds,
linkcheck and canonical executable examples. Dependencies: ADR-0013, a
translation manifest, locale owners/reviewers and CI rendering artifacts.
