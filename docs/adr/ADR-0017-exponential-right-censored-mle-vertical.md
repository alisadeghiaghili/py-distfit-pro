# ADR-0017: Exponential right-censored MLE vertical

Status: Accepted

Owner: Ali Sadeghi Aghili

## Context

The first callable statistical vertical must establish a small, auditable
contract before broader family coverage is attempted.  A one-parameter
exponential model with independently right-censored lifetimes has a closed-form
maximum-likelihood estimate and fixed-size sufficient statistics.  It is a
useful vertical only if its statistical limits, streaming behaviour, and
localized presentation are explicit.

## Decision

Version `0.1.0a1` implements only the exponential lifetime family with fixed
`loc=0` and canonical rate parameter `rate > 0`.  An observation is either an
exact lifetime or an independent right-censored lifetime, each with a finite
time `>= 0`.  No weights, covariates, truncation, left/interval censoring,
free location, selection procedure, goodness-of-fit result, or confidence
interval is provided by this vertical.

For event count `r` and total time on test `tau`, the log likelihood is

`r log(rate) - rate tau`.

When `r > 0` and `tau > 0`, the point estimate is `rate = r / tau`.  If all
observations are censored, the fit returns a typed statistical failure rather
than treating zero as a successful rate.  When there is an event but
`tau = 0`, the likelihood is unbounded and there is no finite MLE; this is also
a typed statistical failure.  The independent-censoring assumption is visible
in the result and is not empirically verifiable by the library.

The reducer keeps fixed O(1) state: event count, observation count, and a
compensated total-time sum.  Its canonical input order is reproducible; changes
to partition/reduction order are assessed with a documented numerical
tolerance rather than a bit-identical claim.  It retains neither raw rows nor
source paths or URIs.  Fit success/failure is distinct from engine execution
failure and all public result objects are frozen and slotted.

The core emits locale-neutral codes and facts.  Reports are explicit-locale,
pure HTML renderings with no silent fallback.  English, Persian, and German
have key parity.  Persian reports declare `lang="fa" dir="rtl"`; Latin code,
API names, URLs, numbers, units, and formulae receive LTR bidi isolation.
Network fonts are prohibited.  Until a licensed pinned font is bundled, the
RTL evidence is computed-style assertions plus a browser screenshot artifact,
not a pixel baseline.

## Evidence

The independent statistical contract is checked against NIST's exponential
distribution guidance, R `stats` exponential documentation, and SciPy's
censored-data fitting documentation, retrieved and pinned on 2026-08-23.
Reference test values use committed Decimal/rational calculations, not legacy
code or production code as an oracle.  Legacy exponential implementation is
only an inspected scenario under LM-002 and is never imported, copied, or used
as a numeric oracle.

## Dependencies

ADR-0002, ADR-0003, ADR-0004, ADR-0005, ADR-0006, ADR-0010, ADR-0011,
ADR-0013, ADR-0016.

## Consequences

The first release has a deliberately narrow surface and offers point estimates
only: `inference=not_provided` is an honest capability declaration, not an
approximate interval.  It gives a reusable family/reducer/report seam without
claiming generic censoring inference or production-scale out-of-core support.

## References

- NIST/SEMATECH e-Handbook, Exponential Distribution, retrieved 2026-08-23.
- R `stats` documentation for `dexp`, `pexp`, `qexp`, and `rexp`, retrieved
  2026-08-23.
- SciPy documentation for `CensoredData` and distribution `fit`, retrieved
  2026-08-23.

