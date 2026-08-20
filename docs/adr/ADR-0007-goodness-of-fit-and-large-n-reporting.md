# ADR-0007: Goodness-of-fit-and-large-n-reporting

Status: Proposed

## Decision

For fitted parameters, default GoF uses a refit parametric Monte-Carlo null for
supported uncensored cells.  Analytic p-values require a cited
family/estimator/sample-size result.  Every result names statistic, calibration
path, replicate count, seed policy, fit failures and Monte-Carlo uncertainty.

At large n, reports foreground effect size, uncertainty and a user-declared
decision threshold.  A configurable warning may mark p-values as often
decision-unhelpful; it does not suppress a valid p-value merely because
`n >= 10_000`.

## Consequences

Calibration is published over stated grids, not advertised as a universal
theorem.  Sketch-based GoF requires its own error analysis and calibration.
