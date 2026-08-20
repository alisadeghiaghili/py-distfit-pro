# ADR-0002: Statistical correctness and capability matrix

Status: Proposed

## Decision

Every public family × estimator × data-semantics × inference operation is an
explicit capability-matrix cell.  A supported cell names assumptions,
parameterization, algorithm, diagnostics, numerical/approximation contract,
references and tests.  An unsupported cell raises a typed capability error.

## Rationale and consequences

This prevents silent substitution of MLE for MoM, uncensored likelihood for
censored data, or a normal critical value for another family.  It makes the
initial matrix small by design and requires documentation and tests whenever a
cell changes.  Citations must identify a specific result/table/equation; where
none exists, the API reports the limitation or uses a documented simulation
procedure.
