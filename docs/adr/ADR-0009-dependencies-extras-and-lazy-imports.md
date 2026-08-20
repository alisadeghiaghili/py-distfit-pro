# ADR-0009: Dependencies-extras-and-lazy-imports

Status: Proposed

## Decision

The statistical runtime depends only on NumPy and SciPy.  Dataframe, Arrow,
parallel, plotting, warehouse and documentation/rendering integrations are
versioned optional extras and are imported lazily at the boundary that needs
them.

## Consequences

Importing the core must not initialize plotting, localization rendering,
parallel workers or optional clients.  Missing extras produce actionable typed
errors.  Dependency versions and optional paths are included in provenance and
platform CI.
