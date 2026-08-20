# ADR-0004: Floating-point reproducibility contract

Status: Proposed

## Decision

Do not claim arbitrary bit-identical equality across chunk order, worker count,
machine, or numerical library.  Use stable accumulation and a canonical,
deterministic reduction schedule in reproducible mode.  Publish comparison
tolerances derived from an oracle, conditioning, or convergence criterion.

## Consequences

Bitwise equality may be tested only when input partition, reduction tree,
platform, Python/NumPy/SciPy versions and relevant threading are pinned.
Ordinary backend contracts are numerical agreement plus deterministic metadata.
This preserves meaningful reproducibility without asserting false floating-point
algebra.
