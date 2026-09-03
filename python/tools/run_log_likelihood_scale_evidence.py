"""Generate retained one-pass exact-state log-likelihood scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks

ROWS = (10_000, 100_000, 1_000_000)
BUDGETS = (1_024, 8_192, 65_536)


def _head(root: Path) -> str:
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise RuntimeError("refusing evidence from a dirty checkout")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _chunks(rows: int, size: int):
    """Return a generated source that fails closed if iterated more than once."""

    return _OnePassGeneratedChunks(rows, size)


class _OnePassGeneratedChunks:
    """Generate Normal(0,1) observations while recording actual source traversal."""

    def __init__(self, rows: int, size: int) -> None:
        self.rows = rows
        self.size = size
        self.iterator_acquisitions = 0
        self.observation_yields = 0

    def __iter__(self):
        self.iterator_acquisitions += 1
        if self.iterator_acquisitions != 1:
            raise RuntimeError("generated source was iterated more than once")
        for start in range(0, self.rows, self.size):
            yield self._observations(min(self.size, self.rows - start))

    def _observations(self, count: int):
        for _ in range(count):
            self.observation_yields += 1
            yield 0.0


def _oracle_units(rows: int) -> int:
    contribution = -0.5 * math.log(2.0 * math.pi)
    numerator, denominator = contribution.as_integer_ratio()
    return rows * numerator * ((1 << 1074) // denominator)


def _cell(rows: int, chunk_size: int) -> dict[str, object]:
    chunks = _chunks(rows, chunk_size)
    tracemalloc.start()
    started = time.perf_counter()
    result = reduce_log_likelihood_chunks(FamilyId.NORMAL, chunks, mu=0, sigma=1)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if (
        not isinstance(result, LogLikelihoodSuccess)
        or result.family is not FamilyId.NORMAL
        or result.observation_count != rows
        or chunks.iterator_acquisitions != 1
        or chunks.observation_yields != rows
    ):
        raise RuntimeError("generated stream did not reduce successfully")
    units = _oracle_units(rows)
    expected = float(Fraction(units, 1 << 1074))
    if result.total_log_likelihood.hex() != expected.hex():
        raise RuntimeError("returned total does not match independent exact oracle")
    return {
        "rows": rows,
        "chunk_size": chunk_size,
        "one_pass": {
            "iterator_acquisitions": chunks.iterator_acquisitions,
            "observation_yields": chunks.observation_yields,
        },
        "oracle": {
            "oracle_total_units": units,
            "oracle_total_units_bit_length": abs(units).bit_length(),
            "bound_bits": 2162,
        },
        "actual": {
            "observation_count": rows,
            "total_log_likelihood": result.total_log_likelihood,
            "total_log_likelihood_hex": result.total_log_likelihood.hex(),
        },
        "elapsed_seconds": elapsed,
        "memory": {"tracemalloc_peak_bytes": peak},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sha = _head(root)
    value: dict[str, object] = {
        "schema_version": "2",
        "run": {"git_sha": sha, "git_dirty": False, "generator": "normal-zero-v1"},
        "cells": [_cell(rows, budget) for rows in ROWS for budget in BUDGETS],
    }
    value["artifact_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if _head(root) != sha:
        raise SystemExit("refusing evidence after checkout changed")
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
