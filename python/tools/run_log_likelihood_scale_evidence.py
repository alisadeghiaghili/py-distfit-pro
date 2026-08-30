"""Generate retained one-pass exact-state log-likelihood scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import tracemalloc
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
    """Yield generated observations once; no source or chunk is retained."""

    for start in range(0, rows, size):
        yield (0.0 for _ in range(min(size, rows - start)))


def _cell(rows: int, chunk_size: int) -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
    result = reduce_log_likelihood_chunks(FamilyId.NORMAL, _chunks(rows, chunk_size), mu=0, sigma=1)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not isinstance(result, LogLikelihoodSuccess) or result.observation_count != rows:
        raise RuntimeError("generated stream did not reduce successfully")
    contribution = -0.5 * __import__("math").log(2.0 * __import__("math").pi)
    numerator, denominator = contribution.as_integer_ratio()
    units = rows * numerator * ((1 << 1074) // denominator)
    return {
        "rows": rows,
        "chunk_size": chunk_size,
        "one_pass": True,
        "state": {
            "observation_count": rows,
            "total_units": units,
            "total_units_bit_length": abs(units).bit_length(),
            "bound_bits": 2162,
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
        "schema_version": "1",
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
