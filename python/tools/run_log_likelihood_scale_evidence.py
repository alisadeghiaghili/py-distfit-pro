"""Generate retained one-pass exact-state log-likelihood scale evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

from veridist import DataSourceMetadata, IterableDataSource, Replayability
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks

ROWS = (10_000, 100_000, 1_000_000)
BUDGETS = (1_024, 8_192, 65_536)
FAMILY_CASES = (
    (FamilyId.NORMAL, 0.0, {"mu": 0.0, "sigma": 1.0}, "-0x1.d67f1c864beb4p-1"),
    (FamilyId.GAMMA, 1.0, {"shape": 2.0, "scale": 1.0}, "-0x1.0000000000000p+0"),
    (FamilyId.WEIBULL_MIN, 1.0, {"shape": 2.0, "scale": 1.0}, "-0x1.3a37a020b8c22p-2"),
    (FamilyId.LOGNORMAL, 1.0, {"mu_log": 0.0, "sigma_log": 1.0}, "-0x1.d67f1c864beb4p-1"),
    (FamilyId.GUMBEL_RIGHT, 0.0, {"location": 0.0, "scale": 1.0}, "-0x1.0000000000000p+0"),
)


def _head(root: Path) -> str:
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise RuntimeError("refusing evidence from a dirty checkout")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _chunks(rows: int, size: int, observation: float):
    """Return generated chunks whose underlying traversal is observable."""

    return _GeneratedChunks(rows, size, observation)


class _GeneratedChunks:
    """Generate one fixed supported-family observation while tracking traversal."""

    def __init__(self, rows: int, size: int, observation: float) -> None:
        self.rows = rows
        self.size = size
        self.observation = observation
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
            yield self.observation


def _oracle_units(rows: int, contribution_hex: str) -> int:
    contribution = float.fromhex(contribution_hex)
    numerator, denominator = contribution.as_integer_ratio()
    return rows * numerator * ((1 << 1074) // denominator)


def _rss_bytes() -> int:
    """Return current process RSS on every CI platform or fail closed."""

    if sys.platform == "win32":

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        getter = ctypes.windll.psapi.GetProcessMemoryInfo
        getter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        getter.restype = ctypes.c_int
        ok = getter(process, ctypes.byref(counters), ctypes.sizeof(counters))
        if not ok:
            raise RuntimeError("cannot obtain Windows process RSS")
        return int(counters.WorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _cell(
    rows: int,
    chunk_size: int,
    *,
    family: FamilyId = FamilyId.NORMAL,
    observation: float = 0.0,
    parameters: dict[str, float] | None = None,
    contribution_hex: str = "-0x1.d67f1c864beb4p-1",
) -> dict[str, object]:
    if parameters is None:
        parameters = {"mu": 0.0, "sigma": 1.0}
    chunks = _chunks(rows, chunk_size, observation)
    source = IterableDataSource(
        chunks,
        DataSourceMetadata(
            source_id=f"scale-{family.value}-{rows}-{chunk_size}",
            schema_version="1",
            provenance_schema_version="1",
            replayability=Replayability.SINGLE_PASS,
            redaction_reason="generated",
        ),
    )
    tracemalloc.start()
    rss_before = _rss_bytes()
    started = time.perf_counter()
    result = reduce_log_likelihood_chunks(family, source, **parameters)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_bytes()
    if (
        not isinstance(result, LogLikelihoodSuccess)
        or result.family is not family
        or result.observation_count != rows
        or chunks.iterator_acquisitions != 1
        or chunks.observation_yields != rows
    ):
        raise RuntimeError("generated stream did not reduce successfully")
    units = _oracle_units(rows, contribution_hex)
    expected = float(Fraction(units, 1 << 1074))
    if result.total_log_likelihood.hex() != expected.hex():
        raise RuntimeError("returned total does not match independent exact oracle")
    return {
        "family": family.value,
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
        "throughput_rows_per_second": rows / elapsed if elapsed else float(rows),
        "memory": {
            "tracemalloc_peak_bytes": peak,
            "rss_peak_bytes": rss_after,
            "rss_delta_bytes": max(0, rss_after - rss_before),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sha = _head(root)
    value: dict[str, object] = {
        "schema_version": "4",
        "run": {
            "git_sha": sha,
            "candidate_git_sha": sha,
            "git_dirty": False,
            "generator": "fixed-supported-family-v1",
            "source_contract": "public-iterable-data-source-v1",
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "platform": platform.platform(),
        },
        "cells": [
            _cell(
                rows,
                budget,
                family=family,
                observation=observation,
                parameters=parameters,
                contribution_hex=contribution,
            )
            for family, observation, parameters, contribution in FAMILY_CASES
            for rows in ROWS
            for budget in BUDGETS
        ],
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
