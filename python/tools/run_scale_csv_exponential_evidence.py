"""Generate deterministic retained evidence for the CSV exponential vertical."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from veridist.adapters.csv_lifetimes import CsvLifetimeLimits, CsvLifetimeSchema
from veridist.engine.provenance import PublicSourceId
from veridist.execution import fit_exponential_csv
from veridist.families.exponential import ExponentialFitSuccess

TEMPORARY_ROOT = Path("E:/Project/veridist-tmp")
SCHEMA = CsvLifetimeSchema("time", "event_observed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generate(path: Path, rows: int) -> tuple[int, Decimal]:
    """Write the fixture and independently calculate its Decimal sufficient facts."""

    events = 0
    total = Decimal(0)
    with path.open("w", encoding="utf-8", newline="") as target:
        target.write("time,event_observed\n")
        for index in range(rows):
            time_value = Decimal((index % 997) + 1) / Decimal(1000)
            event = 0 if index % 3 == 0 else 1
            target.write(f"{time_value:f},{event}\n")
            total += time_value
            events += event
    return events, total


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _rss_bytes() -> int | None:
    try:
        import resource  # type: ignore[import-not-found]
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _canonical_digest(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(item) for item in value.split(",")]
    if len(budgets) < 3 or len(set(budgets)) != len(budgets) or any(item <= 0 for item in budgets):
        raise ValueError("chunk budgets must contain at least three distinct positive integers")
    return budgets


def _cell(
    path: Path,
    *,
    rows: int,
    chunk_bytes: int,
    expected_events: int,
    expected_total: Decimal,
) -> dict[str, object]:
    before_rss = _rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    result = fit_exponential_csv(
        path,
        schema=SCHEMA,
        source_id=PublicSourceId(
            f"src_{hashlib.md5(str(rows).encode(), usedforsecurity=False).hexdigest()}"
        ),
        limits=CsvLifetimeLimits(chunk_bytes, chunk_bytes),
    )
    elapsed = time.perf_counter() - started
    _, trace_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_rss = _rss_bytes()
    if not result.execution.outcome.complete or not isinstance(result.fit, ExponentialFitSuccess):
        raise RuntimeError("scale fixture did not produce a complete exponential fit")
    fit = result.fit
    expected_rate = Decimal(expected_events) / expected_total
    actual_rate = Decimal(str(fit.rate))
    absolute_error = abs(actual_rate - expected_rate)
    relative_error = absolute_error / abs(expected_rate)
    execution = result.execution.provenance.execution
    return {
        "rows": rows,
        "chunk_bytes": chunk_bytes,
        "max_inflight_bytes": chunk_bytes,
        "source": {"bytes": path.stat().st_size, "sha256": _sha256(path)},
        "observed": {
            "actual_pass_count": execution.passes.actual_pass_count,
            "max_passes": execution.passes.max_passes,
            "accepted_chunk_count": result.execution.outcome.coverage.accepted_chunk_count,
            "processed_row_count": result.execution.outcome.coverage.processed_row_count,
            "peak_inflight_bytes": execution.buffer.peak_inflight_bytes,
            "largest_retained_chunk_bytes": execution.buffer.largest_retained_chunk_bytes,
            "backpressure_event_count": execution.buffer.backpressure_event_count,
        },
        "fit": {
            "observation_count": fit.observation_count,
            "event_count": fit.event_count,
            "total_time": fit.total_time,
            "rate": fit.rate,
            "expected_event_count": expected_events,
            "expected_total_time": str(expected_total),
            "expected_rate": str(expected_rate),
            "absolute_rate_error": float(absolute_error),
            "relative_rate_error": float(relative_error),
        },
        "memory": {
            "tracemalloc_peak_bytes": trace_peak,
            "rss_peak_bytes": after_rss,
            "rss_delta_bytes": None
            if before_rss is None or after_rss is None
            else max(0, after_rss - before_rss),
        },
        "elapsed_seconds": elapsed,
    }


def _cell_request(request: tuple[str, int, int, int, Decimal]) -> dict[str, object]:
    """Pickle-safe worker boundary; paths stay process-local and unrecorded."""

    path, rows, budget, events, total = request
    return _cell(
        Path(path),
        rows=rows,
        chunk_bytes=budget,
        expected_events=events,
        expected_total=total,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="10000,100000,1000000")
    parser.add_argument("--chunk-bytes", default="32768,65536,131072")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if _git(root, "status", "--porcelain"):
        raise SystemExit("refusing evidence run from a dirty checkout")
    rows = [int(item) for item in args.rows.split(",")]
    budgets = _parse_budgets(args.chunk_bytes)
    if not rows or any(item <= 0 for item in rows) or len(set(rows)) != len(rows):
        raise SystemExit("rows must contain distinct positive integers")
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    TEMPORARY_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    cells: list[dict[str, object]] = []
    chunks_by_row: dict[int, int] = {}
    for row_count in rows:
        fixture = TEMPORARY_ROOT / f"scale-csv-exp-v1-{row_count}.csv"
        expected_events, expected_total = _generate(fixture, row_count)
        requests = [
            (str(fixture), row_count, budget, expected_events, expected_total) for budget in budgets
        ]
        if args.workers == 1:
            row_cells = [_cell_request(request) for request in requests]
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(requests))) as executor:
                row_cells = list(executor.map(_cell_request, requests))
        cells.extend(row_cells)
        chunks_by_row[row_count] = int(row_cells[0]["observed"]["accepted_chunk_count"])
    value: dict[str, object] = {
        "schema_version": "1",
        "run": {
            "git_sha": _git(root, "rev-parse", "HEAD"),
            "git_dirty": False,
            "utc_started": started,
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "platform": platform.platform(),
            "measurement_workers": args.workers,
        },
        "generator": {"formula_version": "1", "temporary_root": "redacted"},
        "cells": cells,
        "operation_evidence": {
            "rows": rows,
            "accepted_chunks": [chunks_by_row[row] for row in rows],
        },
    }
    value["artifact_sha256"] = _canonical_digest(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote retained evidence: {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
