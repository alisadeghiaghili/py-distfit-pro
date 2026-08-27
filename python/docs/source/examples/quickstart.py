"""Canonical executable strict CSV exponential documentation example."""

from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    RESULT = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    )

FIT = RESULT.fit
assert isinstance(FIT, ExponentialFitSuccess)
EXAMPLE_RESULT = {
    "rate": repr(FIT.rate),
    "observation_count": str(FIT.observation_count),
    "event_count": str(FIT.event_count),
    "censored_count": str(FIT.censored_count),
    "actual_pass_count": str(RESULT.execution.provenance.execution.passes.actual_pass_count),
}

if __name__ == "__main__":
    print(EXAMPLE_RESULT)
