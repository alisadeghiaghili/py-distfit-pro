"""Veridist's deliberately small, stdlib-only public CSV fit surface."""

__version__ = "0.0.0.dev0"

from veridist.adapters.csv_lifetimes import CsvLifetimeLimits, CsvLifetimeSchema
from veridist.engine.provenance import PublicSourceId
from veridist.engine.streaming import IterableDataSource, StreamSource, StreamSourceError
from veridist.execution import ExponentialSourceFitResult, fit_exponential_csv

__all__ = [
    "__version__",
    "CsvLifetimeLimits",
    "CsvLifetimeSchema",
    "ExponentialSourceFitResult",
    "IterableDataSource",
    "PublicSourceId",
    "StreamSource",
    "StreamSourceError",
    "fit_exponential_csv",
]
