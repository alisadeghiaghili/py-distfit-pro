"""Veridist's deliberately small, stdlib-only public CSV fit surface."""

__version__ = "0.9.1"

from veridist.adapters.csv_lifetimes import CsvLifetimeLimits, CsvLifetimeSchema
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.provenance import PublicSourceId
from veridist.engine.streaming import IterableDataSource, StreamSource, StreamSourceError
from veridist.execution import (
    ExponentialSourceFitResult,
    fit_exponential_checkpointed_chunks,
    fit_exponential_csv,
)
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

__all__ = [
    "__version__",
    "CsvLifetimeLimits",
    "CsvLifetimeSchema",
    "DataSourceMetadata",
    "ExponentialSourceFitResult",
    "FamilyId",
    "IterableDataSource",
    "PublicSourceId",
    "Replayability",
    "StreamSource",
    "StreamSourceError",
    "fit_exponential_csv",
    "fit_exponential_checkpointed_chunks",
    "reduce_log_likelihood_chunks",
]
