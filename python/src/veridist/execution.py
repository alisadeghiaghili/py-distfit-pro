"""Typed landing surface for source-to-family execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from veridist.engine.outcome import CompleteOutcome
from veridist.engine.provenance import ExecutionReport
from veridist.families.exponential import ExponentialFit


@dataclass(frozen=True, slots=True)
class ExponentialSourceFitResult:
    """Closed result shape for a completed fit or a failed execution."""

    fit: ExponentialFit | None
    execution: ExecutionReport

    def __post_init__(self) -> None:
        if type(self.execution) is not ExecutionReport:
            raise TypeError("execution must be ExecutionReport")
        if (self.fit is not None) is not isinstance(self.execution.outcome, CompleteOutcome):
            raise ValueError("fit presence must match complete execution")


def fit_exponential_source(adapter: object) -> ExponentialSourceFitResult:
    """Reserve the one-pass orchestration seam without implementing delivery."""

    del adapter
    return cast(ExponentialSourceFitResult, None)


__all__ = ["ExponentialSourceFitResult", "fit_exponential_source"]
