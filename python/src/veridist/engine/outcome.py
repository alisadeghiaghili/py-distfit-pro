"""Immutable coverage facts and honestly labelled execution outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, cast

from veridist.engine.errors import FailureCode


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class RowRange:
    """A non-empty, half-open range of stable source-row offsets."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _require_non_negative_integer("start", self.start)
        _require_non_negative_integer("stop", self.stop)
        if self.stop <= self.start:
            raise ValueError("row range must be non-empty")

    @property
    def row_count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True, order=True)
class KnownExtent:
    """A half-open expected source extent; an empty source is valid."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _require_non_negative_integer("start", self.start)
        _require_non_negative_integer("stop", self.stop)
        if self.stop < self.start:
            raise ValueError("known extent stop must not precede start")

    @property
    def row_count(self) -> int:
        return self.stop - self.start


def _canonical_ranges(ranges: tuple[RowRange, ...]) -> tuple[RowRange, ...]:
    if type(ranges) is not tuple:
        raise TypeError("processed_ranges must be a tuple")
    canonical: list[RowRange] = []
    for item in ranges:
        if type(item) is not RowRange:
            raise TypeError("processed_ranges must contain RowRange values")
        if not canonical:
            canonical.append(item)
            continue
        previous = canonical[-1]
        if item.start < previous.start:
            raise ValueError("processed_ranges must be ordered")
        if item.start < previous.stop:
            raise ValueError("processed_ranges must not overlap or repeat")
        if item.start == previous.stop:
            canonical[-1] = RowRange(previous.start, item.stop)
        else:
            canonical.append(item)
    return tuple(canonical)


def _validate_chunk_counts(accepted_chunk_count: int, empty_chunk_count: int) -> None:
    _require_non_negative_integer("accepted_chunk_count", accepted_chunk_count)
    _require_non_negative_integer("empty_chunk_count", empty_chunk_count)
    if empty_chunk_count > accepted_chunk_count:
        raise ValueError("empty_chunk_count cannot exceed accepted_chunk_count")


@dataclass(frozen=True, slots=True)
class KnownCoverage:
    """Exact processed and missing coverage within a known final extent."""

    extent: KnownExtent
    processed_ranges: tuple[RowRange, ...]
    accepted_chunk_count: int
    empty_chunk_count: int

    def __post_init__(self) -> None:
        if type(self.extent) is not KnownExtent:
            raise TypeError("extent must be a KnownExtent")
        canonical = _canonical_ranges(self.processed_ranges)
        for item in canonical:
            if item.start < self.extent.start or item.stop > self.extent.stop:
                raise ValueError("processed range lies outside the known extent")
        _validate_chunk_counts(self.accepted_chunk_count, self.empty_chunk_count)
        if canonical and self.accepted_chunk_count == 0:
            raise ValueError("processed ranges require an accepted chunk")
        object.__setattr__(self, "processed_ranges", canonical)

    @property
    def processed_row_count(self) -> int:
        return sum(item.row_count for item in self.processed_ranges)

    @property
    def missing_ranges(self) -> tuple[RowRange, ...]:
        missing: list[RowRange] = []
        cursor = self.extent.start
        for item in self.processed_ranges:
            if cursor < item.start:
                missing.append(RowRange(cursor, item.start))
            cursor = item.stop
        if cursor < self.extent.stop:
            missing.append(RowRange(cursor, self.extent.stop))
        return tuple(missing)

    @property
    def missing_row_count(self) -> int:
        return self.extent.row_count - self.processed_row_count


@dataclass(frozen=True, slots=True)
class UnknownMissingRanges:
    """Exact progress when failure prevents learning the source's final extent."""

    processed_ranges: tuple[RowRange, ...]
    accepted_chunk_count: int
    empty_chunk_count: int
    reason: FailureCode = field(init=False, default=FailureCode.MISSING_RANGE_UNKNOWN)

    def __post_init__(self) -> None:
        canonical = _canonical_ranges(self.processed_ranges)
        _validate_chunk_counts(self.accepted_chunk_count, self.empty_chunk_count)
        if canonical and self.accepted_chunk_count == 0:
            raise ValueError("processed ranges require an accepted chunk")
        object.__setattr__(self, "processed_ranges", canonical)

    @property
    def processed_row_count(self) -> int:
        return sum(item.row_count for item in self.processed_ranges)


class FailureStage(StrEnum):
    """Stable stage at which the primary execution failure occurred."""

    PREFLIGHT = "preflight"
    DELIVERY = "delivery"
    REDUCTION = "reduction"
    CHECKPOINT = "checkpoint"
    SINK = "sink"
    FINALIZATION = "finalization"
    CANCELLATION = "cancellation"


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """A context-free public cause retaining only typed, allowlisted facts."""

    code: FailureCode
    stage: FailureStage

    def __post_init__(self) -> None:
        if not isinstance(self.code, FailureCode):
            raise TypeError("code must be a FailureCode")
        if not isinstance(self.stage, FailureStage):
            raise TypeError("stage must be a FailureStage")


class OutcomeStatus(StrEnum):
    """The exhaustive public execution-outcome tags."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CompleteOutcome:
    """Execution reached a known final extent without a failure."""

    coverage: KnownCoverage

    def __post_init__(self) -> None:
        if type(self.coverage) is not KnownCoverage:
            raise TypeError("complete coverage must be known")
        if self.coverage.missing_ranges:
            raise ValueError("complete outcome cannot contain missing ranges")

    @property
    def status(self) -> OutcomeStatus:
        return OutcomeStatus.COMPLETE

    @property
    def complete(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class PartialOutcome:
    """Known incomplete execution progress, never a scientific fit value."""

    coverage: KnownCoverage
    failure: FailureRecord

    def __post_init__(self) -> None:
        if type(self.coverage) is not KnownCoverage:
            raise TypeError("partial coverage must be known")
        if type(self.failure) is not FailureRecord:
            raise TypeError("failure must be a FailureRecord")
        if self.coverage.processed_row_count == 0:
            raise ValueError("partial outcome requires processed rows")
        if not self.coverage.missing_ranges:
            raise ValueError("partial outcome requires exact missing ranges")

    @property
    def status(self) -> OutcomeStatus:
        return OutcomeStatus.PARTIAL

    @property
    def complete(self) -> bool:
        return False


Coverage: TypeAlias = KnownCoverage | UnknownMissingRanges


@dataclass(frozen=True, slots=True)
class FailedOutcome:
    """Execution failed without claiming a partial scientific result."""

    coverage: Coverage
    failure: FailureRecord

    def __post_init__(self) -> None:
        if type(self.coverage) not in {KnownCoverage, UnknownMissingRanges}:
            raise TypeError("failed coverage must be known or explicitly unknown")
        if type(self.failure) is not FailureRecord:
            raise TypeError("failure must be a FailureRecord")

    @property
    def status(self) -> OutcomeStatus:
        return OutcomeStatus.FAILED

    @property
    def complete(self) -> bool:
        return False


ExecutionOutcome: TypeAlias = CompleteOutcome | PartialOutcome | FailedOutcome


def classify_execution_outcome(
    coverage: Coverage,
    failure: FailureRecord | None,
) -> ExecutionOutcome:
    """Classify final execution facts without inventing coverage or a fit value."""

    if type(coverage) not in {KnownCoverage, UnknownMissingRanges}:
        raise TypeError("coverage must be a supported coverage value")
    if failure is not None and type(failure) is not FailureRecord:
        raise TypeError("failure must be a FailureRecord or None")
    if failure is None:
        if type(coverage) is KnownCoverage and not coverage.missing_ranges:
            return CompleteOutcome(coverage)
        raise ValueError("incomplete or unknown coverage requires a typed failure")
    if type(coverage) is UnknownMissingRanges:
        return FailedOutcome(coverage, failure)
    known_coverage = cast(KnownCoverage, coverage)
    if known_coverage.processed_row_count > 0 and known_coverage.missing_ranges:
        return PartialOutcome(known_coverage, failure)
    return FailedOutcome(known_coverage, failure)


__all__ = [
    "CompleteOutcome",
    "Coverage",
    "ExecutionOutcome",
    "FailedOutcome",
    "FailureRecord",
    "FailureStage",
    "KnownCoverage",
    "KnownExtent",
    "OutcomeStatus",
    "PartialOutcome",
    "RowRange",
    "UnknownMissingRanges",
    "classify_execution_outcome",
]
