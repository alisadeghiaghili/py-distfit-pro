"""One-pass source-to-family execution orchestration."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeAdapterError,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
)
from veridist.domain.lifetimes import LifetimeObservation
from veridist.engine.data_source import DataSourceLike, ExecutionPlan, SpoolPolicy, plan_passes
from veridist.engine.delivery import (
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    DeliveryValidator,
)
from veridist.engine.errors import EngineContractError
from veridist.engine.outcome import (
    CompleteOutcome,
    FailureStage,
    KnownCoverage,
    KnownExtent,
    RowRange,
    UnknownMissingRanges,
    classify_execution_outcome,
)
from veridist.engine.pass_budget import PassEnforcer
from veridist.engine.provenance import (
    AdapterProvenance,
    CheckpointNotUsed,
    EstimatorProvenance,
    ExactComputation,
    ExecutionProvenance,
    ExecutionReport,
    PublicSourceId,
    RngPolicy,
    RngProvenance,
    SourceMutationStatus,
    SourceProvenance,
    SourceRedaction,
    SourceRedactionReason,
    SpoolNotUsed,
    failure_record_from_error,
    snapshot_execution_observation,
)
from veridist.families.exponential import ExponentialFit, fit_exponential_chunks


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
    """Fit the exponential vertical through exactly one sequential source pass."""

    if type(adapter) is not CsvLifetimeAdapter:
        raise TypeError("adapter must be CsvLifetimeAdapter")
    # CsvLifetimeAdapter exposes an immutable metadata property; the planning
    # protocol's legacy writable attribute is semantically narrower.
    plan = plan_passes(
        cast(DataSourceLike, adapter), required_passes=1, spool=SpoolPolicy.disabled()
    )
    passes = PassEnforcer(max_passes=1)
    buffer = BoundedChunkBuffer(
        chunk_bytes=adapter.limits.chunk_bytes,
        max_inflight_bytes=adapter.limits.max_inflight_bytes,
    )
    validator = DeliveryValidator(adapter.source_id.value)
    stage = FailureStage.PREFLIGHT
    error: EngineContractError | None = None
    fit: ExponentialFit | None = None
    coverage: KnownCoverage | UnknownMissingRanges
    expected_row_stop = 0
    expected_chunk_count = 0
    iterator: object | None = None

    def delivered_payloads() -> Generator[tuple[LifetimeObservation, ...], None, None]:
        """Deliver one validated payload at a time and release it before advancing."""

        nonlocal stage, expected_row_stop, expected_chunk_count, iterator
        acquired = iter(adapter.iter_chunks())
        iterator = acquired
        try:
            for chunk in acquired:
                stage = FailureStage.DELIVERY
                validator.accept(chunk.envelope)
                # These terminal facts are deliberately maintained separately
                # from DeliveryValidator, so finish validates rather than merely
                # echoes the validator's own counters.
                expected_row_stop = chunk.envelope.row_stop
                expected_chunk_count += 1
                lease = BufferedChunk(envelope=chunk.envelope, payload=chunk.observations)
                buffer.put(lease)
                received = buffer.get()
                try:
                    payload = received.payload
                    if type(payload) is not tuple:
                        raise TypeError("CSV chunk payload must be a tuple")
                    yield cast(tuple[LifetimeObservation, ...], payload)
                finally:
                    received.release()
            stage = FailureStage.FINALIZATION
            validator.finish(
                expected_row_stop=expected_row_stop,
                expected_chunk_count=expected_chunk_count,
            )
        finally:
            close = getattr(acquired, "close", None)
            if callable(close):
                close()
            # This is idempotent; it drains/reclaims a lease if a consumer,
            # validator, or reducer fails between put and get.
            buffer.cancel()

    try:
        fit = fit_exponential_chunks(passes.begin_pass(delivered_payloads()))
    except EngineContractError as captured:
        error = captured
        if type(captured) is CsvLifetimeAdapterError:
            stage = FailureStage(captured.phase.value)
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        buffer.cancel()

    if error is None:
        extent = KnownExtent(0, validator.next_offset)
        coverage = KnownCoverage(
            extent,
            (RowRange(0, validator.next_offset),) if validator.next_offset else (),
            validator.accepted_chunks,
            0,
        )
        outcome = classify_execution_outcome(coverage, None)
        assert fit is not None
        mutation = SourceMutationStatus.VERIFIED_UNCHANGED
    else:
        coverage = UnknownMissingRanges(
            (RowRange(0, validator.next_offset),) if validator.next_offset else (),
            validator.accepted_chunks,
            0,
        )
        outcome = classify_execution_outcome(coverage, failure_record_from_error(error, stage))
        fit = None
        mutation = (
            SourceMutationStatus.MISMATCH_DETECTED
            if error.code.value == "SOURCE_REVISION_MISMATCH"
            else SourceMutationStatus.UNAVAILABLE
        )
    execution = ExecutionReport(
        outcome,
        _provenance(adapter, plan, passes, buffer, mutation),
    )
    return ExponentialSourceFitResult(fit, execution)


def fit_exponential_csv(
    path: Path,
    *,
    schema: CsvLifetimeSchema,
    source_id: PublicSourceId,
    limits: CsvLifetimeLimits,
) -> ExponentialSourceFitResult:
    """Construct the strict CSV adapter and execute its single allowed pass."""

    return fit_exponential_source(CsvLifetimeAdapter(path, schema, source_id, limits))


def _provenance(
    adapter: CsvLifetimeAdapter,
    plan: ExecutionPlan,
    passes: PassEnforcer,
    buffer: BoundedChunkBuffer,
    mutation: SourceMutationStatus,
) -> ExecutionProvenance:
    return ExecutionProvenance(
        schema_version="1",
        run_id=f"run_{secrets.token_hex(16)}",
        source=SourceProvenance(
            adapter.source_id,
            "1",
            SourceRedaction(SourceRedactionReason.HASH_UNAVAILABLE),
            mutation,
        ),
        execution=snapshot_execution_observation(
            plan=plan,
            pass_enforcer=passes,
            buffer=buffer,
            adapter=AdapterProvenance(AdapterKind.CSV, "1"),
            spool=SpoolNotUsed(),
        ),
        estimator=EstimatorProvenance(
            "exponential", "censored_mle", "1", _exponential_settings_sha256()
        ),
        rng=RngProvenance(RngPolicy.NO_RANDOMNESS, "none", None),
        approximation=ExactComputation("closed_form_mle"),
        checkpoint=CheckpointNotUsed(),
    )


def _exponential_settings_sha256() -> str:
    """Hash the complete canonical settings contract rather than a placeholder."""

    settings = {
        "censoring_assumption": "independent_right_censoring",
        "family": "exponential",
        "location": 0.0,
        "parameterization": "rate",
        "reduction": "neumaier_canonical_input_order",
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["ExponentialSourceFitResult", "fit_exponential_csv", "fit_exponential_source"]
