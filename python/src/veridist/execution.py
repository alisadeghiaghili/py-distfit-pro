"""One-pass source-to-family execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
)
from veridist.engine.data_source import ExecutionPlan, SpoolPolicy, plan_passes
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
from veridist.families.exponential import ExponentialFit, fit_exponential_reduction_state
from veridist.statistics.exponential import ExponentialReductionState


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
    plan = plan_passes(adapter, required_passes=1, spool=SpoolPolicy.disabled())
    passes = PassEnforcer(max_passes=1)
    buffer = BoundedChunkBuffer(
        chunk_bytes=adapter.limits.chunk_bytes,
        max_inflight_bytes=adapter.limits.max_inflight_bytes,
    )
    validator = DeliveryValidator(adapter.source_id.value)
    state = ExponentialReductionState.empty()
    stage = FailureStage.PREFLIGHT
    error: EngineContractError | None = None
    try:
        for chunk in passes.begin_pass(adapter.iter_chunks()):
            stage = FailureStage.DELIVERY
            validator.accept(chunk.envelope)
            lease = BufferedChunk(envelope=chunk.envelope, payload=chunk.observations)
            buffer.put(lease)
            received = buffer.get()
            try:
                payload = received.payload
                if type(payload) is not tuple:
                    raise TypeError("CSV chunk payload must be a tuple")
                for observation in payload:
                    state = state.add(observation)
            finally:
                received.release()
        stage = FailureStage.FINALIZATION
        validator.finish(
            expected_row_stop=validator.next_offset,
            expected_chunk_count=validator.next_sequence,
        )
    except EngineContractError as captured:
        error = captured

    if error is None:
        extent = KnownExtent(0, validator.next_offset)
        coverage = KnownCoverage(
            extent,
            (RowRange(0, validator.next_offset),) if validator.next_offset else (),
            validator.accepted_chunks,
            0,
        )
        outcome = classify_execution_outcome(coverage, None)
        fit = fit_exponential_reduction_state(state)
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
        run_id="run_00000000000000000000000000000000",
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
            "exponential", "censored_mle", "1", "0" * 64
        ),
        rng=RngProvenance(RngPolicy.NO_RANDOMNESS, "none", None),
        approximation=ExactComputation("closed_form_mle"),
        checkpoint=CheckpointNotUsed(),
    )


__all__ = ["ExponentialSourceFitResult", "fit_exponential_csv", "fit_exponential_source"]
