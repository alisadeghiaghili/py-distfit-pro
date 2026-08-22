"""DS-12 contracts for closed, redacted and deterministic execution provenance."""

from __future__ import annotations

import inspect
import json
import threading
import time
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType

from veridist import __version__
from veridist.engine.data_source import (
    DataSourceMetadata,
    ExecutionPlan,
    Replayability,
    SpoolPolicy,
    plan_passes,
)
from veridist.engine.delivery import (
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    BufferObservation,
    ChunkEnvelope,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.outcome import (
    CompleteOutcome,
    FailedOutcome,
    FailureRecord,
    FailureStage,
    KnownCoverage,
    KnownExtent,
    PartialOutcome,
    RowRange,
    UnknownMissingRanges,
)
from veridist.engine.pass_budget import PassEnforcer, PassObservation
from veridist.engine.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    AdapterProvenance,
    ApproximateComputation,
    CheckpointNotUsed,
    CheckpointStoreKind,
    CheckpointUsed,
    EstimatorProvenance,
    ExactComputation,
    ExecutionObservation,
    ExecutionProvenance,
    ExecutionReport,
    PublicSourceId,
    RngPolicy,
    RngProvenance,
    SourceHash,
    SourceHashAlgorithm,
    SourceMutationStatus,
    SourceProvenance,
    SourceRedaction,
    SourceRedactionReason,
    SpoolCleanupStatus,
    SpoolNotUsed,
    SpoolObservation,
    SpoolRetention,
    checkpoint_observation_from_resume,
    failure_record_from_error,
    snapshot_execution_observation,
    to_canonical_json_bytes,
)
from veridist.engine.resume import PublicResumeMetadata

SHA_A = "a" * 64
SHA_B = "b" * 64


def complete_outcome(*, chunks: int = 2) -> CompleteOutcome:
    return CompleteOutcome(
        KnownCoverage(KnownExtent(0, 4), (RowRange(0, 4),), chunks, 0)
    )


def execution_observation() -> ExecutionObservation:
    return ExecutionObservation(
        adapter=AdapterProvenance(AdapterKind.CSV, "1.2.3"),
        engine_version=__version__,
        replayability=Replayability.REPLAYABLE,
        required_passes=1,
        passes=PassObservation(max_passes=2, actual_pass_count=1),
        buffer=BufferObservation(
            chunk_bytes=1024,
            max_inflight_bytes=2048,
            peak_inflight_bytes=1024,
            largest_retained_chunk_bytes=512,
            backpressure_event_count=1,
        ),
        spool=SpoolNotUsed(),
    )


def provenance(
    *,
    disclosure: SourceHash | SourceRedaction | None = None,
    checkpoint: CheckpointNotUsed | CheckpointUsed | None = None,
) -> ExecutionProvenance:
    return ExecutionProvenance(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        run_id="run_0123456789abcdef0123456789abcdef",
        source=SourceProvenance(
            public_source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
            schema_version="schema-v1",
            disclosure=disclosure or SourceRedaction(SourceRedactionReason.POLICY),
            mutation_status=SourceMutationStatus.VERIFIED_UNCHANGED,
        ),
        execution=execution_observation(),
        estimator=EstimatorProvenance(
            family_id="exponential",
            estimator_id="mle",
            estimator_version="1",
            settings_sha256=SHA_A,
        ),
        rng=RngProvenance(RngPolicy.NO_RANDOMNESS, "none", None),
        approximation=ExactComputation("streaming-sum-v1"),
        checkpoint=checkpoint or CheckpointNotUsed(),
    )


def report(outcome: object | None = None, **kwargs: object) -> ExecutionReport:
    selected = complete_outcome() if outcome is None else outcome
    return ExecutionReport(
        selected,  # type: ignore[arg-type]
        provenance(**kwargs),  # type: ignore[arg-type]
    )


class ExecutionProvenanceContractTests(unittest.TestCase):
    def test_ds12_public_source_id_is_caller_supplied_and_opaque(self) -> None:
        value = PublicSourceId("src_0123456789abcdef0123456789abcdef")
        self.assertEqual(value.value, "src_0123456789abcdef0123456789abcdef")

        invalid = (
            "source",
            "src_0123456789ABCDEF0123456789ABCDEF",
            "src_0123456789abcdef0123456789abcde",
            "file:///private/source.csv",
            "C:/private/source.csv",
            "src_0123456789abcdef0123456789abcdef?token=x",
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                PublicSourceId(candidate)

    def test_ds12_source_disclosure_is_structural_hash_xor_redaction(self) -> None:
        hashed = SourceHash(SourceHashAlgorithm.SHA256, SHA_A)
        redacted = SourceRedaction(SourceRedactionReason.USER_REQUEST)

        self.assertIs(provenance(disclosure=hashed).source.disclosure, hashed)
        self.assertIs(provenance(disclosure=redacted).source.disclosure, redacted)
        for digest in ("", "A" * 64, "a" * 63, "g" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                SourceHash(SourceHashAlgorithm.SHA256, digest)
        with self.assertRaises(TypeError):
            SourceHash("sha256", SHA_A)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SourceProvenance(
                PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                "schema-v1",
                object(),  # type: ignore[arg-type]
                SourceMutationStatus.NOT_CHECKED,
            )

    def test_ds12_provenance_metadata_excludes_outcome_facts(self) -> None:
        value = provenance()
        names = {item.name for item in fields(value)}

        self.assertFalse({"coverage", "status", "complete", "failure"} & names)
        self.assertFalse(hasattr(value, "__dict__"))
        with self.assertRaises(AttributeError):
            value.run_id = "run_ffffffffffffffffffffffffffffffff"  # type: ignore[misc]

    def test_ds12_report_owns_one_outcome_and_no_duplicate_coverage(self) -> None:
        outcome = complete_outcome()
        value = ExecutionReport(outcome, provenance())

        self.assertIs(value.outcome, outcome)
        self.assertIs(value.outcome.coverage, outcome.coverage)
        self.assertFalse(hasattr(value.provenance, "coverage"))
        with self.assertRaises(TypeError):
            ExecutionReport(object(), provenance())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ExecutionReport(outcome, object())  # type: ignore[arg-type]

    def test_ds12_execution_observations_enforce_numeric_budgets(self) -> None:
        observation = execution_observation()
        self.assertEqual(observation.buffer.peak_inflight_bytes, 1024)

        invalid_buffers = (
            (0, 1, 0, 0, 0),
            (2, 1, 0, 0, 0),
            (2, 2, 3, 1, 0),
            (2, 2, 1, 3, 0),
            (2, 2, 1, 1, -1),
            (2, 2, True, 1, 0),
        )
        for values in invalid_buffers:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                BufferObservation(*values)

        with self.assertRaises(ValueError):
            ExecutionObservation(
                adapter=observation.adapter,
                engine_version=observation.engine_version,
                replayability=observation.replayability,
                required_passes=3,
                passes=PassObservation(2, 1),
                buffer=observation.buffer,
                spool=observation.spool,
            )

    def test_ds12_buffer_observation_is_runtime_owned_and_historical(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)

        def item(chunk_id: str, sequence: int, byte_size: int) -> BufferedChunk:
            return BufferedChunk(
                envelope=ChunkEnvelope(
                    "source",
                    chunk_id,
                    sequence,
                    sequence,
                    sequence + 1,
                    byte_size,
                ),
                payload=None,
            )

        first = item("first", 0, 4)
        second = item("second", 1, 4)
        buffer.put(first)
        producer = threading.Thread(target=lambda: buffer.put(second, timeout=1.0))
        producer.start()
        deadline = time.monotonic() + 1.0
        while buffer.waiting_producers == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(buffer.waiting_producers, 1)
        buffer.get().release()
        producer.join(timeout=1.0)
        self.assertFalse(producer.is_alive())
        buffer.get().release()

        observation = buffer.observation
        self.assertEqual(observation.peak_inflight_bytes, 4)
        self.assertEqual(observation.largest_retained_chunk_bytes, 4)
        self.assertEqual(observation.backpressure_event_count, 1)
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.peak_inflight_bytes, 4)
        self.assertEqual(buffer.observation, observation)

    def test_ds12_buffer_rejects_non_integer_budgets_before_runtime(self) -> None:
        invalid = (
            {"chunk_bytes": True, "max_inflight_bytes": 4},
            {"chunk_bytes": 4.0, "max_inflight_bytes": 4},
            {"chunk_bytes": 4, "max_inflight_bytes": True},
            {"chunk_bytes": 4, "max_inflight_bytes": 8.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                BoundedChunkBuffer(**kwargs)  # type: ignore[arg-type]

    def test_ds12_spool_observation_has_no_path_or_free_text(self) -> None:
        used = SpoolObservation(
            disk_budget_bytes=4096,
            retention=SpoolRetention.DELETE_ON_CLOSE,
            cleanup_status=SpoolCleanupStatus.COMPLETED,
        )
        self.assertEqual(used.disk_budget_bytes, 4096)
        self.assertFalse(hasattr(used, "path"))
        with self.assertRaises(TypeError):
            SpoolObservation(4096, "keep", SpoolCleanupStatus.COMPLETED)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SpoolObservation(0, SpoolRetention.DELETE_ON_CLOSE, SpoolCleanupStatus.COMPLETED)

    def test_ds12_snapshot_records_explicit_runtime_spool_not_plan_intent(self) -> None:
        class Source:
            metadata = DataSourceMetadata(
                source_id="internal",
                schema_version="1",
                provenance_schema_version="1",
                replayability=Replayability.SINGLE_PASS,
                source_hash=SHA_A,
            )

        plan = plan_passes(
            Source(),
            required_passes=2,
            spool=SpoolPolicy(True, 4096, "private/runtime/path", True),
        )
        enforcer = PassEnforcer(max_passes=2)
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)

        unused = snapshot_execution_observation(
            plan=plan,
            pass_enforcer=enforcer,
            buffer=buffer,
            adapter=AdapterProvenance(AdapterKind.CSV, "1"),
            spool=SpoolNotUsed(),
        )
        actual = SpoolObservation(
            4096,
            SpoolRetention.DELETE_ON_CLOSE,
            SpoolCleanupStatus.COMPLETED,
        )
        used = snapshot_execution_observation(
            plan=plan,
            pass_enforcer=enforcer,
            buffer=buffer,
            adapter=AdapterProvenance(AdapterKind.CSV, "1"),
            spool=actual,
        )

        self.assertIsInstance(unused.spool, SpoolNotUsed)
        self.assertIs(used.spool, actual)
        self.assertIs(unused.replayability, Replayability.SINGLE_PASS)
        self.assertNotIn("private/runtime/path", repr(used))

    def test_ds12_snapshot_rejects_forged_execution_plan_facts(self) -> None:
        enforcer = PassEnforcer(max_passes=2)
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        common = {
            "spool_requirements": None,
            "provenance": MappingProxyType({}),
        }
        forged = (
            ExecutionPlan(
                required_passes=True,  # type: ignore[arg-type]
                replayability=Replayability.REPLAYABLE,
                **common,
            ),
            ExecutionPlan(
                required_passes=1,
                replayability="replayable",  # type: ignore[arg-type]
                **common,
            ),
        )
        for plan in forged:
            with self.subTest(plan=plan), self.assertRaises(TypeError):
                snapshot_execution_observation(
                    plan=plan,
                    pass_enforcer=enforcer,
                    buffer=buffer,
                    adapter=AdapterProvenance(AdapterKind.CSV, "1"),
                    spool=SpoolNotUsed(),
                )

    def test_ds12_closed_schema_rejects_wrong_nested_runtime_types(self) -> None:
        source_id = PublicSourceId("src_0123456789abcdef0123456789abcdef")
        redaction = SourceRedaction(SourceRedactionReason.POLICY)
        observation = execution_observation()
        base = provenance()
        invalid = (
            lambda: PublicSourceId(1),  # type: ignore[arg-type]
            lambda: AdapterProvenance(AdapterKind.CSV, 1),  # type: ignore[arg-type]
            lambda: EstimatorProvenance("exp", "mle", "1", 1),  # type: ignore[arg-type]
            lambda: SourceRedaction("policy"),  # type: ignore[arg-type]
            lambda: SourceProvenance(
                object(),  # type: ignore[arg-type]
                "1",
                redaction,
                SourceMutationStatus.NOT_CHECKED,
            ),
            lambda: AdapterProvenance("csv", "1"),  # type: ignore[arg-type]
            lambda: SpoolObservation(
                1,
                SpoolRetention.DELETE_ON_CLOSE,
                "completed",  # type: ignore[arg-type]
            ),
            lambda: replace(observation, adapter=object()),
            lambda: replace(observation, passes=object()),
            lambda: replace(observation, buffer=object()),
            lambda: replace(observation, spool=object()),
            lambda: RngProvenance("no_randomness", "none", None),  # type: ignore[arg-type]
            lambda: CheckpointUsed(
                1,  # type: ignore[arg-type]
                "1",
                "sum-v1",
                0,
                0,
                0,
                0,
                CheckpointStoreKind.IN_MEMORY_TEST_DOUBLE,
                "1",
            ),
            lambda: replace(base, run_id=1),
            lambda: replace(base, source=object()),
            lambda: replace(base, execution=object()),
            lambda: replace(base, estimator=object()),
            lambda: replace(base, rng=object()),
            lambda: replace(base, checkpoint=object()),
        )
        self.assertEqual(source_id.value[:4], "src_")
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(TypeError):
                case()

        with self.assertRaises(ValueError):
            RngProvenance(RngPolicy.EXPLICIT_SEED, "pcg64", 2**128)

    def test_ds12_bridges_reject_untyped_internal_inputs(self) -> None:
        class Source:
            metadata = DataSourceMetadata(
                source_id="internal",
                schema_version="1",
                provenance_schema_version="1",
                replayability=Replayability.REPLAYABLE,
                source_hash=SHA_A,
            )

        plan = plan_passes(Source(), required_passes=1)
        enforcer = PassEnforcer(max_passes=1)
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        adapter = AdapterProvenance(AdapterKind.CSV, "1")
        common = {
            "plan": plan,
            "pass_enforcer": enforcer,
            "buffer": buffer,
            "adapter": adapter,
            "spool": SpoolNotUsed(),
        }
        invalid_snapshots = (
            {**common, "plan": object()},
            {**common, "pass_enforcer": object()},
            {**common, "buffer": object()},
        )
        for kwargs in invalid_snapshots:
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                snapshot_execution_observation(**kwargs)  # type: ignore[arg-type]

        resume_args = {
            "accumulator_schema_version": "sum-v1",
            "final_generation": 0,
            "retry_count": 0,
            "commit_count": 0,
            "store_version": "1",
        }
        with self.assertRaises(TypeError):
            checkpoint_observation_from_resume(object(), **resume_args)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            failure_record_from_error(object(), FailureStage.DELIVERY)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            failure_record_from_error(
                EngineContractError(FailureCode.CANCELLED),
                "read",  # type: ignore[arg-type]
            )

    def test_ds12_serializer_emits_typed_runtime_variants(self) -> None:
        base = provenance()
        metadata = replace(
            base,
            execution=replace(
                base.execution,
                spool=SpoolObservation(
                    4096,
                    SpoolRetention.DELETE_ON_SUCCESS,
                    SpoolCleanupStatus.PENDING,
                ),
            ),
            rng=RngProvenance(RngPolicy.EXPLICIT_SEED, "pcg64", 42),
            approximation=ApproximateComputation("sketch-v1", "relative-error-v1"),
        )

        payload = json.loads(
            to_canonical_json_bytes(ExecutionReport(complete_outcome(), metadata))
        )

        self.assertEqual(payload["execution"]["spool"]["kind"], "used")
        self.assertEqual(payload["rng"]["seed"], 42)
        self.assertEqual(payload["approximation"]["kind"], "approximate")

    def test_ds12_estimator_records_only_identity_version_and_settings_hash(self) -> None:
        estimator = provenance().estimator
        self.assertEqual(
            {item.name for item in fields(estimator)},
            {"family_id", "estimator_id", "estimator_version", "settings_sha256"},
        )
        self.assertFalse(hasattr(estimator, "settings"))
        with self.assertRaises(ValueError):
            EstimatorProvenance("exp", "mle", "1", "secret-settings")
        with self.assertRaises(ValueError):
            EstimatorProvenance("file:///private", "mle", "1", SHA_A)

    def test_ds12_every_serialized_string_slot_is_allowlisted(self) -> None:
        sentinel = "file:///private/source.csv?query=SELECT&credential=secret"
        base = provenance()
        checkpoint = CheckpointUsed(
            True,
            "1",
            "sum-v1",
            2,
            3,
            0,
            1,
            CheckpointStoreKind.IN_MEMORY_TEST_DOUBLE,
            "1",
        )
        cases = (
            lambda: replace(base.source, schema_version=sentinel),
            lambda: AdapterProvenance(AdapterKind.CSV, sentinel),
            lambda: replace(base.execution, engine_version=sentinel),
            lambda: replace(base.estimator, family_id=sentinel),
            lambda: replace(base.estimator, estimator_id=sentinel),
            lambda: replace(base.estimator, estimator_version=sentinel),
            lambda: RngProvenance(RngPolicy.EXTERNAL_GENERATOR, sentinel, None),
            lambda: ExactComputation(sentinel),
            lambda: ApproximateComputation(sentinel, "relative-error-v1"),
            lambda: ApproximateComputation("sketch-v1", sentinel),
            lambda: replace(checkpoint, checkpoint_schema_version=sentinel),
            lambda: replace(checkpoint, accumulator_schema_version=sentinel),
            lambda: replace(checkpoint, store_version=sentinel),
            lambda: replace(base, schema_version=sentinel),
            lambda: replace(base, run_id=sentinel),
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                case()

    def test_ds12_schema_mutation_status_and_nested_objects_are_strictly_typed_frozen(self) -> None:
        base = provenance()
        with self.assertRaises(TypeError):
            replace(base.source, mutation_status="verified")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            replace(base, schema_version="2")
        with self.assertRaises((FrozenInstanceError, TypeError)):
            base.source.schema_version = "schema-v2"  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, TypeError)):
            base.execution.buffer.peak_inflight_bytes = 0  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, TypeError)):
            base.checkpoint = CheckpointNotUsed()  # type: ignore[misc]

    def test_ds12_rng_policy_enforces_seed_and_algorithm_invariants(self) -> None:
        explicit = RngProvenance(RngPolicy.EXPLICIT_SEED, "pcg64", 42)
        external = RngProvenance(RngPolicy.EXTERNAL_GENERATOR, "pcg64", None)
        self.assertEqual(explicit.seed, 42)
        self.assertIsNone(external.seed)

        invalid = (
            (RngPolicy.NO_RANDOMNESS, "pcg64", None),
            (RngPolicy.NO_RANDOMNESS, "none", 1),
            (RngPolicy.EXPLICIT_SEED, "pcg64", None),
            (RngPolicy.EXPLICIT_SEED, "pcg64", -1),
            (RngPolicy.EXPLICIT_SEED, "pcg64", True),
            (RngPolicy.EXTERNAL_GENERATOR, "pcg64", 1),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                RngProvenance(*values)

    def test_ds12_approximation_is_an_exact_or_declared_contract_union(self) -> None:
        exact = ExactComputation("closed-form-v1")
        approximate = ApproximateComputation("sketch-v1", "relative-error-v1")
        self.assertFalse(hasattr(exact, "error_contract_id"))
        self.assertEqual(approximate.error_contract_id, "relative-error-v1")
        with self.assertRaises(ValueError):
            ApproximateComputation("sketch-v1", "")
        value = provenance()
        with self.assertRaises(TypeError):
            ExecutionProvenance(
                value.schema_version,
                value.run_id,
                value.source,
                value.execution,
                value.estimator,
                value.rng,
                object(),  # type: ignore[arg-type]
                value.checkpoint,
            )

    def test_ds12_checkpoint_union_is_honest_about_the_test_double(self) -> None:
        used = CheckpointUsed(
            resumed=True,
            checkpoint_schema_version="1",
            accumulator_schema_version="sum-v1",
            initial_generation=3,
            final_generation=5,
            retry_count=1,
            commit_count=2,
            store_kind=CheckpointStoreKind.IN_MEMORY_TEST_DOUBLE,
            store_version="1",
        )
        value = provenance(checkpoint=used)
        self.assertIs(value.checkpoint, used)
        self.assertFalse(hasattr(used, "durable"))
        with self.assertRaises(ValueError):
            CheckpointUsed(True, "1", "sum-v1", 3, 6, 1, 2, used.store_kind, "1")
        with self.assertRaises(TypeError):
            CheckpointUsed(True, "1", "sum-v1", 3, 5, 1, 2, "persistent", "1")  # type: ignore[arg-type]

    def test_ds12_serializer_derives_complete_partial_failed_and_unknown_facts(self) -> None:
        failure = FailureRecord(FailureCode.CANCELLED, FailureStage.CANCELLATION)
        outcomes = (
            complete_outcome(),
            PartialOutcome(
                KnownCoverage(KnownExtent(0, 4), (RowRange(0, 2),), 1, 0),
                failure,
            ),
            FailedOutcome(
                KnownCoverage(KnownExtent(0, 4), (RowRange(0, 4),), 1, 0),
                failure,
            ),
            FailedOutcome(UnknownMissingRanges((RowRange(0, 2),), 1, 0), failure),
        )
        expected = (
            ("complete", True, "known"),
            ("partial", False, "known"),
            ("failed", False, "known"),
            ("failed", False, "unknown_missing_ranges"),
        )
        for outcome, facts in zip(outcomes, expected, strict=True):
            with self.subTest(outcome=outcome):
                payload = json.loads(to_canonical_json_bytes(report(outcome)))
                self.assertEqual(
                    (payload["status"], payload["complete"], payload["coverage"]["kind"]),
                    facts,
                )
                if outcome.complete:
                    self.assertNotIn("failure", payload)
                else:
                    self.assertEqual(payload["failure"]["code"], "CANCELLED")

        unknown = json.loads(to_canonical_json_bytes(report(outcomes[-1])))
        self.assertEqual(unknown["coverage"]["reason"], "MISSING_RANGE_UNKNOWN")
        self.assertNotIn("missing_ranges", unknown["coverage"])

    def test_ds12_json_is_canonical_deterministic_and_fail_closed(self) -> None:
        value = report(disclosure=SourceHash(SourceHashAlgorithm.SHA256, SHA_B))
        first = to_canonical_json_bytes(value)
        second = to_canonical_json_bytes(value)

        self.assertEqual(first, second)
        self.assertEqual(first, first.decode("utf-8").encode("utf-8"))
        self.assertNotIn(b" ", first)

        def assert_sorted_keys(item: object) -> None:
            if isinstance(item, dict):
                self.assertEqual(list(item), sorted(item))
                for nested in item.values():
                    assert_sorted_keys(nested)
            elif isinstance(item, list):
                for nested in item:
                    assert_sorted_keys(nested)

        assert_sorted_keys(json.loads(first))

        class ExtendedReport(ExecutionReport):
            pass

        extended = ExtendedReport(value.outcome, value.provenance)
        # A frozen/slotted base can reject this direct assignment with a
        # CPython-version-specific setter exception.  Build the adversarial
        # subclass payload directly so this contract always exercises the
        # serializer's exact-type rejection.
        object.__setattr__(extended, "payload", "private")
        with self.assertRaises(TypeError):
            to_canonical_json_bytes(extended)
        with self.assertRaises(TypeError):
            to_canonical_json_bytes(object())  # type: ignore[arg-type]

        source = inspect.getsource(to_canonical_json_bytes)
        self.assertNotIn("asdict(", source)
        self.assertNotIn(".__dict__", source)

    def test_ds12_safe_bridges_exclude_internal_sentinels_meaningfully(self) -> None:
        sentinel = "file:///private/source.csv?query=SELECT&credential=secret"

        class Source:
            metadata = DataSourceMetadata(
                source_id=sentinel,
                schema_version=sentinel,
                provenance_schema_version="1",
                replayability=Replayability.REPLAYABLE,
                redaction_reason=sentinel,
            )

        plan = plan_passes(Source(), required_passes=1)
        self.assertIn(sentinel, repr(plan.provenance))
        enforcer = PassEnforcer(max_passes=2)
        list(enforcer.begin_pass([1]))
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(
            BufferedChunk(
                envelope=ChunkEnvelope(sentinel, sentinel, 0, 0, 1, 4),
                payload={"raw": sentinel},
            )
        )
        observation = snapshot_execution_observation(
            plan=plan,
            pass_enforcer=enforcer,
            buffer=buffer,
            adapter=AdapterProvenance(AdapterKind.CSV, "1"),
            spool=SpoolNotUsed(),
        )
        self.assertEqual(observation.required_passes, 1)
        self.assertEqual(observation.buffer.peak_inflight_bytes, 4)
        self.assertEqual(observation.buffer.largest_retained_chunk_bytes, 4)
        self.assertEqual(observation.engine_version, __version__)
        self.assertIs(observation.replayability, plan.replayability)

        resume = PublicResumeMetadata(
            format_version=1,
            source_id=sentinel,
            source_schema=sentinel,
            reducer_id=sentinel,
            accumulator_schema=sentinel,
            plan_digest=sentinel,
            cursor=1,
            committed_ranges=((0, 1),),
            generation=7,
        )
        checkpoint = checkpoint_observation_from_resume(
            resume,
            accumulator_schema_version="sum-v1",
            final_generation=8,
            retry_count=2,
            commit_count=1,
            store_version="1",
        )
        error = EngineContractError(FailureCode.CANCELLED, {"expected": sentinel})
        failure = failure_record_from_error(error, FailureStage.CANCELLATION)
        outcome = FailedOutcome(
            KnownCoverage(KnownExtent(0, 1), (RowRange(0, 1),), 1, 0),
            failure,
        )
        metadata = provenance(checkpoint=checkpoint)
        metadata = ExecutionProvenance(
            metadata.schema_version,
            metadata.run_id,
            metadata.source,
            observation,
            metadata.estimator,
            metadata.rng,
            metadata.approximation,
            metadata.checkpoint,
        )
        encoded = to_canonical_json_bytes(ExecutionReport(outcome, metadata))

        self.assertNotIn(sentinel.encode(), encoded)
        self.assertNotIn(b"private", encoded)
        self.assertNotIn(b"credential", encoded)
        self.assertNotIn(b"plan_digest", encoded)
        self.assertNotIn(b"source_revision", encoded)
        self.assertNotIn(b"payload", encoded)
        self.assertEqual(json.loads(encoded)["checkpoint"]["initial_generation"], 7)
        self.assertEqual(json.loads(encoded)["execution"]["required_passes"], 1)
        buffer.get().release()

    def test_ds12_serializer_rejects_legacy_open_mappings(self) -> None:
        plan = ExecutionPlan(
            required_passes=1,
            spool_requirements=None,
            provenance=MappingProxyType({"source_id": "private"}),
            replayability=Replayability.REPLAYABLE,
        )
        enforcer = PassEnforcer(max_passes=1)

        with self.assertRaises(TypeError):
            to_canonical_json_bytes(plan.provenance)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            to_canonical_json_bytes(enforcer.provenance)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
