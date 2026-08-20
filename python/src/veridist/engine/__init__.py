"""The v1 fitting-engine public surface."""

from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    CheckpointStore,
    InMemoryCheckpointStore,
)
from veridist.engine.data_source import (
    CheckpointMetadata,
    DataSourceCapabilityError,
    DataSourceMetadata,
    Replayability,
    SpoolPolicy,
    plan_passes,
)
from veridist.engine.delivery import (
    AdapterCapabilities,
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
    DeliveryValidator,
    OrderingGuarantee,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.pass_budget import PassBudgetError, PassEnforcer
from veridist.engine.retry import (
    EffectStatus,
    IdempotentSink,
    PureReducer,
    SinkResult,
    apply_pure_update,
    apply_sink_update,
)

__all__ = [
    "AdapterCapabilities",
    "AdapterKind",
    "BoundedChunkBuffer",
    "BufferedChunk",
    "CheckpointMetadata",
    "CheckpointCommitUncertain",
    "CheckpointRecord",
    "CheckpointStore",
    "ChunkEnvelope",
    "DataSourceCapabilityError",
    "DataSourceMetadata",
    "DeliveryContractError",
    "DeliveryValidator",
    "EngineContractError",
    "EffectStatus",
    "FailureCode",
    "IdempotentSink",
    "InMemoryCheckpointStore",
    "OrderingGuarantee",
    "PassBudgetError",
    "PassEnforcer",
    "PureReducer",
    "Replayability",
    "SpoolPolicy",
    "SinkResult",
    "apply_pure_update",
    "apply_sink_update",
    "plan_passes",
]
