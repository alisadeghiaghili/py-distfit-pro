"""The v1 fitting-engine public surface."""

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

__all__ = [
    "AdapterCapabilities",
    "AdapterKind",
    "BoundedChunkBuffer",
    "BufferedChunk",
    "CheckpointMetadata",
    "ChunkEnvelope",
    "DataSourceCapabilityError",
    "DataSourceMetadata",
    "DeliveryContractError",
    "DeliveryValidator",
    "OrderingGuarantee",
    "Replayability",
    "SpoolPolicy",
    "plan_passes",
]
