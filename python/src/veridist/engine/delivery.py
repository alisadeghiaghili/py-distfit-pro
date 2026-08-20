"""Stdlib-only delivery contracts for chunk identity and bounded buffering."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, Lock
from types import MappingProxyType

from veridist.engine.data_source import Replayability


class AdapterKind(StrEnum):
    """Adapter categories whose declarations require no third-party import."""

    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"
    PANDAS = "pandas"
    POLARS = "polars"
    DASK = "dask"
    DATABASE = "database"


class OrderingGuarantee(StrEnum):
    """Ordering semantics declared by an adapter capability record."""

    STABLE_ROW_OFFSETS = "stable_row_offsets"


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Dependency-free declaration, not evidence that an adapter exists."""

    kind: AdapterKind
    replayability: Replayability
    ordering: OrderingGuarantee
    stable_offsets: bool


@dataclass(frozen=True, slots=True)
class ChunkEnvelope:
    """Stable source/offset identity and byte accounting for one chunk."""

    source_id: str
    chunk_id: str
    row_start: int
    row_stop: int
    byte_size: int

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")
        if self.row_start < 0:
            raise ValueError("row_start must be non-negative")
        if self.row_stop < self.row_start:
            raise ValueError("row_stop must not precede row_start")
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    def row_identity(self, local_index: int) -> tuple[str, int]:
        """Return partition-independent identity for a row in this envelope."""

        if local_index < 0 or local_index >= self.row_count:
            raise IndexError("local row index is outside the chunk")
        return self.source_id, self.row_start + local_index


class DeliveryContractError(Exception):
    """Stable, localization-independent delivery contract failure."""

    def __init__(self, code: str, context: Mapping[str, object]) -> None:
        super().__init__(code)
        self.code = code
        self.context: Mapping[str, object] = MappingProxyType(dict(context))


class DeliveryValidator:
    """Validate contiguous non-overlapping delivery without silent repair."""

    __slots__ = (
        "_accepted_chunks",
        "_accepted_rows",
        "_next_offset",
        "_seen_chunk_ids",
        "_source_id",
    )

    def __init__(self, source_id: str, *, initial_offset: int = 0) -> None:
        if not source_id.strip():
            raise ValueError("source_id must be non-empty")
        if initial_offset < 0:
            raise ValueError("initial_offset must be non-negative")
        self._source_id = source_id
        self._next_offset = initial_offset
        self._accepted_rows = 0
        self._accepted_chunks = 0
        self._seen_chunk_ids: set[str] = set()

    @property
    def next_offset(self) -> int:
        return self._next_offset

    @property
    def accepted_rows(self) -> int:
        return self._accepted_rows

    @property
    def accepted_chunks(self) -> int:
        return self._accepted_chunks

    def accept(self, envelope: ChunkEnvelope) -> None:
        """Accept exactly the next source range or fail without state mutation."""

        context = {
            "chunk_id": envelope.chunk_id,
            "expected_offset": self._next_offset,
            "row_start": envelope.row_start,
            "row_stop": envelope.row_stop,
        }
        if envelope.source_id != self._source_id:
            raise DeliveryContractError(
                "SOURCE_MISMATCH",
                {**context, "expected_source_id": self._source_id, "source_id": envelope.source_id},
            )
        if envelope.chunk_id in self._seen_chunk_ids:
            raise DeliveryContractError("DUPLICATE_CHUNK", context)
        if envelope.row_start > self._next_offset:
            raise DeliveryContractError("MISSING_CHUNK", context)
        if envelope.row_start < self._next_offset:
            raise DeliveryContractError("OUT_OF_ORDER_CHUNK", context)

        self._seen_chunk_ids.add(envelope.chunk_id)
        self._next_offset = envelope.row_stop
        self._accepted_rows += envelope.row_count
        self._accepted_chunks += 1

    def finish(self, *, expected_row_stop: int) -> None:
        """Detect a missing terminal range that no subsequent chunk can reveal."""

        if expected_row_stop < 0:
            raise ValueError("expected_row_stop must be non-negative")
        if self._next_offset < expected_row_stop:
            raise DeliveryContractError(
                "MISSING_CHUNK",
                {
                    "expected_offset": self._next_offset,
                    "expected_row_stop": expected_row_stop,
                },
            )
        if self._next_offset > expected_row_stop:
            raise DeliveryContractError(
                "OUT_OF_ORDER_CHUNK",
                {
                    "expected_offset": expected_row_stop,
                    "row_stop": self._next_offset,
                },
            )


class BufferedChunk:
    """A payload lease whose release callback is safe to call more than once."""

    __slots__ = ("_release_callback", "_release_lock", "_released", "envelope", "payload")

    def __init__(
        self,
        *,
        envelope: ChunkEnvelope,
        payload: object,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        self.envelope = envelope
        self.payload = payload
        self._release_callback = release_callback
        self._release_lock = Lock()
        self._released = False

    @property
    def released(self) -> bool:
        with self._release_lock:
            return self._released

    def release(self) -> None:
        callback: Callable[[], None] | None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            callback = self._release_callback
        if callback is not None:
            callback()


class BoundedChunkBuffer:
    """Condition-backed FIFO with hard byte bounds and explicit cancellation."""

    def __init__(self, *, chunk_bytes: int, max_inflight_bytes: int) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if max_inflight_bytes < chunk_bytes:
            raise ValueError("max_inflight_bytes must be at least chunk_bytes")
        self.chunk_bytes = chunk_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self._condition = Condition()
        self._queue: deque[BufferedChunk] = deque()
        self._inflight_bytes = 0
        self._peak_inflight_bytes = 0
        self._waiting_producers = 0
        self._cancelled = False

    @property
    def inflight_bytes(self) -> int:
        with self._condition:
            return self._inflight_bytes

    @property
    def peak_inflight_bytes(self) -> int:
        with self._condition:
            return self._peak_inflight_bytes

    @property
    def queued_chunks(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def waiting_producers(self) -> int:
        with self._condition:
            return self._waiting_producers

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    def _raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise DeliveryContractError("CANCELLED", {})

    def put(self, item: BufferedChunk, *, timeout: float | None = None) -> None:
        """Queue an item, blocking while its bytes would exceed the hard bound."""

        byte_size = item.envelope.byte_size
        if byte_size > self.chunk_bytes:
            raise DeliveryContractError(
                "CHUNK_TOO_LARGE",
                {"byte_size": byte_size, "chunk_bytes": self.chunk_bytes},
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._raise_if_cancelled()
            while self._inflight_bytes + byte_size > self.max_inflight_bytes:
                self._waiting_producers += 1
                try:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("timed out waiting for buffer capacity")
                    self._condition.wait(remaining)
                finally:
                    self._waiting_producers -= 1
                self._raise_if_cancelled()
            self._queue.append(item)
            self._inflight_bytes += byte_size
            self._peak_inflight_bytes = max(self._peak_inflight_bytes, self._inflight_bytes)
            self._condition.notify_all()

    def read_and_put(
        self,
        read_next: Callable[[], BufferedChunk],
        *,
        timeout: float | None = None,
    ) -> None:
        """Read only while active, releasing a read item rejected by the buffer."""

        with self._condition:
            self._raise_if_cancelled()
            item = read_next()
        try:
            self.put(item, timeout=timeout)
        except BaseException:
            item.release()
            raise

    def get(self, *, timeout: float | None = None) -> BufferedChunk:
        """Return the oldest item and release its bytes from the buffer budget."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._queue:
                self._raise_if_cancelled()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for a buffered chunk")
                self._condition.wait(remaining)
            item = self._queue.popleft()
            self._inflight_bytes -= item.envelope.byte_size
            self._condition.notify_all()
            return item

    def cancel(self) -> None:
        """Wake waiters and release every resource still owned by the queue."""

        with self._condition:
            if self._cancelled:
                return
            self._cancelled = True
            queued = tuple(self._queue)
            self._queue.clear()
            self._inflight_bytes = 0
            self._condition.notify_all()
        for item in queued:
            item.release()


__all__ = [
    "AdapterCapabilities",
    "AdapterKind",
    "BoundedChunkBuffer",
    "BufferedChunk",
    "ChunkEnvelope",
    "DeliveryContractError",
    "DeliveryValidator",
    "OrderingGuarantee",
]
