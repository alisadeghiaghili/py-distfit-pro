"""Test-only containment for deliberately blocking buffer contract probes."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Thread
from typing import TypeVar

from veridist.engine.delivery import BoundedChunkBuffer

T = TypeVar("T")


def bounded_buffer_call(
    buffer: BoundedChunkBuffer,
    operation: Callable[[], T],
    *,
    timeout: float = 0.5,
) -> T:
    """Return an operation result or fail without leaving a live test worker.

    Mutants that break a condition wakeup must be reported as an assertion
    failure, not as a mutation-runner timeout during interpreter shutdown.
    The daemon worker is deliberately isolated to this test buffer; normal
    operations always complete before the deadline.
    """

    completed = Event()
    results: list[T] = []
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            results.append(operation())
        except BaseException as error:
            failures.append(error)
        finally:
            completed.set()

    worker = Thread(target=invoke, name="bounded-buffer-test-call", daemon=True)
    worker.start()
    if not completed.wait(timeout):
        buffer.cancel()
        worker.join(0.05)
        raise AssertionError("buffer operation did not complete before the test deadline")
    worker.join(0.05)
    if failures:
        raise failures[0]
    if not results:
        raise AssertionError("buffer operation completed without a result")
    return results[0]
