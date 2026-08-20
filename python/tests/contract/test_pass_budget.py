"""DS-07 contracts for source pass-budget enforcement."""

from __future__ import annotations

import unittest
from collections.abc import Iterator

from veridist.engine.pass_budget import PassBudgetError, PassEnforcer, PassObservation


class InstrumentedIterator(Iterator[int]):
    """A closable stream with observable advancement and injected failure."""

    def __init__(self, values: tuple[int, ...], *, fail_after: int | None = None) -> None:
        self._values = iter(values)
        self._fail_after = fail_after
        self.next_count = 0
        self.closed = False

    def __next__(self) -> int:
        if self._fail_after == self.next_count:
            raise RuntimeError("injected source failure")
        value = next(self._values)
        self.next_count += 1
        return value

    def close(self) -> None:
        self.closed = True


class CountingSource:
    """A source whose iterator acquisitions are externally observable."""

    def __init__(self, *values: int, fail_after: int | None = None) -> None:
        self._values = values
        self._fail_after = fail_after
        self.read_count = 0
        self.iterators: list[InstrumentedIterator] = []

    def __iter__(self) -> InstrumentedIterator:
        self.read_count += 1
        iterator = InstrumentedIterator(self._values, fail_after=self._fail_after)
        self.iterators.append(iterator)
        return iterator

    def __len__(self) -> int:
        raise AssertionError("streaming source must not be sized")

    def __getitem__(self, index: int) -> int:
        raise AssertionError(f"streaming source must not be indexed: {index}")


class PassBudgetContractTests(unittest.TestCase):
    """DS-07: pass attempts are bounded before source iteration begins."""

    def test_ds07_max_passes_is_positive_and_publicly_immutable(self) -> None:
        for invalid in (0, -1, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PassEnforcer(max_passes=invalid)

        enforcer = PassEnforcer(max_passes=2)
        self.assertEqual(enforcer.max_passes, 2)
        self.assertFalse(hasattr(enforcer, "__dict__"))
        with self.assertRaises((AttributeError, TypeError)):
            enforcer.max_passes = 3  # type: ignore[misc]

    def test_ds07_excess_attempt_fails_before_source_read_with_stable_code(self) -> None:
        source = CountingSource(1, 2)
        enforcer = PassEnforcer(max_passes=1)

        self.assertEqual(list(enforcer.begin_pass(source)), [1, 2])
        self.assertEqual(source.read_count, 1)

        with self.assertRaises(PassBudgetError) as caught:
            enforcer.begin_pass(source)

        self.assertEqual(caught.exception.code, "PASS_BUDGET_EXCEEDED")
        self.assertEqual(caught.exception.context["max_passes"], 1)
        self.assertEqual(caught.exception.context["actual_pass_count"], 1)
        self.assertEqual(caught.exception.context["attempted_pass"], 2)
        self.assertEqual(source.read_count, 1)

    def test_ds07_provenance_records_actual_pass_count_and_is_immutable(self) -> None:
        source = CountingSource(1)
        enforcer = PassEnforcer(max_passes=3)

        self.assertEqual(enforcer.provenance["actual_pass_count"], 0)
        list(enforcer.begin_pass(source))
        list(enforcer.begin_pass(source))

        provenance = enforcer.provenance
        self.assertEqual(provenance["max_passes"], 3)
        self.assertEqual(provenance["actual_pass_count"], 2)
        with self.assertRaises(TypeError):
            provenance["actual_pass_count"] = 0  # type: ignore[index]

    def test_ds12_pass_enforcer_exposes_a_closed_typed_observation(self) -> None:
        enforcer = PassEnforcer(max_passes=3)
        list(enforcer.begin_pass(CountingSource(1)))

        observation = enforcer.observation

        self.assertIsInstance(observation, PassObservation)
        self.assertEqual(observation.max_passes, 3)
        self.assertEqual(observation.actual_pass_count, 1)
        self.assertFalse(hasattr(observation, "__dict__"))
        with self.assertRaises(AttributeError):
            observation.actual_pass_count = 0  # type: ignore[misc]

    def test_ds12_pass_observation_rejects_invalid_counters(self) -> None:
        invalid = (
            (True, 0),
            (0, 0),
            (1.0, 0),
            (1, True),
            (1, -1),
            (1, 0.0),
            (1, 2),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PassObservation(*values)  # type: ignore[arg-type]

    def test_ds07_failed_and_cancelled_iterations_remain_counted(self) -> None:
        failing_source = CountingSource(1, 2, fail_after=1)
        enforcer = PassEnforcer(max_passes=2)

        failing_iterator = enforcer.begin_pass(failing_source)
        self.assertEqual(next(failing_iterator), 1)
        with self.assertRaisesRegex(RuntimeError, "injected source failure"):
            next(failing_iterator)
        self.assertEqual(enforcer.actual_pass_count, 1)

        cancelled_source = CountingSource(3, 4)
        cancelled_iterator = enforcer.begin_pass(cancelled_source)
        self.assertEqual(next(cancelled_iterator), 3)
        cancelled_iterator.close()

        self.assertTrue(cancelled_source.iterators[0].closed)
        self.assertEqual(enforcer.actual_pass_count, 2)
        self.assertEqual(enforcer.provenance["actual_pass_count"], 2)


if __name__ == "__main__":
    unittest.main()
