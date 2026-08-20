"""DS-07 scale guards for lazy pass-budget enforcement."""

from __future__ import annotations

import unittest

from veridist.engine.pass_budget import PassEnforcer

from tests.contract.test_pass_budget import CountingSource


class PassBudgetScaleTests(unittest.TestCase):
    """DS-07: enforcing a pass never materializes or advances the source."""

    def test_ds07_begin_pass_is_lazy_and_preserves_streaming_iteration(self) -> None:
        source = CountingSource(*range(10_000))
        enforcer = PassEnforcer(max_passes=1)

        stream = enforcer.begin_pass(source)

        self.assertIs(stream, source.iterators[0])
        self.assertEqual(source.read_count, 1)
        self.assertEqual(stream.next_count, 0)
        self.assertEqual(next(stream), 0)
        self.assertEqual(stream.next_count, 1)
        self.assertEqual(enforcer.actual_pass_count, 1)


if __name__ == "__main__":
    unittest.main()
