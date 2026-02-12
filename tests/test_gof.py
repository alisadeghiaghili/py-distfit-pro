import numpy as np

from distfit_pro import get_distribution
from distfit_pro.core.gof_tests import GOFTests


def test_gof_on_correct_model_does_not_always_reject():
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, size=1000)

    dist = get_distribution("normal")
    dist.fit(data, method="mle")

    results = GOFTests.run_all_tests(data, dist)

    # All tests should return finite statistics and p-values in [0,1]
    for res in results.values():
        assert np.isfinite(res.statistic)
        assert 0.0 <= res.p_value <= 1.0
