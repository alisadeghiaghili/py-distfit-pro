import numpy as np

from distfit_pro import get_distribution
from distfit_pro.core.weighted import WeightedFitting


def test_weighted_mle_matches_expanded_sample():
    rng = np.random.default_rng(555)
    values = np.array([0.0, 1.0, 2.0])
    freqs = np.array([10, 20, 30])

    expanded = np.repeat(values, freqs)

    dist1 = get_distribution("normal")
    dist1.fit(expanded, method="mle")

    dist2 = get_distribution("normal")
    params = WeightedFitting.fit_weighted_mle(values, freqs, dist2)
    dist2.params = params
    dist2.fitted = True

    assert abs(dist1.mean() - dist2.mean()) < 1e-6
