import numpy as np

from distfit_pro import get_distribution
from distfit_pro.core.diagnostics import Diagnostics


def test_outlier_detection_basic_behaviour():
    rng = np.random.default_rng(2024)
    data = np.concatenate([
        rng.normal(0.0, 1.0, size=500),
        np.array([8.0, 9.0, -7.0]),  # obvious outliers
    ])

    dist = get_distribution("normal")
    dist.fit(data, method="mle")

    outliers = Diagnostics.detect_outliers(data, dist, method="zscore", threshold=3.0)

    # Expect to catch at least some of the injected outliers
    assert len(outliers.outlier_indices) >= 2
