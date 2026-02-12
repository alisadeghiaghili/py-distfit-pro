import numpy as np
import pytest

from distfit_pro import get_distribution


@pytest.mark.parametrize("name, true_mean, true_var", [
    ("normal", 0.0, 1.0),
    ("lognormal", None, None),
    ("gamma", 2.0, 2.0),
    ("weibull", 1.5, None),
])
def test_basic_distribution_properties(name, true_mean, true_var):
    rng = np.random.default_rng(42)

    if name == "normal":
        data = rng.normal(loc=true_mean, scale=1.0, size=10000)
    elif name == "lognormal":
        data = rng.lognormal(mean=1.0, sigma=0.5, size=10000)
    elif name == "gamma":
        data = rng.gamma(shape=true_mean, scale=1.0, size=10000)
    elif name == "weibull":
        data = rng.weibull(a=1.5, size=10000)
    else:
        pytest.skip("Unsupported test distribution")

    dist = get_distribution(name)
    dist.fit(data, method="mle")

    # CDF should be non-decreasing and within [0,1]
    xs = np.linspace(np.min(data), np.max(data), 200)
    cdf_vals = dist.cdf(xs)
    assert np.all(np.diff(cdf_vals) >= -1e-8)
    assert np.all((cdf_vals >= -1e-8) & (cdf_vals <= 1 + 1e-8))

    # Mean / var sanity checks where we know the target
    if true_mean is not None:
        assert np.isfinite(dist.mean())

    if true_var is not None:
        assert dist.variance() > 0
