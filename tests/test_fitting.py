import numpy as np

from distfit_pro import get_distribution


def test_mle_recovers_normal_params():
    rng = np.random.default_rng(123)
    true_mu = 2.5
    true_sigma = 1.3
    data = rng.normal(loc=true_mu, scale=true_sigma, size=5000)

    dist = get_distribution("normal")
    dist.fit(data, method="mle")

    est = dist.params
    assert abs(est["mu"] - true_mu) < 0.1
    assert abs(est["sigma"] - true_sigma) < 0.1


def test_moments_not_exploding_on_simple_data():
    rng = np.random.default_rng(456)
    data = rng.exponential(scale=2.0, size=2000)

    dist = get_distribution("exponential")
    dist.fit(data, method="moments")

    est = dist.params
    assert est["scale"] > 0
