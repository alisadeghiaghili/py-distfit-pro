import numpy as np

from distfit_pro import get_distribution
from distfit_pro.core.bootstrap import Bootstrap


def test_parametric_bootstrap_runs_and_produces_ci():
    rng = np.random.default_rng(99)
    data = rng.normal(loc=1.0, scale=2.0, size=500)

    dist = get_distribution("normal")
    dist.fit(data, method="mle")

    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=100, n_jobs=1)

    # Expect at least mean parameter to be present with lower/upper bounds
    assert "mu" in ci_results
    mu_ci = ci_results["mu"]
    assert mu_ci.lower < mu_ci.upper
