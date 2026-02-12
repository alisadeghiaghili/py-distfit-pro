Weighted Data
=============

Fit distributions to weighted observations.

Basic Usage
-----------

.. code-block:: python

    from distfit_pro.core.weighted import WeightedFitting
    from distfit_pro import get_distribution
    import numpy as np
    
    data = np.random.normal(5, 2, 1000)
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    dist.params = params
    dist.fitted = True

Weighted Statistics
-------------------

.. code-block:: python

    stats = WeightedFitting.weighted_stats(data, weights)
    print(f"Weighted mean: {stats['mean']:.2f}")
    print(f"Weighted std: {stats['std']:.2f}")

Effective Sample Size
---------------------

.. code-block:: python

    ess = WeightedFitting.effective_sample_size(weights)
    print(f"ESS: {ess:.1f} (out of {len(weights)})")
