Bootstrap Confidence Intervals
===============================

Estimate parameter uncertainty.

Basic Usage
-----------

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    import numpy as np
    from distfit_pro import get_distribution
    
    data = np.random.normal(10, 2, 1000)
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Parametric bootstrap
    ci = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    for param, result in ci.items():
        print(result)

Non-Parametric Bootstrap
------------------------

.. code-block:: python

    # Resample from data
    ci = Bootstrap.nonparametric(data, dist, n_bootstrap=1000)

Interpretation
--------------

If 95% CI is [9.8, 10.2], we are 95% confident the true mean is in this range.
