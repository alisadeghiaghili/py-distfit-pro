Quick Start
===========

This page provides a quick introduction to DistFit Pro.

Basic Usage
-----------

1. **Import and Generate Data**

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=1000)

2. **Fit a Distribution**

.. code-block:: python

    # Get distribution object
    dist = get_distribution('normal')
    
    # Fit to data
    dist.fit(data, method='mle')
    
    # View results
    print(dist.summary())

3. **Test Goodness-of-Fit**

.. code-block:: python

    from distfit_pro.core.gof_tests import GOFTests
    
    # Run all tests
    results = GOFTests.run_all_tests(data, dist)
    
    # Summary table
    print(GOFTests.summary_table(results))

4. **Bootstrap Confidence Intervals**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    # Parametric bootstrap
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    for param, result in ci_results.items():
        print(result)

5. **Diagnostics**

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    # Residual analysis
    residuals = Diagnostics.residual_analysis(data, dist)
    print(residuals.summary())
    
    # Detect outliers
    outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
    print(outliers.summary())

Compare Multiple Distributions
-------------------------------

.. code-block:: python

    from distfit_pro import list_distributions
    from distfit_pro.core.gof_tests import GOFTests
    
    # Try multiple distributions
    distributions = ['normal', 'lognormal', 'gamma', 'weibull']
    
    results = {}
    for dist_name in distributions:
        dist = get_distribution(dist_name)
        try:
            dist.fit(data)
            # Use AIC for comparison
            n = len(data)
            k = len(dist.params)
            log_lik = np.sum(dist.logpdf(data))
            aic = 2 * k - 2 * log_lik
            
            results[dist_name] = {
                'dist': dist,
                'aic': aic,
                'params': dist.params
            }
        except:
            pass
    
    # Find best fit
    best = min(results.items(), key=lambda x: x[1]['aic'])
    print(f"Best distribution: {best[0]}")
    print(f"AIC: {best[1]['aic']:.2f}")

Weighted Data
-------------

.. code-block:: python

    from distfit_pro.core.weighted import WeightedFitting
    
    # Data with weights
    data = np.random.normal(5, 2, 1000)
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    # Weighted fitting
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    dist.params = params
    dist.fitted = True
    
    print(dist.summary())

Next Steps
----------

- :doc:`tutorial/index` - Detailed tutorials
- :doc:`user_guide/distributions` - Available distributions
- :doc:`examples/index` - Real-world examples
- :doc:`api/index` - Complete API reference
