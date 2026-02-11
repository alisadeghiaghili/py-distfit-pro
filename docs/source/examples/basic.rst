Basic Examples
==============

Simple examples to get started.

Example 1: Fitting Normal Distribution
---------------------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=500)
    
    # Fit
    dist = get_distribution('normal')
    dist.fit(data, method='mle')
    
    # Results
    print(dist.summary())
    print(f"\nMean: {dist.mean():.2f}")
    print(f"Std: {dist.std():.2f}")
    
    # Generate new samples
    new_samples = dist.rvs(size=10)
    print(f"\nNew samples: {new_samples}")

Example 2: Compare Multiple Distributions
------------------------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    # Generate gamma data
    data = np.random.gamma(2, 3, 1000)
    
    # Try different distributions
    candidates = ['gamma', 'lognormal', 'weibull', 'exponential']
    
    results = {}
    for name in candidates:
        dist = get_distribution(name)
        try:
            dist.fit(data)
            ks_result = GOFTests.kolmogorov_smirnov(data, dist)
            
            results[name] = {
                'dist': dist,
                'p_value': ks_result.p_value,
                'aic': 2 * len(dist.params) - 2 * np.sum(dist.logpdf(data))
            }
        except:
            pass
    
    # Print results
    print("Distribution Comparison:\n")
    print(f"{'Distribution':<15} {'P-value':<12} {'AIC':<12}")
    print("-" * 40)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['aic']):
        print(f"{name:<15} {r['p_value']:<12.6f} {r['aic']:<12.2f}")

Example 3: Bootstrap Confidence Intervals
------------------------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.bootstrap import Bootstrap
    import numpy as np
    
    # Small sample (where bootstrap helps!)
    data = np.random.normal(50, 10, 30)
    
    # Fit
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Bootstrap CI
    ci_results = Bootstrap.nonparametric(
        data=data,
        distribution=dist,
        n_bootstrap=2000,
        confidence_level=0.95,
        n_jobs=-1
    )
    
    # Report
    print("Parameter Estimates with 95% CI:\n")
    for param, result in ci_results.items():
        print(f"{param}:")
        print(f"  Point estimate: {result.estimate:.4f}")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"  Width: {result.ci_upper - result.ci_lower:.4f}\n")

Example 4: Outlier Detection
-----------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.diagnostics import Diagnostics
    import numpy as np
    
    # Data with outliers
    np.random.seed(42)
    clean_data = np.random.normal(100, 10, 95)
    outliers = np.array([150, 155, 160, 45, 40])  # 5 outliers
    data = np.concatenate([clean_data, outliers])
    
    # Fit
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Detect outliers
    outliers_detected = Diagnostics.detect_outliers(
        data=data,
        distribution=dist,
        method='zscore',
        threshold=3
    )
    
    print(f"Outliers detected: {len(outliers_detected.outlier_indices)}")
    print(f"Indices: {outliers_detected.outlier_indices}")
    print(f"Values: {data[outliers_detected.outlier_indices]}")

Example 5: Weighted Data
------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.weighted import WeightedFitting
    import numpy as np
    
    # Survey data with sampling weights
    # Urban (70% of pop): 100 samples
    # Rural (30% of pop): 100 samples
    
    data_urban = np.random.normal(75, 12, 100)
    data_rural = np.random.normal(65, 15, 100)
    
    data = np.concatenate([data_urban, data_rural])
    weights = np.concatenate([
        np.ones(100) * 0.7,  # Urban weight
        np.ones(100) * 0.3   # Rural weight
    ])
    
    # Weighted fit
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    dist.params = params
    dist.fitted = True
    
    # Compare
    dist_unweighted = get_distribution('normal')
    dist_unweighted.fit(data)
    
    print("Weighted vs Unweighted:\n")
    print(f"Weighted mean:   {params['loc']:.2f}")
    print(f"Unweighted mean: {dist_unweighted.params['loc']:.2f}")
    print(f"Difference:      {abs(params['loc'] - dist_unweighted.params['loc']):.2f}")
