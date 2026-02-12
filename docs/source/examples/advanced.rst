Advanced Examples
=================

Complex workflows and use cases.

Example 1: Automated Model Selection
-------------------------------------

.. code-block:: python

    from distfit_pro import get_distribution, list_distributions
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    data = np.random.gamma(2, 3, 1000)
    
    # Try all continuous distributions
    candidates = list_continuous_distributions()
    results = {}
    
    for name in candidates:
        dist = get_distribution(name)
        try:
            dist.fit(data)
            
            # AIC
            k = len(dist.params)
            log_lik = np.sum(dist.logpdf(data))
            aic = 2 * k - 2 * log_lik
            
            # KS test
            ks = GOFTests.kolmogorov_smirnov(data, dist)
            
            results[name] = {
                'aic': aic,
                'ks_pvalue': ks.p_value,
                'dist': dist
            }
        except:
            pass
    
    # Best by AIC
    valid = {k: v for k, v in results.items() if v['ks_pvalue'] > 0.05}
    best = min(valid.items(), key=lambda x: x[1]['aic'])
    
    print(f"Best distribution: {best[0]}")
    print(f"AIC: {best[1]['aic']:.2f}")
    print(f"KS p-value: {best[1]['ks_pvalue']:.4f}")

Example 2: Bootstrap + Diagnostics Pipeline
--------------------------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.bootstrap import Bootstrap
    from distfit_pro.core.diagnostics import Diagnostics
    
    # Fit
    data = np.random.weibull(1.5, 1000)
    dist = get_distribution('weibull')
    dist.fit(data)
    
    # Bootstrap CI
    ci = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    for param, result in ci.items():
        print(result)
    
    # Diagnostics
    residuals = Diagnostics.residual_analysis(data, dist)
    outliers = Diagnostics.detect_outliers(data, dist, method='likelihood')
    
    print(f"\nOutliers: {len(outliers.outlier_indices)}")
    
    # Remove outliers and refit
    clean_data = np.delete(data, outliers.outlier_indices)
    dist_clean = get_distribution('weibull')
    dist_clean.fit(clean_data)
    
    print("\nAfter removing outliers:")
    print(dist_clean.summary())

Example 3: Survey Data with Weights
------------------------------------

.. code-block:: python

    from distfit_pro.core.weighted import WeightedFitting
    
    # Survey responses (income in $1000s)
    income = np.array([25, 30, 35, 40, 50, 60, 80, 120, 200])
    
    # Sampling weights (inverse probability)
    weights = np.array([0.8, 1.0, 1.2, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2])
    
    # Weighted fit
    dist = get_distribution('lognormal')
    params = WeightedFitting.fit_weighted_mle(income, weights, dist)
    dist.params = params
    dist.fitted = True
    
    # Weighted statistics
    stats = WeightedFitting.weighted_stats(income, weights)
    print(f"Weighted mean: ${stats['mean']:.1f}k")
    print(f"Weighted median: ${stats['median']:.1f}k")
    
    # ESS
    ess = WeightedFitting.effective_sample_size(weights)
    print(f"ESS: {ess:.1f} (out of {len(weights)})")
