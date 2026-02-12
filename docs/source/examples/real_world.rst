Real-World Case Studies
=======================

Complete analysis examples.

Case Study 1: Manufacturing Quality Control
--------------------------------------------

**Problem:** Monitor widget diameter to detect process drift.

.. code-block:: python

    import numpy as np
    from distfit_pro import get_distribution
    from distfit_pro.core.diagnostics import Diagnostics
    
    # Weekly measurements (mm)
    week1 = np.random.normal(50.0, 0.1, 100)
    week2 = np.random.normal(50.0, 0.1, 100)
    week3 = np.random.normal(50.05, 0.15, 100)  # Drift!
    
    # Baseline (Week 1)
    baseline = get_distribution('normal')
    baseline.fit(week1)
    
    print("Baseline:")
    print(baseline.summary())
    
    # Check Week 3
    outliers_w3 = Diagnostics.detect_outliers(
        week3, baseline, method='zscore', threshold=2.0
    )
    
    defect_rate = len(outliers_w3.outlier_indices) / len(week3) * 100
    print(f"\nWeek 3 defect rate: {defect_rate:.1f}%")
    
    if defect_rate > 5:
        print("⚠️  ALERT: Process out of control!")

Case Study 2: Insurance Claims Analysis
----------------------------------------

**Problem:** Model claim amounts for pricing.

.. code-block:: python

    # Claim amounts ($)
    claims = np.array([500, 800, 1200, 1500, 2000, 3500, 5000, 12000])
    
    # Try multiple distributions
    from distfit_pro.core.gof_tests import GOFTests
    
    candidates = ['lognormal', 'gamma', 'weibull', 'pareto']
    
    for name in candidates:
        dist = get_distribution(name)
        dist.fit(claims)
        
        # GOF test
        ks = GOFTests.kolmogorov_smirnov(claims, dist)
        
        # 95th percentile (high claim)
        q95 = dist.ppf(0.95)
        
        print(f"\n{name.upper()}:")
        print(f"  KS p-value: {ks.p_value:.4f}")
        print(f"  95th %ile: ${q95:.0f}")
    
    # Best fit
    best_dist = get_distribution('lognormal')
    best_dist.fit(claims)
    
    # Expected value (for pricing)
    expected = best_dist.mean()
    print(f"\nExpected claim: ${expected:.0f}")
    
    # VaR (99%)
    var_99 = best_dist.ppf(0.99)
    print(f"VaR(99%): ${var_99:.0f}")

Case Study 3: A/B Test Analysis
--------------------------------

**Problem:** Compare conversion rates with bootstrap CI.

.. code-block:: python

    # Variant A: 1200 trials, 48 conversions
    # Variant B: 1200 trials, 60 conversions
    
    data_a = np.array([0]*1152 + [1]*48)
    data_b = np.array([0]*1140 + [1]*60)
    
    # Fit Beta distributions (conjugate prior for Bernoulli)
    dist_a = get_distribution('beta')
    dist_b = get_distribution('beta')
    
    dist_a.fit(data_a)
    dist_b.fit(data_b)
    
    print(f"Variant A: {dist_a.mean():.3f}")
    print(f"Variant B: {dist_b.mean():.3f}")
    
    # Bootstrap CI for difference
    from distfit_pro.core.bootstrap import Bootstrap
    
    ci_a = Bootstrap.parametric(data_a, dist_a, n_bootstrap=5000)
    ci_b = Bootstrap.parametric(data_b, dist_b, n_bootstrap=5000)
    
    # If CI for B doesn't overlap with A, significant!
