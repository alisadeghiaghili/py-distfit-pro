Tutorial 5: Bootstrap Confidence Intervals
===========================================

Learn to estimate parameter uncertainty with bootstrap.

Why Bootstrap?
--------------

**Parameter estimates have uncertainty!**

Bootstrap helps you answer:

- How confident are we in the fitted parameters?
- What's the range of plausible values?
- How much would parameters change with different samples?

Parametric Bootstrap
--------------------

**Resample from the fitted distribution.**

Process:

1. Fit distribution to original data
2. Generate new sample from fitted distribution
3. Refit to new sample
4. Repeat 1000+ times
5. Calculate confidence intervals from distribution of estimates

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.bootstrap import Bootstrap
    import numpy as np
    
    # Original data
    np.random.seed(42)
    data = np.random.normal(10, 2, 500)
    
    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    print("Original estimates:")
    print(f"μ = {dist.params['loc']:.4f}")
    print(f"σ = {dist.params['scale']:.4f}")
    
    # Parametric bootstrap
    ci_results = Bootstrap.parametric(
        data, 
        dist, 
        n_bootstrap=1000,
        confidence_level=0.95,
        n_jobs=-1,  # Use all cores
        random_state=42
    )
    
    # View results
    for param, result in ci_results.items():
        print(result)

**Output:**

::

    Bootstrap CI for loc
    ==================================================
    Point Estimate: 10.0173
    95% CI: [9.8421, 10.1925]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000
    
    Bootstrap CI for scale
    ==================================================
    Point Estimate: 1.9918
    95% CI: [1.8567, 2.1289]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000

**Interpretation:**

- We're 95% confident μ is between 9.84 and 10.19
- True value (10.0) is inside the interval ✅

Non-Parametric Bootstrap
------------------------

**Resample from the original data.**

More robust - doesn't assume fitted distribution is correct.

.. code-block:: python

    # Non-parametric bootstrap
    ci_results_np = Bootstrap.nonparametric(
        data,
        dist,
        n_bootstrap=1000,
        confidence_level=0.95,
        n_jobs=-1,
        random_state=42
    )
    
    for param, result in ci_results_np.items():
        print(result)

**When to use:**

- Unsure if distribution is correct
- Want robust estimates
- Distribution has heavy tails

Comparing Bootstrap Methods
----------------------------

.. code-block:: python

    import pandas as pd
    
    # Compare both methods
    methods = {
        'Parametric': Bootstrap.parametric(data, dist, n_bootstrap=1000),
        'Non-parametric': Bootstrap.nonparametric(data, dist, n_bootstrap=1000)
    }
    
    # Create comparison table
    comparison = []
    for method_name, results in methods.items():
        for param_name, result in results.items():
            comparison.append({
                'Method': method_name,
                'Parameter': param_name,
                'Estimate': result.estimate,
                'CI Lower': result.ci_lower,
                'CI Upper': result.ci_upper,
                'Width': result.ci_upper - result.ci_lower
            })
    
    df = pd.DataFrame(comparison)
    print(df)

**Output:**

::

    Method           Parameter  Estimate  CI Lower  CI Upper  Width
    Parametric       loc        10.0173   9.8421    10.1925   0.3504
    Parametric       scale      1.9918    1.8567    2.1289    0.2722
    Non-parametric   loc        10.0173   9.8356    10.1989   0.3633
    Non-parametric   scale      1.9918    1.8423    2.1412    0.2989

Changing Confidence Level
-------------------------

.. code-block:: python

    # 90% CI (narrower)
    ci_90 = Bootstrap.parametric(data, dist, confidence_level=0.90)
    
    # 99% CI (wider)
    ci_99 = Bootstrap.parametric(data, dist, confidence_level=0.99)
    
    print("Comparison of confidence levels:")
    param = 'loc'
    print(f"90% CI: [{ci_90[param].ci_lower:.4f}, {ci_90[param].ci_upper:.4f}]")
    print(f"95% CI: [{ci_results[param].ci_lower:.4f}, {ci_results[param].ci_upper:.4f}]")
    print(f"99% CI: [{ci_99[param].ci_lower:.4f}, {ci_99[param].ci_upper:.4f}]")

Advanced: BCa Bootstrap
-----------------------

**Bias-Corrected and Accelerated (BCa)**

More accurate than percentile method.

.. code-block:: python

    # Generate bootstrap samples manually
    bootstrap_estimates = []
    
    for i in range(1000):
        # Resample
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        
        # Fit
        boot_dist = get_distribution('normal')
        boot_dist.fit(boot_sample)
        
        bootstrap_estimates.append(boot_dist.params['loc'])
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # BCa confidence interval
    def estimate_mean(sample):
        return np.mean(sample)
    
    ci_lower, ci_upper = Bootstrap.bca_ci(
        bootstrap_estimates,
        original_estimate=dist.params['loc'],
        data=data,
        estimator_func=estimate_mean,
        confidence_level=0.95
    )
    
    print(f"BCa 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

Parallel Processing
-------------------

**Speed up bootstrap with parallel execution:**

.. code-block:: python

    import time
    
    # Single core (slow)
    start = time.time()
    ci_single = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=1)
    time_single = time.time() - start
    
    # All cores (fast)
    start = time.time()
    ci_parallel = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=-1)
    time_parallel = time.time() - start
    
    print(f"Single core: {time_single:.2f} seconds")
    print(f"Parallel: {time_parallel:.2f} seconds")
    print(f"Speedup: {time_single/time_parallel:.1f}x")

**Typical output:**

::

    Single core: 12.34 seconds
    Parallel: 2.15 seconds
    Speedup: 5.7x

Real Example: Weibull Reliability
----------------------------------

**Estimate confidence intervals for reliability analysis:**

.. code-block:: python

    # Component lifetime data (hours)
    lifetimes = np.random.weibull(2, 500) * 1000
    
    # Fit Weibull
    dist = get_distribution('weibull')
    dist.fit(lifetimes)
    
    print("Weibull parameters:")
    print(f"Shape (k): {dist.params['c']:.4f}")
    print(f"Scale (λ): {dist.params['scale']:.4f}")
    
    # Bootstrap CIs
    ci_results = Bootstrap.parametric(
        lifetimes,
        dist,
        n_bootstrap=2000,
        confidence_level=0.95
    )
    
    print("\n95% Confidence Intervals:")
    for param, result in ci_results.items():
        print(f"{param}: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    
    # Reliability at t=500 hours with uncertainty
    reliabilities = []
    for i in range(1000):
        boot_sample = dist.rvs(size=len(lifetimes))
        boot_dist = get_distribution('weibull')
        boot_dist.fit(boot_sample)
        reliabilities.append(boot_dist.reliability(500))
    
    reliabilities = np.array(reliabilities)
    rel_mean = np.mean(reliabilities)
    rel_ci = np.percentile(reliabilities, [2.5, 97.5])
    
    print(f"\nReliability at 500h: {rel_mean:.4f}")
    print(f"95% CI: [{rel_ci[0]:.4f}, {rel_ci[1]:.4f}]")

Bootstrap Diagnostics
---------------------

**Check if bootstrap is working correctly:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Run bootstrap
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    # Collect bootstrap estimates (manual for plotting)
    boot_estimates_loc = []
    for i in range(1000):
        boot_sample = dist.rvs(size=len(data))
        boot_dist = get_distribution('normal')
        boot_dist.fit(boot_sample)
        boot_estimates_loc.append(boot_dist.params['loc'])
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(boot_estimates_loc, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(dist.params['loc'], color='red', linestyle='--', label='Original')
    plt.axvline(ci_results['loc'].ci_lower, color='green', linestyle='--', label='95% CI')
    plt.axvline(ci_results['loc'].ci_upper, color='green', linestyle='--')
    plt.xlabel('μ estimate')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution of μ')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(boot_estimates_loc, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Bootstrap Estimates')
    
    plt.tight_layout()
    plt.savefig('bootstrap_diagnostics.png', dpi=150)
    plt.show()

Best Practices
--------------

1. **Use enough bootstrap samples**
   
   - Minimum: 1000
   - Recommended: 2000-5000
   - For publication: 10,000

2. **Set random seed for reproducibility**
   
   .. code-block:: python
   
       ci = Bootstrap.parametric(data, dist, random_state=42)

3. **Use parallel processing**
   
   .. code-block:: python
   
       ci = Bootstrap.parametric(data, dist, n_jobs=-1)

4. **Choose appropriate method**
   
   - **Parametric**: When you trust the distribution
   - **Non-parametric**: When unsure or robust needed

5. **Check if bootstrap failed**
   
   .. code-block:: python
   
       ci = Bootstrap.parametric(data, dist, n_bootstrap=1000)
       
       # Check number of successful fits
       for param, result in ci.items():
           if result.n_bootstrap < 900:  # Less than 90% success
               print(f"Warning: Many bootstrap fits failed for {param}")

Common Issues
-------------

**Issue 1: Very wide confidence intervals**

- Small sample size
- High parameter uncertainty
- Solution: Collect more data

**Issue 2: Bootstrap fails frequently**

- Poor distribution choice
- Outliers in data
- Solution: Try different distribution or remove outliers

**Issue 3: Asymmetric intervals**

- This is normal for non-normal parameters!
- Use BCa method for better accuracy

Next Steps
----------

- :doc:`06_diagnostics` - Residuals and influence
- :doc:`07_weighted_data` - Weighted bootstrap
