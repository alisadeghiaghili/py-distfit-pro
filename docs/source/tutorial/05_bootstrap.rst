Tutorial 5: Bootstrap Confidence Intervals
===========================================

Learn how to quantify uncertainty in parameter estimates.

Why Bootstrap?
--------------

Parameter estimates are uncertain. Bootstrap helps you:

- Quantify estimation uncertainty
- Get confidence intervals
- Make robust inferences
- No assumptions about sampling distribution

Parametric Bootstrap
--------------------

**Samples from the fitted distribution.**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.bootstrap import Bootstrap
    import numpy as np
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Parametric bootstrap
    ci_results = Bootstrap.parametric(
        data, 
        dist, 
        n_bootstrap=1000,
        confidence_level=0.95,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    # Print results
    for param, result in ci_results.items():
        print(result)

**Output:**

::

    Bootstrap CI for loc
    ==================================================
    Point Estimate: 10.017342
    95% CI: [9.895234, 10.139876]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000
    
    Bootstrap CI for scale
    ==================================================
    Point Estimate: 1.991847
    95% CI: [1.947823, 2.036234]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000

**How it works:**

1. Fit distribution to data → get parameters
2. Generate bootstrap samples from fitted distribution
3. Re-fit each bootstrap sample
4. Get distribution of parameter estimates
5. Calculate percentile-based confidence intervals

Non-Parametric Bootstrap
------------------------

**Resamples from the original data.**

.. code-block:: python

    # Non-parametric bootstrap
    ci_results = Bootstrap.nonparametric(
        data,
        dist,
        n_bootstrap=1000,
        confidence_level=0.95,
        n_jobs=-1,
        random_state=42
    )
    
    for param, result in ci_results.items():
        print(result)

**How it works:**

1. Resample data with replacement
2. Fit distribution to each resample
3. Get distribution of parameter estimates
4. Calculate confidence intervals

**When to use:**

- No strong distributional assumptions
- Want to be conservative
- Data may not follow the fitted distribution exactly

Parametric vs Non-Parametric
-----------------------------

.. code-block:: python

    import numpy as np
    from distfit_pro import get_distribution
    from distfit_pro.core.bootstrap import Bootstrap
    
    np.random.seed(42)
    data = np.random.gamma(2, 3, 500)
    
    dist = get_distribution('gamma')
    dist.fit(data)
    
    # Both methods
    ci_param = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    ci_nonparam = Bootstrap.nonparametric(data, dist, n_bootstrap=1000)
    
    print("Parametric Bootstrap:")
    print(ci_param['a'])
    
    print("\nNon-Parametric Bootstrap:")
    print(ci_nonparam['a'])

**Comparison:**

+-----------------+-------------------------+---------------------------+
| Feature         | Parametric              | Non-Parametric            |
+=================+=========================+===========================+
| Assumptions     | Data follows fitted dist| Minimal                   |
+-----------------+-------------------------+---------------------------+
| CI Width        | Narrower (if correct)   | Wider (more conservative) |
+-----------------+-------------------------+---------------------------+
| Speed           | Fast                    | Slower                    |
+-----------------+-------------------------+---------------------------+
| Robustness      | Less robust             | More robust               |
+-----------------+-------------------------+---------------------------+
| Best for        | Good fit confirmed      | Uncertain about model     |
+-----------------+-------------------------+---------------------------+

Choosing Sample Size
--------------------

**How many bootstrap samples?**

.. code-block:: python

    # Test different sample sizes
    sample_sizes = [100, 500, 1000, 2000]
    
    for n in sample_sizes:
        ci = Bootstrap.parametric(data, dist, n_bootstrap=n)
        width = ci['loc'].ci_upper - ci['loc'].ci_lower
        print(f"n={n:4d}: CI width = {width:.6f}")

**Guidelines:**

- **n=100**: Quick test, unstable
- **n=500**: Reasonable, fairly stable
- **n=1000**: Standard, good stability ✅
- **n=2000+**: High precision, slow

**Recommendation:** Start with 1000

Confidence Levels
-----------------

.. code-block:: python

    # Different confidence levels
    for conf_level in [0.90, 0.95, 0.99]:
        ci = Bootstrap.parametric(
            data, 
            dist, 
            n_bootstrap=1000,
            confidence_level=conf_level
        )
        
        result = ci['loc']
        print(f"{int(conf_level*100)}% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

**Output:**

::

    90% CI: [9.9156, 10.1191]
    95% CI: [9.8952, 10.1398]
    99% CI: [9.8567, 10.1780]

**Trade-off:**

- Higher confidence → Wider intervals
- 95% is standard in most fields

Bias-Corrected Accelerated (BCa)
---------------------------------

**More accurate method, especially for skewed distributions.**

.. code-block:: python

    # Generate bootstrap samples
    boot_samples = []
    for i in range(1000):
        boot_data = dist.rvs(size=len(data), random_state=i)
        boot_dist = get_distribution('gamma')
        boot_dist.fit(boot_data)
        boot_samples.append(boot_dist.params['a'])
    
    boot_samples = np.array(boot_samples)
    
    # Original estimate
    original_estimate = dist.params['a']
    
    # Define estimator function
    def estimator(d):
        temp_dist = get_distribution('gamma')
        temp_dist.fit(d)
        return temp_dist.params['a']
    
    # BCa confidence interval
    ci_lower, ci_upper = Bootstrap.bca_ci(
        boot_samples,
        original_estimate,
        data,
        estimator,
        confidence_level=0.95
    )
    
    print(f"BCa 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

**When to use BCa:**

- Skewed distributions
- Small samples
- Biased estimators
- Need most accurate CI

Parallel Processing
-------------------

**Speed up with multiple CPU cores:**

.. code-block:: python

    import time
    
    # Serial (1 core)
    start = time.time()
    ci_serial = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=1)
    time_serial = time.time() - start
    
    # Parallel (all cores)
    start = time.time()
    ci_parallel = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=-1)
    time_parallel = time.time() - start
    
    print(f"Serial: {time_serial:.2f}s")
    print(f"Parallel: {time_parallel:.2f}s")
    print(f"Speedup: {time_serial/time_parallel:.1f}x")

**n_jobs parameter:**

- ``n_jobs=1``: Single core
- ``n_jobs=4``: Use 4 cores
- ``n_jobs=-1``: Use all available cores ✅

Practical Example: Risk Analysis
---------------------------------

.. code-block:: python

    # Insurance claim amounts
    np.random.seed(42)
    claims = np.random.lognormal(8, 1.5, 500)
    
    # Fit lognormal distribution
    dist = get_distribution('lognormal')
    dist.fit(claims)
    
    print("Point Estimates:")
    print(f"Mean claim: ${dist.mean():.2f}")
    print(f"95th percentile: ${dist.ppf(0.95):.2f}")
    
    # Bootstrap CI for parameters
    ci_results = Bootstrap.parametric(claims, dist, n_bootstrap=1000)
    
    print("\nParameter Uncertainty:")
    for param, result in ci_results.items():
        print(f"{param}: {result.estimate:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    # Bootstrap CI for derived quantity (mean)
    mean_samples = []
    for i in range(1000):
        boot_dist = get_distribution('lognormal')
        boot_data = dist.rvs(size=len(claims), random_state=i)
        boot_dist.fit(boot_data)
        mean_samples.append(boot_dist.mean())
    
    mean_ci = np.percentile(mean_samples, [2.5, 97.5])
    print(f"\nMean claim 95% CI: ${mean_ci[0]:.2f} - ${mean_ci[1]:.2f}")

Reporting Results
-----------------

**Professional format:**

.. code-block:: python

    # Fit and bootstrap
    dist = get_distribution('normal')
    dist.fit(data)
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    # Report
    print(f"""\nStatistical Analysis Results\n{'='*50}\n\nDistribution: {dist.info.display_name}\nSample size: n = {len(data)}\nEstimation: Maximum Likelihood\n\nParameter Estimates (95% CI):\n""")
    
    for param, result in ci_results.items():
        param_desc = dist.info.parameters[param]
        print(f"  {param_desc}: {result.estimate:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    print(f"\nBootstrap method: {ci_results[list(ci_results.keys())[0]].method}")
    print(f"Bootstrap samples: {ci_results[list(ci_results.keys())[0]].n_bootstrap}")

Best Practices
--------------

1. **Always use bootstrap for inference**
   
   Don't trust point estimates alone.

2. **Choose appropriate method**
   
   - Good fit confirmed → Parametric
   - Uncertain about model → Non-parametric

3. **Use enough samples**
   
   Minimum 1000 for stable results.

4. **Set random seed**
   
   For reproducibility.
   
   .. code-block:: python
   
       Bootstrap.parametric(data, dist, random_state=42)

5. **Check convergence**
   
   .. code-block:: python
   
       # Try different sample sizes
       for n in [500, 1000, 2000]:
           ci = Bootstrap.parametric(data, dist, n_bootstrap=n)
           print(f"n={n}: width={ci['loc'].ci_upper - ci['loc'].ci_lower:.6f}")
       
       # Should stabilize

Next Steps
----------

- :doc:`06_diagnostics` - Residual analysis
- :doc:`07_weighted_data` - Weighted bootstrap
