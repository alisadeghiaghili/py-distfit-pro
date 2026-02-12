Tutorial 5: Bootstrap Confidence Intervals
===========================================

Learn how to quantify uncertainty in parameter estimates.

Why Bootstrap?
--------------

**Point estimates aren't enough!**

When you fit a distribution, you get parameter estimates. But how confident are you in those values?

.. code-block:: python

    dist.fit(data)
    print(f"μ = {dist.params['loc']:.4f}")  # Point estimate
    
    # But what's the uncertainty? 🤔

**Bootstrap answers:**

- What's the range of plausible parameter values?
- How stable are the estimates?
- What's the standard error?

Parametric Bootstrap
--------------------

**Resamples from the fitted distribution.**

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
    
    # View results
    for param, result in ci_results.items():
        print(result)

**Output:**

::

    Bootstrap CI for loc
    ==================================================
    Point Estimate: 10.017342
    95% CI: [9.895234, 10.139451]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000
    
    Bootstrap CI for scale
    ==================================================
    Point Estimate: 1.991847
    95% CI: [1.930123, 2.053571]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000

**How it works:**

1. Generate synthetic data from fitted distribution
2. Refit distribution to synthetic data
3. Repeat 1000 times
4. Calculate percentiles of bootstrap estimates

Non-Parametric Bootstrap
------------------------

**Resamples from the original data.**

More conservative, makes fewer assumptions.

.. code-block:: python

    # Non-parametric bootstrap
    ci_nonparam = Bootstrap.nonparametric(
        data, 
        dist, 
        n_bootstrap=1000,
        confidence_level=0.95,
        n_jobs=-1
    )
    
    for param, result in ci_nonparam.items():
        print(result)

**Differences from Parametric:**

- ✅ More robust (doesn't assume distribution is correct)
- ✅ Better for small samples
- ❌ Wider intervals (more conservative)
- ❌ Slower for large datasets

Choosing Bootstrap Type
-----------------------

**Use Parametric when:**

- Confident in distribution choice
- GOF tests passed
- Need precise intervals
- Large dataset available

**Use Non-Parametric when:**

- Uncertain about distribution
- Small sample
- Want conservative estimates
- Robustness is priority

**Example comparison:**

.. code-block:: python

    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Both methods
    ci_param = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    ci_nonparam = Bootstrap.nonparametric(data, dist, n_bootstrap=1000)
    
    # Compare for 'loc' parameter
    print("Parametric CI:")
    print(f"  [{ci_param['loc'].ci_lower:.4f}, {ci_param['loc'].ci_upper:.4f}]")
    
    print("\nNon-parametric CI:")
    print(f"  [{ci_nonparam['loc'].ci_lower:.4f}, {ci_nonparam['loc'].ci_upper:.4f}]")

Bias-Corrected Accelerated (BCa)
---------------------------------

**Most accurate bootstrap method.**

Adjusts for bias and skewness in bootstrap distribution.

.. code-block:: python

    # BCa requires custom estimator function
    def estimate_mean(data):
        """Estimator function for BCa"""
        dist_temp = get_distribution('normal')
        dist_temp.fit(data)
        return dist_temp.params['loc']
    
    # Generate bootstrap samples first
    boot_samples = []
    for i in range(1000):
        boot_data = np.random.choice(data, size=len(data), replace=True)
        boot_estimate = estimate_mean(boot_data)
        boot_samples.append(boot_estimate)
    
    boot_samples = np.array(boot_samples)
    
    # BCa CI
    original_estimate = dist.params['loc']
    ci_lower, ci_upper = Bootstrap.bca_ci(
        boot_samples,
        original_estimate,
        data,
        estimate_mean,
        confidence_level=0.95
    )
    
    print(f"BCa 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

**When to use BCa:**

- Small samples (n < 100)
- Skewed bootstrap distribution
- Maximum accuracy needed
- Academic/research settings

Confidence Levels
-----------------

**Default is 95%, but you can change it:**

.. code-block:: python

    # 90% CI (narrower)
    ci_90 = Bootstrap.parametric(data, dist, confidence_level=0.90)
    
    # 99% CI (wider)
    ci_99 = Bootstrap.parametric(data, dist, confidence_level=0.99)
    
    # Compare widths
    for level, ci in [(0.90, ci_90), (0.95, ci_results), (0.99, ci_99)]:
        width = ci['loc'].ci_upper - ci['loc'].ci_lower
        print(f"{int(level*100)}% CI width: {width:.4f}")

**Output:**

::

    90% CI width: 0.2043
    95% CI width: 0.2442
    99% CI width: 0.3215

Parallel Processing
-------------------

**Bootstrap is embarrassingly parallel!**

Speed up with multiple CPU cores:

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

**Typical speedup:** 4-6x on 8-core CPU

Progress Bars
-------------

**Track bootstrap progress with tqdm:**

.. code-block:: python

    # Automatically shows progress bar
    ci = Bootstrap.parametric(data, dist, n_bootstrap=5000)

**Output:**

::

    Parametric Bootstrap: 100%|██████████| 5000/5000 [00:12<00:00, 412.34it/s]

Interpreting Results
--------------------

**What do confidence intervals tell you?**

.. code-block:: python

    result = ci_results['loc']
    print(result)

**Interpretation:**

1. **Point Estimate:** Best guess for parameter
2. **CI Lower/Upper:** Plausible range
3. **Confidence Level:** How often CI contains true value

**Example:**

::

    95% CI: [9.895, 10.139]

- We're 95% confident the true μ is between 9.895 and 10.139
- If we repeated this experiment 100 times, ~95 CIs would contain true μ
- Narrower CI = more precise estimate

Practical Examples
------------------

**Example 1: Small Sample Uncertainty**

.. code-block:: python

    # Only 30 observations
    small_data = np.random.normal(10, 2, 30)
    
    dist = get_distribution('normal')
    dist.fit(small_data)
    
    # Bootstrap shows high uncertainty
    ci = Bootstrap.nonparametric(small_data, dist, n_bootstrap=1000)
    
    width = ci['loc'].ci_upper - ci['loc'].ci_lower
    print(f"CI width: {width:.4f}")  # Wide interval!

**Example 2: Reliability Analysis**

.. code-block:: python

    # Component lifetimes
    lifetimes = np.random.weibull(2, 100) * 1000
    
    dist = get_distribution('weibull')
    dist.fit(lifetimes)
    
    # Bootstrap CI for Weibull parameters
    ci = Bootstrap.parametric(lifetimes, dist, n_bootstrap=2000)
    
    # Reliability at t=500 with uncertainty
    # (would need additional bootstrapping of reliability function)

**Example 3: Comparing Estimators**

.. code-block:: python

    # Compare MLE vs Moments
    dist_mle = get_distribution('gamma')
    dist_mle.fit(data, method='mle')
    
    dist_mom = get_distribution('gamma')
    dist_mom.fit(data, method='moments')
    
    # Bootstrap both
    ci_mle = Bootstrap.parametric(data, dist_mle, n_bootstrap=1000)
    ci_mom = Bootstrap.parametric(data, dist_mom, n_bootstrap=1000)
    
    # Compare widths (narrower = more efficient)
    for param in ci_mle.keys():
        width_mle = ci_mle[param].ci_upper - ci_mle[param].ci_lower
        width_mom = ci_mom[param].ci_upper - ci_mom[param].ci_lower
        print(f"{param}: MLE width={width_mle:.4f}, Moments width={width_mom:.4f}")

Bootstrap Diagnostics
---------------------

**Check bootstrap distribution:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Collect bootstrap estimates
    boot_estimates = []
    for i in range(1000):
        boot_sample = dist.rvs(size=len(data), random_state=i)
        dist_boot = get_distribution('normal')
        dist_boot.fit(boot_sample)
        boot_estimates.append(dist_boot.params['loc'])
    
    boot_estimates = np.array(boot_estimates)
    
    # Plot histogram
    plt.hist(boot_estimates, bins=50, density=True, alpha=0.7)
    plt.axvline(dist.params['loc'], color='red', label='Original estimate')
    plt.axvline(np.percentile(boot_estimates, 2.5), 
                color='blue', linestyle='--', label='95% CI')
    plt.axvline(np.percentile(boot_estimates, 97.5), 
                color='blue', linestyle='--')
    plt.xlabel('Parameter value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Bootstrap Distribution')
    plt.show()

Best Practices
--------------

1. **Use enough bootstrap samples**
   
   - Minimum: 1000
   - Standard: 2000-5000
   - Publication: 10,000+

2. **Set random seed for reproducibility**
   
   .. code-block:: python
   
       Bootstrap.parametric(data, dist, random_state=42)

3. **Use parallel processing**
   
   .. code-block:: python
   
       Bootstrap.parametric(data, dist, n_jobs=-1)

4. **Check GOF first**
   
   Bootstrap assumes your distribution is reasonable!

5. **Report CI width**
   
   Not just the interval, but how wide it is.

Limitations
-----------

**Bootstrap doesn't fix:**

- Poor distribution choice
- Model misspecification
- Biased data
- Small sample fundamental limits

**Bootstrap helps with:**

- Quantifying uncertainty
- Non-normal parameter distributions
- Complex estimators
- Hypothesis testing

Next Steps
----------

- :doc:`06_diagnostics` - Detailed model diagnostics
- :doc:`07_weighted_data` - Weighted bootstrap
- :doc:`08_visualization` - Visualize uncertainty
