Tutorial 5: Bootstrap Confidence Intervals
===========================================

Learn how to quantify uncertainty in parameter estimates using bootstrap methods.

Why Bootstrap?
--------------

**Problem:** You fit a distribution and get parameter estimates, but how confident are you?

**Solution:** Bootstrap resampling provides confidence intervals without assumptions about sampling distributions.

**Benefits:**

- No need for mathematical formulas
- Works for any statistic
- Accounts for sample variability
- Easy to understand and implement

Parametric Bootstrap
--------------------

**Idea:** Resample from the fitted distribution.

**Steps:**

1. Fit distribution to data → get θ̂
2. Generate B bootstrap samples from f(x; θ̂)
3. Refit distribution to each bootstrap sample → get θ̂*₁, θ̂*₂, ..., θ̂*ᵦ
4. Calculate percentiles of {θ̂*ᵢ} for CI

**Code:**

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
        data=data,
        distribution=dist,
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
    95% CI: [9.894521, 10.138976]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000
    
    Bootstrap CI for scale
    ==================================================
    Point Estimate: 1.991847
    95% CI: [1.948132, 2.036543]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000

**Interpretation:**

- We're 95% confident the true mean is between 9.89 and 10.14
- We're 95% confident the true std is between 1.95 and 2.04

Non-parametric Bootstrap
------------------------

**Idea:** Resample from the observed data (with replacement).

**Steps:**

1. Fit distribution to data → get θ̂
2. Generate B bootstrap samples by resampling data
3. Refit distribution to each bootstrap sample
4. Calculate percentiles for CI

**More conservative** than parametric (doesn't assume fitted distribution is correct).

**Code:**

.. code-block:: python

    # Non-parametric bootstrap
    ci_nonparam = Bootstrap.nonparametric(
        data=data,
        distribution=dist,
        n_bootstrap=1000,
        confidence_level=0.95,
        n_jobs=-1
    )
    
    for param, result in ci_nonparam.items():
        print(result)

**When to use:**

- Not confident about distribution choice
- Want more conservative estimates
- Data may have unusual features

Comparing Methods
-----------------

.. code-block:: python

    print("Parametric Bootstrap:")
    for p, r in ci_param.items():
        print(f"  {p}: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
    
    print("\nNon-parametric Bootstrap:")
    for p, r in ci_nonparam.items():
        print(f"  {p}: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")

**Typical pattern:**

- Non-parametric CIs are usually **wider** (more conservative)
- If CIs are very different, distribution may not fit well

Bias-Corrected Accelerated (BCa)
---------------------------------

**Most accurate bootstrap CI method.**

Corrects for:

1. **Bias** - when bootstrap distribution is shifted
2. **Skewness** - when bootstrap distribution is asymmetric

**Code:**

.. code-block:: python

    # BCa requires manual implementation for each parameter
    # Example for mean
    def estimate_mean(sample):
        d = get_distribution('normal')
        d.fit(sample)
        return d.params['loc']
    
    # Get bootstrap samples for loc
    boot_samples_loc = []
    for i in range(1000):
        boot_data = dist.rvs(size=len(data), random_state=i)
        d = get_distribution('normal')
        d.fit(boot_data)
        boot_samples_loc.append(d.params['loc'])
    
    boot_samples_loc = np.array(boot_samples_loc)
    
    # BCa CI
    bca_ci = Bootstrap.bca_ci(
        bootstrap_samples=boot_samples_loc,
        original_estimate=dist.params['loc'],
        data=data,
        estimator_func=estimate_mean,
        confidence_level=0.95
    )
    
    print(f"BCa CI: [{bca_ci[0]:.4f}, {bca_ci[1]:.4f}]")

**When to use:**

- Need most accurate CIs
- Parameter distribution is skewed
- Have computational resources

Practical Examples
------------------

Example 1: Small Sample
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Only 30 observations
    small_data = np.random.gamma(2, 3, 30)
    
    dist = get_distribution('gamma')
    dist.fit(small_data)
    
    # Bootstrap CI (more reliable with small n)
    ci = Bootstrap.parametric(
        small_data, dist,
        n_bootstrap=2000,  # More samples for small data
        n_jobs=-1
    )
    
    for param, result in ci.items():
        width = result.ci_upper - result.ci_lower
        print(f"{param}: {result.estimate:.3f} ± {width/2:.3f}")

Example 2: Reliability Metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Component lifetimes
    lifetimes = np.random.weibull(1.5, 500) * 1000
    
    dist = get_distribution('weibull')
    dist.fit(lifetimes)
    
    # Bootstrap
    ci = Bootstrap.parametric(lifetimes, dist, n_bootstrap=1000)
    
    # Now calculate derived metric with uncertainty
    # MTTF = scale * Gamma(1 + 1/k)
    from scipy.special import gamma as gamma_func
    
    def mttf_from_params(params):
        k = params['c']
        lam = params['scale']
        return lam * gamma_func(1 + 1/k)
    
    # Bootstrap MTTF
    boot_mttfs = []
    for i in range(1000):
        boot_sample = dist.rvs(size=len(lifetimes), random_state=i)
        d = get_distribution('weibull')
        d.fit(boot_sample)
        boot_mttfs.append(mttf_from_params(d.params))
    
    mttf_ci = Bootstrap.percentile_ci(np.array(boot_mttfs), 0.95)
    print(f"MTTF: {mttf_from_params(dist.params):.1f}")
    print(f"95% CI: [{mttf_ci[0]:.1f}, {mttf_ci[1]:.1f}]")

Example 3: Quantile CI
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # CI for 99th percentile (VaR)
    returns = np.random.standard_t(df=5, size=1000) * 0.02
    
    dist = get_distribution('studentt')
    dist.fit(returns)
    
    # Bootstrap 99th percentile
    boot_q99 = []
    for i in range(1000):
        boot_sample = dist.rvs(size=len(returns), random_state=i)
        d = get_distribution('studentt')
        d.fit(boot_sample)
        boot_q99.append(d.ppf(0.99))
    
    q99_ci = Bootstrap.percentile_ci(np.array(boot_q99), 0.95)
    
    print(f"99th percentile: {dist.ppf(0.99):.4f}")
    print(f"95% CI: [{q99_ci[0]:.4f}, {q99_ci[1]:.4f}]")

Choosing Bootstrap Parameters
------------------------------

**Number of Bootstrap Samples (B)**

.. code-block:: python

    # Quick check
    B = 100  # Fast but less stable
    
    # Standard
    B = 1000  # Good balance
    
    # High precision
    B = 10000  # Slow but very stable
    
    # Publication quality
    B = 50000  # For important results

**Rule of thumb:** B = 1000 is usually enough.

**Confidence Level**

.. code-block:: python

    # Standard
    confidence_level = 0.95  # 95% CI
    
    # Stricter
    confidence_level = 0.99  # 99% CI
    
    # More lenient
    confidence_level = 0.90  # 90% CI

Parallel Processing
-------------------

**Speed up bootstrap with multiple cores:**

.. code-block:: python

    # Use all cores
    ci = Bootstrap.parametric(data, dist, n_jobs=-1)
    
    # Use 4 cores
    ci = Bootstrap.parametric(data, dist, n_jobs=4)
    
    # No parallelization
    ci = Bootstrap.parametric(data, dist, n_jobs=1)

**Speedup example:**

::

    1000 samples, 10000 data points
    n_jobs=1:  45 seconds
    n_jobs=4:  13 seconds (3.5x faster)
    n_jobs=-1: 8 seconds (5.6x faster on 8-core CPU)

Limitations
-----------

**1. Assumes IID data**

Bootstrap assumes independent, identically distributed observations.

**Not suitable for:**
- Time series (autocorrelation)
- Spatial data (spatial correlation)
- Clustered data

**2. Computational cost**

.. code-block:: python

    # Can be slow for large data/many samples
    # 1M data points × 10k bootstrap = 10 billion fits!

**3. Model misspecification**

Parametric bootstrap assumes the fitted model is correct.

**Solution:** Use non-parametric or check GOF tests first.

Best Practices
--------------

1. **Always set random_state for reproducibility**
   
   .. code-block:: python
   
       ci = Bootstrap.parametric(data, dist, random_state=42)

2. **Use parallel processing**
   
   .. code-block:: python
   
       ci = Bootstrap.parametric(data, dist, n_jobs=-1)

3. **Check convergence**
   
   .. code-block:: python
   
       # Try different B
       for B in [100, 500, 1000, 2000]:
           ci = Bootstrap.parametric(data, dist, n_bootstrap=B)
           print(f"B={B}: {ci['loc'].ci_lower:.4f}")
       
       # Should stabilize

4. **Compare methods**
   
   .. code-block:: python
   
       # Both parametric and non-parametric
       # If very different, investigate why

5. **Visualize bootstrap distribution**
   
   .. code-block:: python
   
       import matplotlib.pyplot as plt
       
       boot_params = []  # Collect from bootstrap
       plt.hist(boot_params, bins=50)
       plt.axvline(ci_lower, color='r')
       plt.axvline(ci_upper, color='r')
       plt.show()

Next Steps
----------

- :doc:`06_diagnostics` - Residuals and outliers
- :doc:`07_weighted_data` - Weighted bootstrap
- :doc:`08_visualization` - Plot bootstrap results
