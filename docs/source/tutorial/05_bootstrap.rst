Tutorial 5: Bootstrap Confidence Intervals
===========================================

Learn how to estimate uncertainty in fitted parameters.

Why Bootstrap?
--------------

Parameter estimates are just **point estimates**. They have uncertainty!

**Questions Bootstrap Answers:**

- How confident are we in the fitted parameters?
- What is the range of plausible values?
- How much does the estimate vary with different samples?

**Example:**

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    
    data = np.random.normal(10, 2, 100)
    dist = get_distribution('normal')
    dist.fit(data)
    
    print(f"Mean estimate: {dist.params['loc']:.4f}")
    # But what is the uncertainty?

Parametric Bootstrap
--------------------

**Resamples from the fitted distribution.**

**How it works:**

1. Fit distribution to data → get parameters θ̂
2. Generate new data from fitted distribution
3. Refit to get new parameters θ*
4. Repeat 1000+ times
5. Use distribution of θ* values to get confidence interval

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    import numpy as np
    from distfit_pro import get_distribution
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)
    
    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Parametric bootstrap (1000 samples)
    ci_results = Bootstrap.parametric(
        data=data,
        distribution=dist,
        n_bootstrap=1000,
        confidence_level=0.95,
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
    95% CI: [9.634521, 10.412987]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000
    
    Bootstrap CI for scale
    ==================================================
    Point Estimate: 1.991847
    95% CI: [1.782341, 2.234561]
    Method: Parametric Bootstrap
    Bootstrap Samples: 1000

**Interpretation:**

- We are 95% confident that the true mean is between 9.63 and 10.41
- We are 95% confident that the true std is between 1.78 and 2.23

Non-Parametric Bootstrap
------------------------

**Resamples from the original data (with replacement).**

**How it works:**

1. Resample n points from data (with replacement)
2. Fit distribution to resampled data
3. Repeat 1000+ times
4. Get confidence intervals from bootstrap distribution

.. code-block:: python

    # Non-parametric bootstrap
    ci_results_np = Bootstrap.nonparametric(
        data=data,
        distribution=dist,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42
    )
    
    for param, result in ci_results_np.items():
        print(result)

**When to use:**

- Don't want to assume fitted distribution is exactly correct
- More robust
- Generally recommended for real data

Parametric vs Non-Parametric
-----------------------------

.. code-block:: python

    # Compare both methods
    print("Parametric Bootstrap:")
    for param, result in ci_results.items():
        print(f"  {param}: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    print("\nNon-Parametric Bootstrap:")
    for param, result in ci_results_np.items():
        print(f"  {param}: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

**Differences:**

+------------------------+------------------------+------------------------+
| Aspect                 | Parametric             | Non-Parametric         |
+========================+========================+========================+
| Assumptions            | Fitted dist is correct | Minimal assumptions    |
+------------------------+------------------------+------------------------+
| Speed                  | Faster (can generate)  | Slower (resampling)    |
+------------------------+------------------------+------------------------+
| Smoothness             | Smoother CIs           | Can be jagged          |
+------------------------+------------------------+------------------------+
| Recommended for        | Well-fitting models    | General use, real data |
+------------------------+------------------------+------------------------+

Parallel Processing
-------------------

**Bootstrap is embarrassingly parallel!**

.. code-block:: python

    # Use all CPU cores
    ci_results = Bootstrap.parametric(
        data=data,
        distribution=dist,
        n_bootstrap=10000,  # More samples!
        n_jobs=-1,  # Use all cores
        random_state=42
    )

**Performance:**

- 1 core: ~30 seconds for 10,000 samples
- 8 cores: ~5 seconds for 10,000 samples

**Progress bar** shows during computation:

::

    Parametric Bootstrap: 100%|██████████| 10000/10000 [00:05<00:00, 1834.56it/s]

Custom Confidence Levels
------------------------

.. code-block:: python

    # 99% confidence interval (stricter)
    ci_99 = Bootstrap.parametric(
        data=data,
        distribution=dist,
        n_bootstrap=1000,
        confidence_level=0.99
    )
    
    # 90% confidence interval (wider)
    ci_90 = Bootstrap.parametric(
        data=data,
        distribution=dist,
        n_bootstrap=1000,
        confidence_level=0.90
    )
    
    # Compare widths
    param = 'loc'
    width_99 = ci_99[param].ci_upper - ci_99[param].ci_lower
    width_90 = ci_90[param].ci_upper - ci_90[param].ci_lower
    
    print(f"99% CI width: {width_99:.4f}")
    print(f"90% CI width: {width_90:.4f}")

Bias-Corrected and Accelerated (BCa)
-------------------------------------

**More accurate than percentile method.**

.. code-block:: python

    # Generate bootstrap samples manually
    boot_samples_loc = []
    
    for i in range(1000):
        boot_data = np.random.choice(data, size=len(data), replace=True)
        boot_dist = get_distribution('normal')
        boot_dist.fit(boot_data)
        boot_samples_loc.append(boot_dist.params['loc'])
    
    boot_samples_loc = np.array(boot_samples_loc)
    
    # BCa confidence interval
    def estimate_mean(d):
        return np.mean(d)
    
    ci_lower, ci_upper = Bootstrap.bca_ci(
        bootstrap_samples=boot_samples_loc,
        original_estimate=dist.params['loc'],
        data=data,
        estimator_func=estimate_mean,
        confidence_level=0.95
    )
    
    print(f"BCa 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

**When to use:**

- Skewed bootstrap distributions
- Small samples
- More accurate (but slower)

Small Sample Example
--------------------

**Bootstrap shines with small samples!**

.. code-block:: python

    # Only 20 observations
    small_data = np.random.gamma(2, 3, 20)
    
    dist_small = get_distribution('gamma')
    dist_small.fit(small_data)
    
    # Bootstrap CI
    ci_small = Bootstrap.nonparametric(
        data=small_data,
        distribution=dist_small,
        n_bootstrap=2000,  # More samples for small data
        confidence_level=0.95
    )
    
    for param, result in ci_small.items():
        print(f"{param}:")
        print(f"  Estimate: {result.estimate:.4f}")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        width = result.ci_upper - result.ci_lower
        print(f"  Width: {width:.4f}")

**Notice:** CIs are wider with small samples (more uncertainty).

Multiple Distributions
----------------------

**Compare uncertainty across distributions:**

.. code-block:: python

    data = np.random.gamma(2, 3, 200)
    
    candidates = ['gamma', 'lognormal', 'weibull']
    results = {}
    
    for dist_name in candidates:
        dist = get_distribution(dist_name)
        dist.fit(data)
        
        # Bootstrap
        ci = Bootstrap.nonparametric(
            data=data,
            distribution=dist,
            n_bootstrap=1000
        )
        
        results[dist_name] = ci
    
    # Compare
    print("\nParameter Uncertainty Comparison:")
    for dist_name, ci_dict in results.items():
        print(f"\n{dist_name}:")
        for param, result in ci_dict.items():
            width = result.ci_upper - result.ci_lower
            print(f"  {param}: width = {width:.4f}")

Reporting Results
-----------------

**How to present bootstrap CIs in reports:**

.. code-block:: python

    dist.fit(data)
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    # Format for reports
    mu = ci_results['loc']
    sigma = ci_results['scale']
    
    report = f"""
    Distribution: Normal
    
    Parameters (95% CI):
      μ (mean):  {mu.estimate:.3f} [{mu.ci_lower:.3f}, {mu.ci_upper:.3f}]
      σ (std):   {sigma.estimate:.3f} [{sigma.ci_lower:.3f}, {sigma.ci_upper:.3f}]
    
    Method: Parametric Bootstrap (n=1000)
    """
    
    print(report)

**LaTeX format:**

.. code-block:: python

    print(f"$\\mu = {mu.estimate:.3f}$ (95\% CI: [{mu.ci_lower:.3f}, {mu.ci_upper:.3f}])")

Troubleshooting
---------------

**Failed Bootstrap Samples**

.. code-block:: python

    # Some bootstrap samples may fail to fit
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    # Check how many succeeded
    print(f"Successful samples: {ci_results['loc'].n_bootstrap}")
    
    # If < 90% succeeded, warning is printed

**Too Slow**

.. code-block:: python

    # Use parallel processing
    ci_results = Bootstrap.parametric(
        data, dist, 
        n_bootstrap=1000,
        n_jobs=-1  # All cores
    )
    
    # Or reduce bootstrap samples (minimum ~500)
    ci_results = Bootstrap.parametric(
        data, dist, 
        n_bootstrap=500
    )

Best Practices
--------------

1. **Use 1000+ bootstrap samples**
   
   .. code-block:: python
   
       n_bootstrap=1000  # Good
       n_bootstrap=10000  # Better (if time allows)

2. **Prefer non-parametric for real data**
   
   .. code-block:: python
   
       Bootstrap.nonparametric(data, dist, ...)

3. **Use parallel processing**
   
   .. code-block:: python
   
       n_jobs=-1

4. **Set random seed for reproducibility**
   
   .. code-block:: python
   
       random_state=42

5. **Report method and n_bootstrap**
   
   "95% CI via non-parametric bootstrap (n=1000)"

6. **Check if CIs are reasonable**
   
   .. code-block:: python
   
       # Should contain the point estimate
       assert ci_results['loc'].ci_lower <= dist.params['loc'] <= ci_results['loc'].ci_upper

Next Steps
----------

- :doc:`06_diagnostics` - Residuals and outliers
- :doc:`07_weighted_data` - Weighted bootstrap
- :doc:`examples/advanced` - Real-world examples
