Tutorial 3: Fitting Methods
===========================

Learn different parameter estimation methods.

Maximum Likelihood Estimation (MLE)
------------------------------------

**Most common and usually best method.**

Finds parameters that maximize the likelihood of observing the data.

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    
    data = np.random.normal(10, 2, 1000)
    
    dist = get_distribution('normal')
    dist.fit(data, method='mle')
    
    print(f"MLE Parameters: {dist.params}")

**Advantages:**

- Asymptotically efficient
- Consistent
- Usually most accurate

**Disadvantages:**

- Can fail with small samples
- Sensitive to outliers
- May require numerical optimization

Method of Moments
-----------------

**Matches sample moments to theoretical moments.**

.. code-block:: python

    dist = get_distribution('normal')
    dist.fit(data, method='moments')
    
    print(f"Moments Parameters: {dist.params}")

**How it works:**

For Normal distribution:

- Sample mean → μ
- Sample std → σ

.. code-block:: python

    # Manually
    mu_mom = np.mean(data)
    sigma_mom = np.std(data, ddof=1)
    
    print(f"Manual: μ={mu_mom:.4f}, σ={sigma_mom:.4f}")

**Advantages:**

- Simple and fast
- Always converges
- Good for quick estimates

**Disadvantages:**

- Less efficient than MLE
- May give impossible values
- Not always optimal

Quantile Matching
-----------------

**Matches empirical quantiles to theoretical quantiles.**

.. code-block:: python

    dist = get_distribution('weibull')
    dist.fit(data, method='quantile', quantiles=[0.25, 0.5, 0.75])
    
    print(f"Quantile Parameters: {dist.params}")

**Use when:**

- Robust estimation needed
- Tails are important
- MLE fails

**Example with custom quantiles:**

.. code-block:: python

    # Focus on tails
    dist.fit(data, method='quantile', quantiles=[0.1, 0.5, 0.9])

Comparing Methods
-----------------

**Example: Which method is best?**

.. code-block:: python

    import numpy as np
    from distfit_pro import get_distribution
    
    # Generate data
    np.random.seed(42)
    data = np.random.gamma(2, 3, 1000)
    
    methods = ['mle', 'moments', 'quantile']
    results = {}
    
    for method in methods:
        dist = get_distribution('gamma')
        dist.fit(data, method=method)
        
        # Calculate log-likelihood
        log_lik = np.sum(dist.logpdf(data))
        
        results[method] = {
            'params': dist.params,
            'log_likelihood': log_lik
        }
        
        print(f"\n{method.upper()}:")
        print(f"  Parameters: {dist.params}")
        print(f"  Log-likelihood: {log_lik:.2f}")

**Output:**

::

    MLE:
      Parameters: {'a': 2.01, 'scale': 2.98}
      Log-likelihood: -2456.32
    
    MOMENTS:
      Parameters: {'a': 1.97, 'scale': 3.05}
      Log-likelihood: -2457.89
    
    QUANTILE:
      Parameters: {'a': 1.99, 'scale': 3.01}
      Log-likelihood: -2456.78

**Winner:** MLE (highest log-likelihood)

Handling Edge Cases
-------------------

Small Sample Sizes
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Only 20 observations
    small_data = np.random.normal(10, 2, 20)
    
    # MLE might be unstable
    dist = get_distribution('normal')
    
    try:
        dist.fit(small_data, method='mle')
        print("MLE succeeded")
    except:
        print("MLE failed, using moments")
        dist.fit(small_data, method='moments')

Outliers
^^^^^^^^

.. code-block:: python

    # Data with outliers
    data_with_outliers = np.concatenate([
        np.random.normal(10, 2, 950),
        np.random.uniform(50, 100, 50)  # 5% outliers
    ])
    
    # Quantile method is more robust
    dist = get_distribution('normal')
    dist.fit(data_with_outliers, method='quantile')

Bounded Data
^^^^^^^^^^^^

.. code-block:: python

    # Data must be positive
    positive_data = np.abs(np.random.normal(5, 2, 1000))
    
    # Use distribution with positive support
    dist = get_distribution('lognormal')
    dist.fit(positive_data, method='mle')
    
    # NOT Normal (can give negative values)

Custom Initialization
---------------------

Provide starting values for optimization:

.. code-block:: python

    # For difficult cases
    dist = get_distribution('weibull')
    
    # MLE with custom starting point
    params_mle = dist.fit_mle(
        data,
        # Custom starting values can be passed via kwargs
        # depending on distribution implementation
    )

Best Practices
--------------

1. **Start with MLE**
   
   .. code-block:: python
   
       dist.fit(data, method='mle')

2. **Check if it worked**
   
   .. code-block:: python
   
       if not dist.fitted:
           print("Fit failed!")
       
       # Check parameters are reasonable
       print(dist.params)

3. **Try moments if MLE fails**
   
   .. code-block:: python
   
       try:
           dist.fit(data, method='mle')
       except:
           dist.fit(data, method='moments')

4. **Use quantile for robustness**
   
   .. code-block:: python
   
       # When outliers suspected
       dist.fit(data, method='quantile')

5. **Compare methods**
   
   .. code-block:: python
   
       # Fit with both
       dist_mle = get_distribution('gamma')
       dist_mle.fit(data, method='mle')
       
       dist_mom = get_distribution('gamma')
       dist_mom.fit(data, method='moments')
       
       # Compare AIC
       aic_mle = 2 * len(dist_mle.params) - 2 * np.sum(dist_mle.logpdf(data))
       aic_mom = 2 * len(dist_mom.params) - 2 * np.sum(dist_mom.logpdf(data))
       
       print(f"MLE AIC: {aic_mle:.2f}")
       print(f"Moments AIC: {aic_mom:.2f}")
       
       if aic_mle < aic_mom:
           print("MLE is better")
       else:
           print("Moments is better")

Next Steps
----------

- :doc:`04_gof_tests` - Test goodness-of-fit
- :doc:`05_bootstrap` - Get confidence intervals
