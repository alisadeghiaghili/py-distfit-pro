Tutorial 7: Weighted Data
=========================

Learn to fit distributions to weighted observations.

When to Use Weights
-------------------

**Use weighted fitting when:**

1. **Survey Data** - sampling weights
2. **Stratified Sampling** - different strata sizes
3. **Aggregated Data** - frequency counts
4. **Reliability** - different observation qualities
5. **Meta-Analysis** - combining studies

Basic Weighted Fitting
----------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.weighted import WeightedFitting
    import numpy as np
    
    # Data with different reliabilities
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Weights (higher = more reliable)
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    # Weighted MLE
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    
    dist.params = params
    dist.fitted = True
    
    print("Weighted fit:")
    print(dist.summary())

Weighted MLE
------------

**Maximum likelihood with observation weights.**

.. code-block:: python

    # Weighted MLE
    params_weighted = WeightedFitting.fit_weighted_mle(
        data,
        weights,
        dist
    )
    
    print("Weighted parameters:", params_weighted)

**How it works:**

Maximizes weighted log-likelihood:

.. math::

    \sum_{i=1}^n w_i \log f(x_i; \theta)

Where:
- :math:`w_i` = weight for observation :math:`i`
- :math:`f(x_i; \theta)` = PDF with parameters :math:`\theta`

Weighted Method of Moments
---------------------------

**Faster alternative to weighted MLE.**

.. code-block:: python

    # Weighted moments
    params_moments = WeightedFitting.fit_weighted_moments(
        data,
        weights,
        dist
    )
    
    print("Weighted moments:", params_moments)

**How it works:**

Calculates weighted moments:

.. code-block:: python

    # Normalize weights
    w = weights / np.sum(weights)
    
    # Weighted mean
    wmean = np.sum(w * data)
    
    # Weighted variance
    wvar = np.sum(w * (data - wmean)**2)

Comparing Methods
-----------------

.. code-block:: python

    import numpy as np
    from distfit_pro import get_distribution
    from distfit_pro.core.weighted import WeightedFitting
    
    np.random.seed(42)
    data = np.random.gamma(2, 3, 1000)
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    dist = get_distribution('gamma')
    
    # Unweighted MLE
    dist.fit(data, method='mle')
    params_unweighted = dist.params.copy()
    
    # Weighted MLE
    params_wmle = WeightedFitting.fit_weighted_mle(data, weights, dist)
    
    # Weighted moments
    params_wmom = WeightedFitting.fit_weighted_moments(data, weights, dist)
    
    print("Comparison:")
    print(f"Unweighted MLE: {params_unweighted}")
    print(f"Weighted MLE:   {params_wmle}")
    print(f"Weighted Mom:   {params_wmom}")

Weighted Statistics
-------------------

**Calculate weighted summary statistics.**

.. code-block:: python

    # Weighted statistics
    stats = WeightedFitting.weighted_stats(data, weights)
    
    print("Weighted Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

**Output:**

::

    Weighted Statistics:
      mean: 10.1234
      var: 3.9876
      std: 1.9969
      median: 10.0987
      q25: 8.6543
      q75: 11.5432

Weighted Quantiles
------------------

.. code-block:: python

    # Calculate weighted median
    median = WeightedFitting.weighted_quantile(data, weights, 0.5)
    print(f"Weighted median: {median:.4f}")
    
    # Weighted quartiles
    q25 = WeightedFitting.weighted_quantile(data, weights, 0.25)
    q75 = WeightedFitting.weighted_quantile(data, weights, 0.75)
    
    print(f"IQR: [{q25:.4f}, {q75:.4f}]")
    
    # Any quantile
    q95 = WeightedFitting.weighted_quantile(data, weights, 0.95)
    print(f"95th percentile: {q95:.4f}")

Effective Sample Size
---------------------

**Weights reduce effective sample size.**

.. code-block:: python

    # Calculate effective sample size
    n = len(data)
    ess = WeightedFitting.effective_sample_size(weights)
    
    print(f"Actual sample size: {n}")
    print(f"Effective sample size: {ess:.1f}")
    print(f"Efficiency: {ess/n*100:.1f}%")

**Formula:**

.. math::

    ESS = \frac{(\sum w_i)^2}{\sum w_i^2}

**Interpretation:**

- Equal weights → ESS = n
- Unequal weights → ESS < n
- Very unequal → Much smaller ESS

Practical Examples
------------------

Example 1: Survey Data
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Survey responses with sampling weights
    np.random.seed(42)
    
    # Income data (stratified sample)
    incomes = np.concatenate([
        np.random.lognormal(10, 0.5, 300),   # Low income (oversampled)
        np.random.lognormal(11, 0.7, 500),   # Middle income
        np.random.lognormal(12, 0.9, 200)    # High income (undersampled)
    ])
    
    # Sampling weights (inverse of sampling probability)
    weights = np.concatenate([
        np.ones(300) * 0.5,   # Low income: weight down
        np.ones(500) * 1.0,   # Middle: normal weight
        np.ones(200) * 2.0    # High income: weight up
    ])
    
    # Fit lognormal distribution
    dist = get_distribution('lognormal')
    
    # Weighted fit (corrects for sampling)
    params_weighted = WeightedFitting.fit_weighted_mle(incomes, weights, dist)
    dist.params = params_weighted
    dist.fitted = True
    
    print("Survey-weighted income distribution:")
    print(f"Median income: ${np.exp(dist.params['scale']):.2f}")
    print(f"Mean income: ${dist.mean():.2f}")

Example 2: Frequency Data
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Aggregated data: value + count
    values = np.array([1, 2, 3, 4, 5, 6])
    counts = np.array([5, 15, 25, 30, 20, 5])  # Frequencies
    
    # Expand to full data (inefficient)
    # data_expanded = np.repeat(values, counts)
    
    # Better: use weights = counts
    dist = get_distribution('poisson')
    params = WeightedFitting.fit_weighted_mle(values, counts, dist)
    
    dist.params = params
    dist.fitted = True
    
    print(f"Fitted Poisson λ: {params['mu']:.4f}")

Example 3: Reliability-Weighted Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Measurements with different precisions
    measurements = np.array([10.2, 9.8, 10.1, 9.9, 10.0])
    
    # Standard deviations (measurement errors)
    std_errors = np.array([0.1, 0.3, 0.15, 0.2, 0.1])
    
    # Weights = 1 / variance
    weights = 1 / (std_errors ** 2)
    
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(measurements, weights, dist)
    
    print(f"Precision-weighted mean: {params['loc']:.4f}")
    print(f"Precision-weighted std: {params['scale']:.4f}")

Weighted Bootstrap
------------------

**Combine weighting with bootstrap for uncertainty.**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    # Weighted fit
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    dist.params = params
    dist.fitted = True
    
    # For bootstrap, resample with weights as probabilities
    # Normalize weights to probabilities
    probs = weights / np.sum(weights)
    
    # Custom weighted bootstrap
    n_bootstrap = 1000
    boot_params = []
    
    for i in range(n_bootstrap):
        # Resample with probability = weight
        boot_indices = np.random.choice(
            len(data), 
            size=len(data), 
            replace=True,
            p=probs
        )
        
        boot_data = data[boot_indices]
        boot_weights = weights[boot_indices]
        
        # Fit to bootstrap sample
        boot_dist = get_distribution('normal')
        boot_params_i = WeightedFitting.fit_weighted_mle(
            boot_data, 
            boot_weights, 
            boot_dist
        )
        boot_params.append(boot_params_i)
    
    # Calculate CI
    loc_samples = [p['loc'] for p in boot_params]
    ci_loc = np.percentile(loc_samples, [2.5, 97.5])
    
    print(f"Weighted estimate: {params['loc']:.4f}")
    print(f"95% CI: [{ci_loc[0]:.4f}, {ci_loc[1]:.4f}]")

Best Practices
--------------

1. **Normalize weights**
   
   WeightedFitting does this automatically, but be aware.

2. **Check effective sample size**
   
   .. code-block:: python
   
       ess = WeightedFitting.effective_sample_size(weights)
       if ess < 30:
           print("⚠️  Warning: Low effective sample size")

3. **Handle zero weights**
   
   .. code-block:: python
   
       # Remove zero-weight observations
       mask = weights > 0
       data_clean = data[mask]
       weights_clean = weights[mask]

4. **Validate weights**
   
   .. code-block:: python
   
       # Check for negative weights
       if np.any(weights < 0):
           raise ValueError("Weights must be non-negative")
       
       # Check for NaN
       if np.any(np.isnan(weights)):
           print("⚠️  Warning: NaN weights detected")

5. **Document weighting scheme**
   
   Always explain how weights were calculated!

Common Pitfalls
---------------

**Pitfall 1: Using counts as weights incorrectly**

.. code-block:: python

    # WRONG: Don't duplicate data
    data_wrong = np.repeat(values, counts)  # Memory waste
    
    # RIGHT: Use counts as weights
    params = WeightedFitting.fit_weighted_mle(values, counts, dist)

**Pitfall 2: Forgetting to normalize**

.. code-block:: python

    # If you manually calculate, normalize first
    weights_norm = weights / np.sum(weights)

**Pitfall 3: Ignoring effective sample size**

.. code-block:: python

    # Very unequal weights reduce power
    ess = WeightedFitting.effective_sample_size(weights)
    if ess < 0.5 * len(weights):
        print("⚠️  Weights are very unequal")

Next Steps
----------

- :doc:`08_visualization` - Visualize weighted fits
- :doc:`09_advanced` - Advanced weighted techniques
