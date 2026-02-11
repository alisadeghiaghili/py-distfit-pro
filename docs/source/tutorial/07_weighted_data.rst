Tutorial 7: Weighted Data
=========================

Learn to fit distributions to weighted observations.

When to Use Weighted Fitting
-----------------------------

**Weighted fitting is essential when:**

1. **Observations have different reliabilities**
   
   Example: Measurements from different instruments

2. **Stratified sampling**
   
   Example: Survey data with unequal sampling rates

3. **Frequency data**
   
   Example: Aggregated counts or grouped data

4. **Heteroscedastic errors**
   
   Example: Variance differs across observations

Weighted MLE
------------

**Maximum Likelihood with observation weights:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.weighted import WeightedFitting
    import numpy as np
    
    # Generate data with different reliabilities
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Higher weights = more reliable observations
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    # Fit distribution
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    
    dist.params = params
    dist.fitted = True
    
    print("Weighted MLE Parameters:")
    print(dist.params)

**Compare with unweighted:**

.. code-block:: python

    # Unweighted fit
    dist_unweighted = get_distribution('normal')
    dist_unweighted.fit(data, method='mle')
    
    print("\nComparison:")
    print(f"Weighted μ: {dist.params['loc']:.4f}")
    print(f"Unweighted μ: {dist_unweighted.params['loc']:.4f}")
    print(f"\nWeighted σ: {dist.params['scale']:.4f}")
    print(f"Unweighted σ: {dist_unweighted.params['scale']:.4f}")

Weighted Method of Moments
---------------------------

**Faster alternative to weighted MLE:**

.. code-block:: python

    # Weighted moments
    params_mom = WeightedFitting.fit_weighted_moments(data, weights, dist)
    
    print("Weighted Moments Parameters:")
    print(params_mom)

**How it works:**

.. code-block:: python

    # Manually calculate weighted mean and std
    w_normalized = weights / np.sum(weights)
    w_mean = np.sum(w_normalized * data)
    w_var = np.sum(w_normalized * (data - w_mean)**2)
    w_std = np.sqrt(w_var)
    
    print(f"Manual weighted mean: {w_mean:.4f}")
    print(f"Manual weighted std: {w_std:.4f}")

Weighted Statistics
-------------------

**Compute comprehensive weighted statistics:**

.. code-block:: python

    # Get weighted statistics
    stats = WeightedFitting.weighted_stats(data, weights)
    
    print("Weighted Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

**Output:**

::

    Weighted Statistics:
      mean: 10.0234
      var: 3.9876
      std: 1.9969
      median: 10.0123
      q25: 8.6543
      q75: 11.3987

Effective Sample Size
---------------------

**Weighted data has lower effective sample size:**

.. code-block:: python

    # Calculate ESS
    ess = WeightedFitting.effective_sample_size(weights)
    
    print(f"Actual sample size: {len(data)}")
    print(f"Effective sample size: {ess:.1f}")
    print(f"Efficiency: {ess/len(data)*100:.1f}%")

**Output:**

::

    Actual sample size: 1000
    Effective sample size: 923.4
    Efficiency: 92.3%

**Interpretation:**

- ESS = n: uniform weights (no information loss)
- ESS < n: non-uniform weights (some information loss)
- ESS << n: very unequal weights (large information loss)

Real Example 1: Survey Data
----------------------------

**Survey with stratified sampling:**

.. code-block:: python

    # Income data ($1000s) from survey
    # Different sampling rates by region
    
    incomes = np.array([
        45, 52, 38, 67, 55, 48, 72, 61, 44, 58,  # Urban (10% sampled)
        35, 42, 38, 41, 39, 36, 44, 40, 37, 43,  # Rural (50% sampled)
    ])
    
    # Population weights (inverse of sampling rate)
    sampling_weights = np.array(
        [10] * 10 +  # Urban: 10x weight (10% sample rate)
        [2] * 10     # Rural: 2x weight (50% sample rate)
    )
    
    # Fit lognormal distribution
    dist = get_distribution('lognormal')
    params = WeightedFitting.fit_weighted_mle(
        incomes, 
        sampling_weights, 
        dist
    )
    
    dist.params = params
    dist.fitted = True
    
    print("Population income distribution:")
    print(dist.summary())
    
    # Estimated population statistics
    pop_stats = WeightedFitting.weighted_stats(incomes, sampling_weights)
    print(f"\nPopulation mean income: ${pop_stats['mean']:.1f}K")
    print(f"Population median income: ${pop_stats['median']:.1f}K")

Real Example 2: Grouped Data
-----------------------------

**Fitting to frequency table:**

.. code-block:: python

    # Age groups with frequencies
    age_midpoints = np.array([22, 27, 32, 37, 42, 47, 52, 57, 62])
    frequencies = np.array([150, 230, 280, 320, 290, 240, 180, 120, 90])
    
    # Expand data (or use frequencies as weights)
    # Using weights is more efficient:
    
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(
        age_midpoints,
        frequencies,
        dist
    )
    
    dist.params = params
    dist.fitted = True
    
    print("Age distribution:")
    print(f"Mean age: {dist.mean():.1f} years")
    print(f"Std: {dist.std():.1f} years")

Real Example 3: Measurement Error
----------------------------------

**Data from instruments with different precision:**

.. code-block:: python

    # Temperature measurements
    # Instrument A: precision = 0.1°C
    # Instrument B: precision = 0.5°C
    
    temps_A = np.random.normal(25.0, 0.1, 50)
    temps_B = np.random.normal(25.0, 0.5, 50)
    
    # Combine data
    all_temps = np.concatenate([temps_A, temps_B])
    
    # Weights inversely proportional to variance
    # w ∝ 1/σ²
    weights_A = np.ones(50) / (0.1**2)  # High weight (precise)
    weights_B = np.ones(50) / (0.5**2)  # Low weight (imprecise)
    all_weights = np.concatenate([weights_A, weights_B])
    
    # Fit
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(
        all_temps,
        all_weights,
        dist
    )
    
    dist.params = params
    dist.fitted = True
    
    print("True temperature estimate:")
    print(f"Temperature: {dist.params['loc']:.3f}°C")
    print(f"Uncertainty: {dist.params['scale']:.3f}°C")

Weighted Quantiles
------------------

**Calculate weighted percentiles:**

.. code-block:: python

    # Weighted median
    median = WeightedFitting.weighted_quantile(data, weights, 0.5)
    print(f"Weighted median: {median:.4f}")
    
    # Weighted quartiles
    q25 = WeightedFitting.weighted_quantile(data, weights, 0.25)
    q75 = WeightedFitting.weighted_quantile(data, weights, 0.75)
    print(f"Weighted IQR: [{q25:.4f}, {q75:.4f}]")
    
    # Weighted 90% range
    q05 = WeightedFitting.weighted_quantile(data, weights, 0.05)
    q95 = WeightedFitting.weighted_quantile(data, weights, 0.95)
    print(f"Weighted 90% range: [{q05:.4f}, {q95:.4f}]")

Supported Distributions
-----------------------

**Weighted fitting works with:**

**Continuous:**
- Normal, Lognormal
- Exponential, Gamma, Weibull
- Beta, Uniform, Logistic, Laplace

**Discrete:**
- Poisson, Binomial
- Negative Binomial, Geometric

.. code-block:: python

    # Example: Weighted Gamma
    data_gamma = np.random.gamma(2, 3, 500)
    weights_gamma = np.random.uniform(0.8, 1.2, 500)
    
    dist_gamma = get_distribution('gamma')
    params = WeightedFitting.fit_weighted_mle(
        data_gamma,
        weights_gamma,
        dist_gamma
    )
    
    dist_gamma.params = params
    dist_gamma.fitted = True
    
    print("Weighted Gamma fit:")
    print(dist_gamma.params)

Weighted Bootstrap
------------------

**Combine weighted fitting with bootstrap:**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    # First fit with weights
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    dist.params = params
    dist.fitted = True
    
    # Then bootstrap (will resample with weights)
    # Note: Standard bootstrap doesn't account for weights
    # For proper weighted bootstrap, resample indices with probability ∝ weights
    
    # Weighted resampling
    n_boot = 1000
    boot_params = []
    
    for i in range(n_boot):
        # Sample with replacement, probability ∝ weights
        probs = weights / np.sum(weights)
        indices = np.random.choice(
            len(data), 
            size=len(data), 
            replace=True,
            p=probs
        )
        
        boot_data = data[indices]
        boot_weights = weights[indices]
        
        # Fit to bootstrap sample
        boot_dist = get_distribution('normal')
        boot_params_i = WeightedFitting.fit_weighted_mle(
            boot_data,
            boot_weights,
            boot_dist
        )
        
        boot_params.append(boot_params_i['loc'])
    
    # Calculate CI
    ci_lower = np.percentile(boot_params, 2.5)
    ci_upper = np.percentile(boot_params, 97.5)
    
    print(f"\nWeighted Bootstrap 95% CI for μ:")
    print(f"[{ci_lower:.4f}, {ci_upper:.4f}]")

Best Practices
--------------

1. **Normalize weights**
   
   WeightedFitting automatically normalizes, but be aware:
   
   .. code-block:: python
   
       # These give same results:
       weights1 = np.array([1, 2, 3])
       weights2 = np.array([10, 20, 30])
       weights3 = np.array([0.1, 0.2, 0.3])

2. **Check effective sample size**
   
   .. code-block:: python
   
       ess = WeightedFitting.effective_sample_size(weights)
       if ess < 0.5 * len(weights):
           print("⚠️  Large information loss from weighting!")

3. **Validate weights**
   
   .. code-block:: python
   
       # Check for invalid weights
       if np.any(weights < 0):
           raise ValueError("Weights must be non-negative")
       
       if np.sum(weights) == 0:
           raise ValueError("Sum of weights is zero")

4. **Use moments for initial estimates**
   
   Weighted MLE can fail - use moments as fallback:
   
   .. code-block:: python
   
       try:
           params = WeightedFitting.fit_weighted_mle(data, weights, dist)
       except:
           print("MLE failed, using moments")
           params = WeightedFitting.fit_weighted_moments(data, weights, dist)

5. **Document weight source**
   
   Always document where weights come from!

Common Pitfalls
---------------

**Pitfall 1: Ignoring weights**

.. code-block:: python

    # WRONG: Fit without weights
    dist.fit(data)  # Ignores that observations have different reliability
    
    # CORRECT: Use weights
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)

**Pitfall 2: Equal weights**

.. code-block:: python

    # If all weights are equal, just use unweighted!
    weights = np.ones(len(data))
    
    # This is inefficient - use regular fit instead
    dist.fit(data, method='mle')

**Pitfall 3: Extreme weights**

.. code-block:: python

    # Very unequal weights cause problems
    weights = np.array([1, 1, 1, 1000000])  # Bad!
    
    # Check weight ratio
    max_ratio = np.max(weights) / np.min(weights[weights > 0])
    if max_ratio > 1000:
        print("⚠️  Extreme weight ratios detected!")

Next Steps
----------

- :doc:`08_visualization` - Visualize weighted fits
- :doc:`09_advanced` - Advanced weighted techniques
