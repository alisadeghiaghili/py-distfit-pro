Tutorial 7: Weighted Data
=========================

Learn how to fit distributions to weighted observations.

When to Use Weights
-------------------

**Weights represent:**

1. **Sampling weights** - survey data
2. **Frequency counts** - aggregated data  
3. **Precision weights** - measurement reliability
4. **Importance weights** - different observation importance

**Example scenarios:**

- Survey with stratified sampling
- Grouped data with frequencies
- Measurements with different accuracies
- Expert judgments with confidence scores

Weighted MLE
------------

**Maximum Likelihood with observation weights.**

**Maximizes:** Σ wᵢ log f(xᵢ; θ)

**Code:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.weighted import WeightedFitting
    import numpy as np
    
    # Data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Weights (e.g., sampling weights)
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    # Weighted fit
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    
    # Set parameters
    dist.params = params
    dist.fitted = True
    
    print(dist.summary())

**Compare with unweighted:**

.. code-block:: python

    # Unweighted fit
    dist_unweighted = get_distribution('normal')
    dist_unweighted.fit(data)
    
    # Weighted fit
    params_weighted = WeightedFitting.fit_weighted_mle(data, weights, dist)
    
    print("Unweighted:", dist_unweighted.params)
    print("Weighted:", params_weighted)
    print("Difference:", {k: abs(dist_unweighted.params[k] - params_weighted[k]) 
                         for k in params_weighted})

Weighted Method of Moments
---------------------------

**Faster alternative to weighted MLE.**

**Matches weighted moments:**

- Weighted mean: μₑ = Σ wᵢxᵢ / Σ wᵢ
- Weighted variance: σₑ² = Σ wᵢ(xᵢ - μₑ)² / Σ wᵢ

**Code:**

.. code-block:: python

    # Weighted moments
    params_mom = WeightedFitting.fit_weighted_moments(data, weights, dist)
    
    dist.params = params_mom
    dist.fitted = True
    
    print("Weighted MLE:", params_weighted)
    print("Weighted Moments:", params_mom)

**When to use:**

- Quick estimates
- MLE fails to converge
- Simple distributions (Normal, Exponential)

Survey Data Example
-------------------

**Scenario:** National income survey with stratified sampling.

.. code-block:: python

    # Survey responses (annual income in $1000s)
    incomes = np.array([
        45, 52, 38, 67, 89, 120, 56, 73, 44, 91,
        105, 62, 55, 88, 71, 49, 95, 110, 63, 77
    ])
    
    # Sampling weights (inverse probability of selection)
    # Higher weight = underrepresented group
    sampling_weights = np.array([
        2.5, 1.8, 2.2, 1.5, 1.2, 0.8, 1.9, 1.3, 2.3, 1.1,
        1.0, 1.7, 1.8, 1.2, 1.4, 2.1, 1.1, 0.9, 1.6, 1.3
    ])
    
    # Fit lognormal (typical for income)
    dist = get_distribution('lognormal')
    params = WeightedFitting.fit_weighted_mle(incomes, sampling_weights, dist)
    dist.params = params
    dist.fitted = True
    
    # Population estimates
    print(f"Estimated median income: ${dist.median():.1f}k")
    print(f"Estimated mean income: ${dist.mean():.1f}k")
    print(f"80th percentile: ${dist.ppf(0.80):.1f}k")

**Compare weighted vs unweighted:**

.. code-block:: python

    # Unweighted (biased!)
    dist_naive = get_distribution('lognormal')
    dist_naive.fit(incomes)
    
    print("\nNaive median:", dist_naive.median())
    print("Weighted median:", dist.median())
    print(f"Bias: {dist_naive.median() - dist.median():.2f}k")

Frequency Data Example
----------------------

**Scenario:** Aggregated data with counts.

.. code-block:: python

    # Grouped data
    values = np.array([1, 2, 3, 4, 5, 6])  # Values
    frequencies = np.array([15, 28, 35, 18, 8, 3])  # Counts
    
    # Expand to individual observations (inefficient)
    data_expanded = np.repeat(values, frequencies)
    
    # Better: use weights
    dist = get_distribution('poisson')
    params = WeightedFitting.fit_weighted_mle(values, frequencies, dist)
    dist.params = params
    dist.fitted = True
    
    print(f"Estimated λ: {dist.params['mu']:.3f}")
    
    # Verify
    print(f"Weighted mean: {np.average(values, weights=frequencies):.3f}")

Precision Weights Example
-------------------------

**Scenario:** Measurements with different accuracies.

.. code-block:: python

    # Measurements of same quantity
    measurements = np.array([10.2, 9.8, 10.5, 10.1, 9.9, 10.3])
    
    # Standard errors for each measurement
    std_errors = np.array([0.5, 0.3, 0.8, 0.4, 0.2, 0.6])
    
    # Precision weights (inverse variance)
    precision_weights = 1 / std_errors**2
    
    # Weighted fit
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(
        measurements,
        precision_weights,
        dist
    )
    
    print(f"Weighted estimate: {params['loc']:.3f} ± {params['scale']:.3f}")
    
    # More precise measurements get more weight
    print("\nWeights:")
    for i, (m, w) in enumerate(zip(measurements, precision_weights)):
        print(f"  {m:.1f} (weight={w:.1f})")

Weighted Statistics
-------------------

**Compute weighted descriptive statistics.**

.. code-block:: python

    # Weighted stats
    stats = WeightedFitting.weighted_stats(data, weights)
    
    print("Weighted Statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Variance: {stats['var']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Q25: {stats['q25']:.4f}")
    print(f"  Q75: {stats['q75']:.4f}")

**Weighted quantiles:**

.. code-block:: python

    # Specific quantiles
    q50 = WeightedFitting.weighted_quantile(data, weights, 0.50)
    q95 = WeightedFitting.weighted_quantile(data, weights, 0.95)
    
    print(f"50th percentile: {q50:.4f}")
    print(f"95th percentile: {q95:.4f}")

Effective Sample Size
---------------------

**How much "information" do weighted samples provide?**

**Formula:** ESS = (Σ wᵢ)² / Σ wᵢ²

.. code-block:: python

    # Effective sample size
    ess = WeightedFitting.effective_sample_size(weights)
    
    print(f"Actual n: {len(weights)}")
    print(f"Effective n: {ess:.1f}")
    print(f"Efficiency: {ess/len(weights)*100:.1f}%")

**Interpretation:**

- ESS = n: all weights equal (100% efficient)
- ESS < n: unequal weights (less efficient)
- ESS << n: very unequal weights (much less efficient)

**Example:**

.. code-block:: python

    # Equal weights
    equal_w = np.ones(1000)
    ess_equal = WeightedFitting.effective_sample_size(equal_w)
    print(f"Equal weights: ESS = {ess_equal:.0f} (100%)")
    
    # Moderate variation
    moderate_w = np.random.uniform(0.5, 1.5, 1000)
    ess_mod = WeightedFitting.effective_sample_size(moderate_w)
    print(f"Moderate: ESS = {ess_mod:.0f} ({ess_mod/1000*100:.0f}%)")
    
    # High variation
    high_w = np.random.uniform(0.1, 10, 1000)
    ess_high = WeightedFitting.effective_sample_size(high_w)
    print(f"High variation: ESS = {ess_high:.0f} ({ess_high/1000*100:.0f}%)")

Weighted Bootstrap
------------------

**Uncertainty quantification for weighted estimates.**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    # Weighted fit
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    dist.params = params
    dist.fitted = True
    
    # Bootstrap with weights
    # (resample data AND weights together)
    n = len(data)
    boot_params = []
    
    for i in range(1000):
        # Resample indices
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = data[idx]
        boot_weights = weights[idx]
        
        # Refit
        boot_dist = get_distribution('normal')
        boot_p = WeightedFitting.fit_weighted_mle(
            boot_data, boot_weights, boot_dist
        )
        boot_params.append(boot_p['loc'])
    
    # CI
    ci_lower = np.percentile(boot_params, 2.5)
    ci_upper = np.percentile(boot_params, 97.5)
    
    print(f"Mean: {params['loc']:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

Supported Distributions
-----------------------

**Weighted fitting works for:**

**Continuous:**
- Normal, Lognormal
- Exponential, Gamma, Weibull
- Beta, Uniform
- Logistic, Laplace

**Discrete:**
- Poisson, Binomial
- Negative Binomial, Geometric

.. code-block:: python

    # Try different distributions
    for dist_name in ['normal', 'lognormal', 'gamma', 'weibull']:
        dist = get_distribution(dist_name)
        try:
            params = WeightedFitting.fit_weighted_mle(data, weights, dist)
            print(f"{dist_name}: success")
        except:
            print(f"{dist_name}: failed")

Best Practices
--------------

1. **Normalize weights**
   
   Weights are automatically normalized, but you can do it manually:
   
   .. code-block:: python
   
       weights_normalized = weights / np.sum(weights)

2. **Check effective sample size**
   
   .. code-block:: python
   
       ess = WeightedFitting.effective_sample_size(weights)
       if ess < 30:
           print("⚠️ Warning: Very low ESS, results may be unstable")

3. **Handle zero weights**
   
   Automatically removed, but check:
   
   .. code-block:: python
   
       n_zero = np.sum(weights == 0)
       if n_zero > 0:
           print(f"{n_zero} observations have zero weight")

4. **Validate weights**
   
   .. code-block:: python
   
       assert np.all(weights >= 0), "Weights must be non-negative"
       assert len(data) == len(weights), "Data and weights must match"

5. **Compare with unweighted**
   
   Always check if weights make a difference:
   
   .. code-block:: python
   
       dist_weighted = ...
       dist_unweighted = ...
       
       diff = abs(dist_weighted.mean() - dist_unweighted.mean())
       if diff / dist_unweighted.mean() > 0.1:
           print("Weights make >10% difference!")

Next Steps
----------

- :doc:`08_visualization` - Plot weighted fits
- :doc:`09_advanced` - Advanced weighting techniques
- :doc:`../examples/real_world` - Real-world weighted examples
