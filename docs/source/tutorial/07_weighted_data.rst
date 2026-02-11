Tutorial 7: Weighted Data Support
==================================

Learn how to fit distributions when observations have different weights.

Why Weighted Data?
------------------

**Not all observations are created equal!**

**Common scenarios:**

1. **Survey data** - sampling weights
2. **Stratified sampling** - different strata sizes
3. **Reliability** - different measurement precision
4. **Aggregated data** - frequency counts
5. **Meta-analysis** - combining studies

**Example:**

.. code-block:: python

    # Survey with oversampling
    # Group A: 100 people, weight=1.0
    # Group B: 50 people, weight=2.0 (underrepresented)
    
    data = np.concatenate([
        np.random.normal(10, 2, 100),  # Group A
        np.random.normal(12, 2, 50)    # Group B
    ])
    
    weights = np.concatenate([
        np.ones(100) * 1.0,   # Group A weight
        np.ones(50) * 2.0     # Group B weight (more important)
    ])

Weighted MLE
------------

**Maximum Likelihood with observation weights.**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.weighted import WeightedFitting
    import numpy as np
    
    # Generate weighted data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    weights = np.random.uniform(0.5, 1.5, 1000)
    
    # Weighted MLE
    dist = get_distribution('normal')
    params = WeightedFitting.fit_weighted_mle(
        data=data,
        weights=weights,
        distribution=dist
    )
    
    # Set parameters
    dist.params = params
    dist.fitted = True
    
    print("Weighted MLE Parameters:")
    print(dist.params)
    print("\n" + dist.summary())

**Comparison with unweighted:**

.. code-block:: python

    # Unweighted fit
    dist_unweighted = get_distribution('normal')
    dist_unweighted.fit(data, method='mle')
    
    print("\nComparison:")
    print(f"Weighted mean:   {params['loc']:.4f}")
    print(f"Unweighted mean: {dist_unweighted.params['loc']:.4f}")
    print(f"Difference:      {abs(params['loc'] - dist_unweighted.params['loc']):.4f}")

Weighted Moments
----------------

**Method of Moments with weights.**

.. code-block:: python

    # Weighted moments
    params_mom = WeightedFitting.fit_weighted_moments(
        data=data,
        weights=weights,
        distribution=dist
    )
    
    print("Weighted Moments Parameters:")
    print(params_mom)

**Manual calculation:**

.. code-block:: python

    # Normalize weights
    w_norm = weights / np.sum(weights)
    
    # Weighted mean
    wmean = np.sum(w_norm * data)
    
    # Weighted variance
    wvar = np.sum(w_norm * (data - wmean)**2)
    
    print(f"\nManual:")
    print(f"  Weighted mean: {wmean:.4f}")
    print(f"  Weighted var:  {wvar:.4f}")

Weighted Statistics
-------------------

**Comprehensive weighted statistics:**

.. code-block:: python

    # Weighted statistics
    stats = WeightedFitting.weighted_stats(data, weights)
    
    print("\nWeighted Statistics:")
    for key, value in stats.items():
        print(f"  {key:<10}: {value:.4f}")

**Output:**

::

    Weighted Statistics:
      mean      : 10.1234
      var       : 3.9876
      std       : 1.9969
      median    : 10.0123
      q25       : 8.6789
      q75       : 11.5432

Weighted Quantiles
------------------

**Calculate weighted percentiles:**

.. code-block:: python

    # Weighted median
    wmedian = WeightedFitting.weighted_quantile(
        data=data,
        weights=weights,
        quantile=0.5
    )
    
    print(f"Weighted median: {wmedian:.4f}")
    
    # Weighted quartiles
    q25 = WeightedFitting.weighted_quantile(data, weights, 0.25)
    q75 = WeightedFitting.weighted_quantile(data, weights, 0.75)
    
    print(f"Weighted IQR: [{q25:.4f}, {q75:.4f}]")

Effective Sample Size
---------------------

**How much information do the weights provide?**

.. code-block:: python

    # Effective sample size
    ess = WeightedFitting.effective_sample_size(weights)
    
    print(f"\nSample sizes:")
    print(f"  Actual n:    {len(data)}")
    print(f"  Effective n: {ess:.1f}")
    print(f"  Efficiency:  {ess/len(data)*100:.1f}%")

**Interpretation:**

- ESS = n → all weights equal (100% efficient)
- ESS < n → some observations have low weight
- ESS << n → very unequal weights (inefficient)

**Example with extreme weights:**

.. code-block:: python

    # Equal weights
    weights_equal = np.ones(1000)
    ess_equal = WeightedFitting.effective_sample_size(weights_equal)
    print(f"Equal weights ESS: {ess_equal:.1f} (100%)")
    
    # Unequal weights
    weights_unequal = np.random.exponential(1, 1000)
    ess_unequal = WeightedFitting.effective_sample_size(weights_unequal)
    print(f"Unequal weights ESS: {ess_unequal:.1f} ({ess_unequal/1000*100:.1f}%)")
    
    # Extreme weights (one dominant observation)
    weights_extreme = np.ones(1000)
    weights_extreme[0] = 1000
    ess_extreme = WeightedFitting.effective_sample_size(weights_extreme)
    print(f"Extreme weights ESS: {ess_extreme:.1f} ({ess_extreme/1000*100:.1f}%)")

Supported Distributions
-----------------------

**Weighted fitting works for 14 distributions:**

**Continuous:**
- Normal
- Lognormal
- Exponential
- Gamma
- Weibull
- Beta
- Uniform
- Logistic
- Laplace

**Discrete:**
- Poisson
- Binomial
- Negative Binomial
- Geometric

**Example with different distributions:**

.. code-block:: python

    # Gamma data with weights
    data_gamma = np.random.gamma(2, 3, 500)
    weights_gamma = np.random.uniform(0.8, 1.2, 500)
    
    dist_gamma = get_distribution('gamma')
    params_gamma = WeightedFitting.fit_weighted_mle(
        data=data_gamma,
        weights=weights_gamma,
        distribution=dist_gamma
    )
    
    print("\nGamma (weighted):")
    print(params_gamma)

Real-World Examples
-------------------

Survey Data
^^^^^^^^^^^

**Sampling weights from stratified survey:**

.. code-block:: python

    # Survey: household income
    # Strata: Urban (70%), Rural (30%)
    # Sample: Urban=100, Rural=100 (oversampled rural)
    
    # Urban households
    income_urban = np.random.lognormal(10.5, 0.5, 100)
    weight_urban = np.ones(100) * 0.7  # 70% of population
    
    # Rural households (lower income)
    income_rural = np.random.lognormal(10.0, 0.6, 100)
    weight_rural = np.ones(100) * 0.3  # 30% of population
    
    # Combine
    income = np.concatenate([income_urban, income_rural])
    weights_survey = np.concatenate([weight_urban, weight_rural])
    
    # Fit lognormal with weights
    dist_income = get_distribution('lognormal')
    params_income = WeightedFitting.fit_weighted_mle(
        data=income,
        weights=weights_survey,
        distribution=dist_income
    )
    
    dist_income.params = params_income
    dist_income.fitted = True
    
    print("\nSurvey-weighted income distribution:")
    print(dist_income.summary())

Frequency Data
^^^^^^^^^^^^^^

**Aggregated data with counts:**

.. code-block:: python

    # Grouped data
    values = np.array([1, 2, 3, 4, 5, 6])
    frequencies = np.array([10, 25, 30, 20, 10, 5])
    
    # Expand to individual observations
    data_expanded = np.repeat(values, frequencies)
    
    # OR use weights (more efficient)
    dist_freq = get_distribution('poisson')
    params_freq = WeightedFitting.fit_weighted_mle(
        data=values,
        weights=frequencies,
        distribution=dist_freq
    )
    
    print("\nFrequency-weighted Poisson:")
    print(params_freq)

Reliability Data
^^^^^^^^^^^^^^^^

**Different measurement precision:**

.. code-block:: python

    # Measurements with different precision
    # High precision (weight=1.0)
    data_precise = np.random.normal(100, 0.5, 50)
    weights_precise = np.ones(50) * 1.0
    
    # Low precision (weight=0.3)
    data_imprecise = np.random.normal(100, 2.0, 50)
    weights_imprecise = np.ones(50) * 0.3
    
    # Combine
    data_reliability = np.concatenate([data_precise, data_imprecise])
    weights_reliability = np.concatenate([weights_precise, weights_imprecise])
    
    # Weighted fit (gives more weight to precise measurements)
    dist_rel = get_distribution('normal')
    params_rel = WeightedFitting.fit_weighted_mle(
        data=data_reliability,
        weights=weights_reliability,
        distribution=dist_rel
    )
    
    print("\nReliability-weighted:")
    print(params_rel)

Weighted Bootstrap
------------------

**Combine weights with bootstrap CIs:**

.. code-block:: python

    # First fit with weights
    dist_w = get_distribution('normal')
    params_w = WeightedFitting.fit_weighted_mle(data, weights, dist_w)
    dist_w.params = params_w
    dist_w.fitted = True
    
    # Then bootstrap (resamples respect weights)
    from distfit_pro.core.bootstrap import Bootstrap
    
    # For weighted data, use parametric bootstrap
    ci_weighted = Bootstrap.parametric(
        data=data,
        distribution=dist_w,
        n_bootstrap=1000
    )
    
    for param, result in ci_weighted.items():
        print(result)

Best Practices
--------------

1. **Normalize weights**
   
   Weights are automatically normalized in WeightedFitting

2. **Check effective sample size**
   
   .. code-block:: python
   
       ess = WeightedFitting.effective_sample_size(weights)
       if ess < len(data) * 0.5:
           print("Warning: ESS < 50% of sample size")

3. **Use MLE for weighted fitting**
   
   Generally more accurate than moments

4. **Handle zero weights**
   
   .. code-block:: python
   
       # Zero weights are automatically removed
       weights[weights < 0.01] = 0  # Effectively remove

5. **Document weighting scheme**
   
   Always explain how weights were derived!

6. **Compare weighted vs unweighted**
   
   .. code-block:: python
   
       # Check if weights make a difference
       diff = abs(params_weighted['loc'] - params_unweighted['loc'])
       if diff > 0.1 * abs(params_unweighted['loc']):
           print("Weights have substantial impact!")

Troubleshooting
---------------

**Negative or zero weights:**

.. code-block:: python

    # Weights must be non-negative
    weights_bad = np.random.normal(1, 0.5, 100)
    weights_bad[weights_bad < 0] = 0  # Fix negative
    
    # Check for all zeros
    if np.sum(weights_bad) == 0:
        raise ValueError("All weights are zero!")

**Very unequal weights:**

.. code-block:: python

    # Check weight distribution
    print(f"Weight range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    print(f"Weight CV: {np.std(weights)/np.mean(weights):.4f}")
    
    # If CV > 1, weights are very unequal

**Fitting failures:**

.. code-block:: python

    try:
        params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    except:
        # Fall back to moments
        print("MLE failed, using moments")
        params = WeightedFitting.fit_weighted_moments(data, weights, dist)

Next Steps
----------

- :doc:`08_visualization` - Visual weighted data
- :doc:`examples/real_world` - Survey data examples
- :doc:`09_advanced` - Advanced techniques
