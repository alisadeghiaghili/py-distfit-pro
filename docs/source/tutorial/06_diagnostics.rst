Tutorial 6: Diagnostics & Outlier Detection
============================================

Learn to diagnose problems and detect outliers.

Why Diagnostics?
----------------

GOF tests tell you IF there's a problem.
Diagnostics tell you WHAT and WHERE the problem is.

**Use diagnostics to:**

- Identify systematic deviations
- Find influential observations
- Detect outliers
- Validate model assumptions
- Improve fit quality

Residual Analysis
-----------------

**Residuals = deviations from fitted model.**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.diagnostics import Diagnostics
    import numpy as np
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Residual analysis
    residuals = Diagnostics.residual_analysis(data, dist)
    print(residuals.summary())

**Output:**

::

    Residual Analysis Summary
    ==================================================
    Quantile Residuals:
      Mean: 0.001234
      Std: 0.998765
      Range: [-3.456, 3.234]
    
    Pearson Residuals:
      Mean: 0.000987
      Std: 1.002345
      
    Deviance Residuals:
      Mean: 0.001456
      Std: 0.997654

**Types of Residuals:**

1. **Quantile Residuals**
   
   - Transform data to standard normal
   - Should be N(0,1) if fit is good
   - Best for GOF checking

2. **Pearson Residuals**
   
   - Standardized deviations from mean
   - Easy to interpret
   - Good for detecting outliers

3. **Deviance Residuals**
   
   - Based on likelihood
   - Useful for model comparison

4. **Standardized Residuals**
   
   - Simple z-scores
   - Quick outlier detection

Influence Diagnostics
---------------------

**Identify observations with large influence.**

.. code-block:: python

    # Influence diagnostics
    influence = Diagnostics.influence_diagnostics(data, dist)
    print(influence.summary())

**Output:**

::

    Influence Diagnostics Summary
    ==================================================
    Cook's Distance:
      Max: 0.012345
      Threshold: 0.004000
      Influential: 3 observations
    
    Influential Indices: [234, 567, 892]

**Cook's Distance:**

Measures how much parameters change if observation is removed.

.. code-block:: python

    # Get Cook's distance for all points
    cooks_d = influence.cooks_distance
    
    # Find influential points (threshold = 4/n)
    threshold = 4 / len(data)
    influential = np.where(cooks_d > threshold)[0]
    
    print(f"Influential observations: {len(influential)}")
    print(f"Indices: {influential[:10]}...")  # First 10

**DFFITS:**

.. code-block:: python

    # DFFITS values
    dffits = influence.dffits
    
    # Large |DFFITS| indicates high influence
    high_influence = np.where(np.abs(dffits) > 2 * np.sqrt(len(dist.params)/len(data)))[0]
    
    print(f"High influence points: {len(high_influence)}")

Outlier Detection
-----------------

Z-Score Method
^^^^^^^^^^^^^^

**Most common method.**

.. code-block:: python

    # Z-score outlier detection
    outliers = Diagnostics.detect_outliers(
        data, 
        dist, 
        method='zscore',
        threshold=3.0
    )
    
    print(outliers.summary())

**Output:**

::

    Outlier Detection Summary
    ==================================================
    Method: zscore
    Threshold: 3.000000
    Outliers Detected: 7
    
    Outlier Indices: [123, 456, 789, ...]
    
    Score Range: [0.012, 3.567]

**How it works:**

.. code-block:: python

    # Z-score = |x - μ| / σ
    mean = dist.mean()
    std = dist.std()
    z_scores = np.abs((data - mean) / std)
    
    # Outliers: |z| > 3
    outlier_mask = z_scores > 3.0

IQR Method
^^^^^^^^^^

**Interquartile range method.**

.. code-block:: python

    # IQR method
    outliers = Diagnostics.detect_outliers(
        data,
        dist,
        method='iqr',
        threshold=1.5  # Standard IQR multiplier
    )
    
    print(outliers.summary())

**How it works:**

.. code-block:: python

    # Get quartiles from fitted distribution
    q1 = dist.ppf(0.25)
    q3 = dist.ppf(0.75)
    iqr = q3 - q1
    
    # Outlier bounds
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    # Outliers outside bounds
    outliers = (data < lower) | (data > upper)

Likelihood Method
^^^^^^^^^^^^^^^^^

**Based on low probability.**

.. code-block:: python

    # Likelihood method
    outliers = Diagnostics.detect_outliers(
        data,
        dist,
        method='likelihood',
        threshold=None  # Auto: 1st percentile
    )
    
    print(outliers.summary())

**How it works:**

- Calculate log-likelihood for each point
- Points with very low likelihood are outliers
- Threshold = 1st percentile by default

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^

**Multivariate outlier detection (univariate case).**

.. code-block:: python

    # Mahalanobis distance
    outliers = Diagnostics.detect_outliers(
        data,
        dist,
        method='mahalanobis',
        threshold=None  # Auto: chi-square 99th percentile
    )

Comparing Methods
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Compare all methods
    methods = ['zscore', 'iqr', 'likelihood', 'mahalanobis']
    
    print("Outlier Detection Comparison")
    print(f"{'Method':<15} {'Outliers':<10} {'Percentage':<12}")
    print("-" * 40)
    
    for method in methods:
        outliers = Diagnostics.detect_outliers(data, dist, method=method)
        n_outliers = len(outliers.outlier_indices)
        pct = 100 * n_outliers / len(data)
        print(f"{method:<15} {n_outliers:<10} {pct:<12.2f}%")

Q-Q Plot Diagnostics
--------------------

**Quantile-Quantile plot data.**

.. code-block:: python

    # Get Q-Q plot data
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    print(f"Q-Q Correlation: {qq_data['correlation']:.4f}")
    print(f"Max residual: {np.max(np.abs(qq_data['residuals'])):.4f}")
    
    # Perfect fit: correlation ≈ 1.0
    if qq_data['correlation'] > 0.99:
        print("Excellent fit!")
    elif qq_data['correlation'] > 0.95:
        print("Good fit")
    else:
        print("Poor fit")

**Interpret Q-Q residuals:**

.. code-block:: python

    residuals = qq_data['residuals']
    
    # Systematic patterns indicate problems
    # S-shape → tails too heavy/light
    # U-shape → distribution wrong
    # Random scatter → good fit

P-P Plot Diagnostics
--------------------

**Probability-Probability plot data.**

.. code-block:: python

    # Get P-P plot data
    pp_data = Diagnostics.pp_diagnostics(data, dist)
    
    print(f"Max deviation: {pp_data['max_deviation']:.6f}")
    
    # Similar to KS statistic
    if pp_data['max_deviation'] < 0.05:
        print("Good fit")
    else:
        print("Check for problems")

Worm Plot
---------

**Detrended Q-Q plot - easier to spot deviations.**

.. code-block:: python

    # Worm plot data
    worm_data = Diagnostics.worm_plot_data(data, dist)
    
    # Should fluctuate around 0
    worm_residuals = worm_data['worm_residuals']
    
    print(f"Worm residual range: [{np.min(worm_residuals):.3f}, {np.max(worm_residuals):.3f}]")
    print(f"Worm residual std: {np.std(worm_residuals):.3f}")
    
    # Good fit: std ≈ 1, no systematic patterns

Practical Example: Quality Control
-----------------------------------

.. code-block:: python

    # Manufacturing measurements
    np.random.seed(42)
    
    # Normal process
    normal_process = np.random.normal(100, 2, 950)
    
    # Defective items (outliers)
    defects = np.random.uniform(90, 95, 50)
    
    # Combined data
    measurements = np.concatenate([normal_process, defects])
    np.random.shuffle(measurements)
    
    # Fit normal distribution
    dist = get_distribution('normal')
    dist.fit(measurements)
    
    print("=" * 50)
    print("Quality Control Diagnostics")
    print("=" * 50)
    
    # 1. Residual analysis
    residuals = Diagnostics.residual_analysis(measurements, dist)
    print("\n1. Residual Analysis:")
    print(residuals.summary())
    
    # 2. Outlier detection
    outliers = Diagnostics.detect_outliers(
        measurements, 
        dist, 
        method='zscore',
        threshold=2.5  # Stricter for QC
    )
    
    print("\n2. Outlier Detection:")
    print(f"Detected outliers: {len(outliers.outlier_indices)}")
    print(f"Percentage: {100*len(outliers.outlier_indices)/len(measurements):.1f}%")
    
    # 3. Identify defective items
    outlier_values = measurements[outliers.outlier_indices]
    print(f"\n3. Outlier Statistics:")
    print(f"Mean: {np.mean(outlier_values):.2f}")
    print(f"Range: [{np.min(outlier_values):.2f}, {np.max(outlier_values):.2f}]")
    
    # 4. Q-Q diagnostics
    qq_data = Diagnostics.qq_diagnostics(measurements, dist)
    print(f"\n4. Q-Q Correlation: {qq_data['correlation']:.4f}")
    if qq_data['correlation'] < 0.95:
        print("⚠️  Warning: Poor fit detected")

Handling Outliers
-----------------

**Option 1: Remove and refit**

.. code-block:: python

    # Detect outliers
    outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
    
    # Remove outliers
    clean_data = np.delete(data, outliers.outlier_indices)
    
    # Refit
    dist_clean = get_distribution('normal')
    dist_clean.fit(clean_data)
    
    print(f"Original: μ={dist.params['loc']:.2f}, σ={dist.params['scale']:.2f}")
    print(f"Clean: μ={dist_clean.params['loc']:.2f}, σ={dist_clean.params['scale']:.2f}")

**Option 2: Use robust method**

.. code-block:: python

    # Fit with quantile method (robust to outliers)
    dist_robust = get_distribution('normal')
    dist_robust.fit(data, method='quantile')

**Option 3: Use heavy-tailed distribution**

.. code-block:: python

    # Student's t has heavier tails
    dist_t = get_distribution('studentt')
    dist_t.fit(data)
    
    # Compare GOF
    from distfit_pro.core.gof_tests import GOFTests
    ks_normal = GOFTests.kolmogorov_smirnov(data, dist)
    ks_t = GOFTests.kolmogorov_smirnov(data, dist_t)
    
    print(f"Normal KS p-value: {ks_normal.p_value:.4f}")
    print(f"Student's t KS p-value: {ks_t.p_value:.4f}")

Best Practices
--------------

1. **Always check diagnostics**
   
   Don't just fit - diagnose!

2. **Use multiple methods**
   
   Different diagnostics catch different problems.

3. **Investigate outliers**
   
   Don't automatically remove - understand WHY.

4. **Document decisions**
   
   Record what outliers were found and how handled.

5. **Iterate**
   
   .. code-block:: python
   
       # Fit → Diagnose → Improve → Repeat
       dist.fit(data)
       outliers = Diagnostics.detect_outliers(data, dist)
       
       if len(outliers.outlier_indices) > 0:
           # Investigate and decide
           pass

Next Steps
----------

- :doc:`07_weighted_data` - Handle weighted observations
- :doc:`08_visualization` - Visualize diagnostics
