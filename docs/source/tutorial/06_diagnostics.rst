Tutorial 6: Diagnostics
=======================

Learn how to diagnose distribution fits using residuals, influence analysis, and outlier detection.

Why Diagnostics?
----------------

GOF tests tell you **IF** the fit is good.  
Diagnostics tell you **WHERE** and **WHY** it might be bad.

**Questions diagnostics answer:**

- Which observations don't fit well?
- Are there outliers affecting the fit?
- Is the fit poor in the tails? Middle?
- Which points have high influence?

Residual Analysis
-----------------

Residuals measure how far observations are from the fitted distribution.

**4 Types of Residuals:**

1. **Quantile Residuals** - transform to N(0,1)
2. **Pearson Residuals** - standardized differences
3. **Deviance Residuals** - based on likelihood
4. **Standardized Residuals** - normalized

**Code:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.diagnostics import Diagnostics
    import numpy as np
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Fit
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
      Mean: -0.000234
      Std: 1.001234
      Range: [-3.234, 3.567]
    
    Pearson Residuals:
      Mean: 0.000123
      Std: 0.998765
    
    Deviance Residuals:
      Mean: -0.000456
      Std: 1.002345

**Interpretation:**

- Mean ≈ 0 ✓ (no systematic bias)
- Std ≈ 1 ✓ (correctly standardized)
- Range ≈ [-3, 3] ✓ (no extreme outliers)

**Good fit indicators:**
- Residuals look like N(0,1)
- No patterns in residual plots
- Mean close to 0, std close to 1

Quantile Residuals in Detail
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Best residuals for continuous distributions.**

**How they work:**

1. Transform data through CDF: u = F(x)
2. Apply inverse normal: r = Φ⁻¹(u)
3. Result: r ~ N(0,1) if fit is good

.. code-block:: python

    # Access specific residuals
    q_residuals = residuals.quantile_residuals
    
    # Plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Histogram
    plt.subplot(131)
    plt.hist(q_residuals, bins=50, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x), 'r-', label='N(0,1)')
    plt.xlabel('Quantile Residuals')
    plt.legend()
    
    # Q-Q plot
    plt.subplot(132)
    from scipy import stats
    stats.probplot(q_residuals, dist='norm', plot=plt)
    plt.title('Q-Q Plot')
    
    # Index plot
    plt.subplot(133)
    plt.scatter(range(len(q_residuals)), q_residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.axhline(2, color='orange', linestyle=':')
    plt.axhline(-2, color='orange', linestyle=':')
    plt.xlabel('Index')
    plt.ylabel('Quantile Residuals')
    
    plt.tight_layout()
    plt.show()

Influence Diagnostics
---------------------

**Identify observations that strongly affect parameter estimates.**

**3 Metrics:**

1. **Cook's Distance** - overall influence
2. **Leverage** - potential influence
3. **DFFITS** - impact on fitted values

**Code:**

.. code-block:: python

    # Influence diagnostics
    influence = Diagnostics.influence_diagnostics(data, dist)
    print(influence.summary())

**Output:**

::

    Influence Diagnostics Summary
    ==================================================
    Cook's Distance:
      Max: 0.023456
      Threshold: 0.004000
      Influential: 12 observations
    
    Influential Indices: [45, 123, 234, 456, 567, 678, 789, 890, 901, 912]
    ... and 2 more

**Cook's Distance Interpretation:**

- D > 1: **Very influential** (investigate!)
- D > 4/n: **Moderately influential**
- D < 4/n: Not influential

**Example: Remove influential points**

.. code-block:: python

    # Get influential points
    influential_idx = influence.influential_indices
    
    # Refit without them
    data_clean = np.delete(data, influential_idx)
    dist_clean = get_distribution('normal')
    dist_clean.fit(data_clean)
    
    # Compare parameters
    print("Original params:", dist.params)
    print("Clean params:", dist_clean.params)
    print(f"Change: {abs(dist.params['loc'] - dist_clean.params['loc']):.4f}")

Outlier Detection
-----------------

**4 Methods available:**

1. **Z-score** - based on standard deviations
2. **IQR** - interquartile range method
3. **Likelihood** - based on probability
4. **Mahalanobis** - distance-based

Z-Score Method
^^^^^^^^^^^^^^

**Most common and intuitive.**

.. code-block:: python

    # Z-score outliers
    outliers = Diagnostics.detect_outliers(
        data, dist,
        method='zscore',
        threshold=3.0  # |z| > 3
    )
    
    print(outliers.summary())

**Output:**

::

    Outlier Detection Summary
    ==================================================
    Method: zscore
    Threshold: 3.000000
    Outliers Detected: 3
    
    Outlier Indices: [234, 567, 890]
    
    Score Range: [0.001, 3.456]

**Access outliers:**

.. code-block:: python

    outlier_idx = outliers.outlier_indices
    outlier_values = data[outlier_idx]
    
    print(f"Outliers: {outlier_values}")

**Custom threshold:**

.. code-block:: python

    # Stricter (fewer outliers)
    outliers_strict = Diagnostics.detect_outliers(
        data, dist, method='zscore', threshold=4.0
    )
    
    # More lenient (more outliers)
    outliers_lenient = Diagnostics.detect_outliers(
        data, dist, method='zscore', threshold=2.5
    )

IQR Method
^^^^^^^^^^

**Robust to distribution assumptions.**

.. code-block:: python

    outliers_iqr = Diagnostics.detect_outliers(
        data, dist,
        method='iqr',
        threshold=1.5  # Standard multiplier
    )
    
    print(outliers_iqr.summary())

**How it works:**

- Q1 = 25th percentile
- Q3 = 75th percentile  
- IQR = Q3 - Q1
- Lower fence = Q1 - 1.5*IQR
- Upper fence = Q3 + 1.5*IQR
- Outliers = anything outside fences

Likelihood Method
^^^^^^^^^^^^^^^^^

**Based on probability.**

.. code-block:: python

    outliers_lik = Diagnostics.detect_outliers(
        data, dist,
        method='likelihood',
        threshold=None  # Auto: 1st percentile of log-lik
    )

**Interpretation:**

Outliers have very low probability under the fitted distribution.

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^

**Multivariate generalization (univariate = Z-score squared).**

.. code-block:: python

    outliers_mah = Diagnostics.detect_outliers(
        data, dist,
        method='mahalanobis',
        threshold=None  # Auto: χ²(0.99, df=1)
    )

Q-Q and P-P Plot Data
---------------------

**Generate data for diagnostic plots.**

Q-Q Plot
^^^^^^^^

**Quantile-Quantile:** compare empirical vs theoretical quantiles.

.. code-block:: python

    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    print(f"Correlation: {qq_data['correlation']:.4f}")
    
    # Plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 6))
    plt.scatter(
        qq_data['theoretical'],
        qq_data['sample'],
        alpha=0.5
    )
    
    # Reference line
    lims = [qq_data['theoretical'].min(), qq_data['theoretical'].max()]
    plt.plot(lims, lims, 'r--', label='Perfect fit')
    
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f"Q-Q Plot (r={qq_data['correlation']:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

**Good fit:** Points lie on diagonal line.

**Interpretation:**

- Points above line in left tail → left-skewed
- Points below line in right tail → right-skewed
- S-curve → heavy tails
- Correlation > 0.99 → excellent fit

P-P Plot
^^^^^^^^

**Probability-Probability:** compare empirical vs theoretical CDFs.

.. code-block:: python

    pp_data = Diagnostics.pp_diagnostics(data, dist)
    
    print(f"Max deviation: {pp_data['max_deviation']:.6f}")
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(
        pp_data['theoretical'],
        pp_data['empirical'],
        alpha=0.5
    )
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect fit')
    plt.xlabel('Theoretical Probabilities')
    plt.ylabel('Empirical Probabilities')
    plt.title(f"P-P Plot (max dev={pp_data['max_deviation']:.4f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

**Better for detecting:**
- Shape differences
- Center deviations

Worm Plot
^^^^^^^^^

**Detrended Q-Q plot** - easier to see deviations.

.. code-block:: python

    worm_data = Diagnostics.worm_plot_data(data, dist)
    
    plt.figure(figsize=(8, 4))
    plt.scatter(
        worm_data['theoretical'],
        worm_data['worm_residuals'],
        alpha=0.5
    )
    plt.axhline(0, color='r', linestyle='--')
    plt.axhline(2, color='orange', linestyle=':', alpha=0.5)
    plt.axhline(-2, color='orange', linestyle=':', alpha=0.5)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Worm Residuals')
    plt.title('Worm Plot')
    plt.grid(alpha=0.3)
    plt.show()

**Good fit:** Points randomly scattered around 0 within ±2.

Complete Diagnostic Workflow
-----------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.diagnostics import Diagnostics
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    # 1. Fit distribution
    data = load_your_data()
    dist = get_distribution('lognormal')
    dist.fit(data)
    
    # 2. GOF tests
    gof_results = GOFTests.run_all_tests(data, dist)
    print(GOFTests.summary_table(gof_results))
    
    # 3. Residual analysis
    residuals = Diagnostics.residual_analysis(data, dist)
    print(residuals.summary())
    
    # 4. Check for outliers
    outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
    print(f"Found {len(outliers.outlier_indices)} outliers")
    
    # 5. Influence diagnostics
    influence = Diagnostics.influence_diagnostics(data, dist)
    print(f"Influential points: {len(influence.influential_indices)}")
    
    # 6. Visual diagnostics
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    print(f"Q-Q correlation: {qq_data['correlation']:.4f}")
    
    # 7. Decision
    if qq_data['correlation'] > 0.99 and not any(gof_results.values() for r in gof_results.values() if r.reject_null):
        print("✅ Excellent fit!")
    elif len(outliers.outlier_indices) > len(data) * 0.05:
        print("⚠️ Many outliers - consider robust method or different distribution")
    else:
        print("✅ Acceptable fit")

Next Steps
----------

- :doc:`07_weighted_data` - Handle weighted observations
- :doc:`08_visualization` - Create diagnostic plots
- :doc:`09_advanced` - Advanced techniques
