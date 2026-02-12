Tutorial 6: Enhanced Diagnostics
=================================

Deep dive into residuals, influence, and outlier detection.

Why Diagnostics?
----------------

**GOF tests aren't enough!**

Tests tell you *if* a fit is bad, but not *why* or *where*.

**Diagnostics answer:**

- Which observations don't fit well?
- Are there influential points?
- Where are the outliers?
- What patterns exist in residuals?

Residual Analysis
-----------------

**Residuals = observed - expected**

DistFit Pro provides 4 types of residuals.

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
      Mean: 0.000123
      Std: 0.998765
      Range: [-3.245, 3.187]
    
    Pearson Residuals:
      Mean: -0.000456
      Std: 1.001234
      
    Deviance Residuals:
      Mean: -0.000234
      Std: 0.997654

Types of Residuals
^^^^^^^^^^^^^^^^^^

**1. Quantile Residuals (Best!)**

.. code-block:: python

    q_resid = residuals.quantile_residuals
    print(f"Mean: {np.mean(q_resid):.6f}")  # Should be ~0
    print(f"Std: {np.std(q_resid):.6f}")    # Should be ~1

**Properties:**

- Transforms to standard normal
- Best for checking normality of fit
- Works for any distribution

**2. Pearson Residuals**

.. code-block:: python

    p_resid = residuals.pearson_residuals
    # (observed - expected) / std

**Use for:**

- Traditional residual plots
- Comparing across datasets

**3. Deviance Residuals**

.. code-block:: python

    d_resid = residuals.deviance_residuals
    # Based on log-likelihood

**Use for:**

- Model comparison
- GLM-style diagnostics

**4. Standardized Residuals**

.. code-block:: python

    s_resid = residuals.standardized_residuals
    # Residuals / std(residuals)

**Use for:**

- Outlier detection
- Influence analysis

Influence Diagnostics
---------------------

**Identify observations that strongly affect the fit.**

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
      Influential: 12 observations
    
    Influential Indices: [23, 45, 67, 89, 123, 234, 345, 456, 567, 678]
    ... and 2 more

Cook's Distance
^^^^^^^^^^^^^^^

**Measures overall influence of each observation.**

.. code-block:: python

    cooks_d = influence.cooks_distance
    
    # Rule of thumb: D > 4/n is influential
    threshold = 4 / len(data)
    influential = np.where(cooks_d > threshold)[0]
    
    print(f"Influential observations: {len(influential)}")
    print(f"Indices: {influential[:10]}")

**Interpretation:**

- Large D = removing this point significantly changes the fit
- D > 4/n = potentially problematic
- D > 1 = definitely investigate!

Leverage
^^^^^^^^

**How much a point pulls the fit toward itself.**

.. code-block:: python

    leverage = influence.leverage
    
    # High leverage points
    high_leverage = leverage > 2 * np.mean(leverage)
    print(f"High leverage: {np.sum(high_leverage)} observations")

**Interpretation:**

- High leverage = extreme x value
- High leverage + large residual = influential

DFFITS
^^^^^^

**Standardized difference in fit.**

.. code-block:: python

    dffits = influence.dffits
    
    # Rule: |DFFITS| > 2*sqrt(p/n)
    # For 2 parameters and n=1000: threshold ≈ 0.089

Outlier Detection
-----------------

**Find observations that don't fit the distribution.**

DistFit Pro provides 4 detection methods.

Z-Score Method
^^^^^^^^^^^^^^

**Most common method.**

.. code-block:: python

    # Z-score outlier detection
    outliers_z = Diagnostics.detect_outliers(
        data, 
        dist, 
        method='zscore',
        threshold=3.0  # Standard: 3 standard deviations
    )
    
    print(outliers_z.summary())

**Output:**

::

    Outlier Detection Summary
    ==================================================
    Method: zscore
    Threshold: 3.000000
    Outliers Detected: 7
    
    Outlier Indices: [45, 123, 234, 567, 789, 890, 912]
    
    Score Range: [0.012, 3.456]

**When to use:**

- Normal distribution
- Symmetric data
- Standard analysis

IQR Method
^^^^^^^^^^

**Interquartile Range - robust to distribution shape.**

.. code-block:: python

    # IQR method
    outliers_iqr = Diagnostics.detect_outliers(
        data, 
        dist, 
        method='iqr',
        threshold=1.5  # Standard: 1.5 * IQR
    )
    
    print(f"Outliers: {len(outliers_iqr.outlier_indices)}")

**Thresholds:**

- 1.5: Mild outliers (Tukey's method)
- 3.0: Extreme outliers

**When to use:**

- Skewed data
- Unknown distribution
- Robust analysis needed

Likelihood Method
^^^^^^^^^^^^^^^^^

**Based on probability density.**

.. code-block:: python

    # Likelihood-based
    outliers_lik = Diagnostics.detect_outliers(
        data, 
        dist, 
        method='likelihood'
    )
    
    # Observations with low likelihood
    print(f"Low-probability observations: {len(outliers_lik.outlier_indices)}")

**When to use:**

- When you trust the fitted distribution
- For probability-based filtering
- Tail detection

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^

**Multivariate outlier detection (univariate version).**

.. code-block:: python

    # Mahalanobis distance
    outliers_mah = Diagnostics.detect_outliers(
        data, 
        dist, 
        method='mahalanobis'
    )

**When to use:**

- Statistical distance measure needed
- Chi-square threshold desired
- Formal statistical framework

Comparing Methods
^^^^^^^^^^^^^^^^^

**Different methods find different outliers!**

.. code-block:: python

    methods = ['zscore', 'iqr', 'likelihood', 'mahalanobis']
    
    results = {}
    for method in methods:
        outliers = Diagnostics.detect_outliers(data, dist, method=method)
        results[method] = set(outliers.outlier_indices)
    
    # Venn diagram analysis
    print("Outlier counts by method:")
    for method, indices in results.items():
        print(f"  {method}: {len(indices)}")
    
    # Consensus outliers (found by all methods)
    consensus = results['zscore']
    for indices in results.values():
        consensus &= indices
    
    print(f"\nConsensus outliers (all methods): {len(consensus)}")
    print(f"Indices: {sorted(consensus)}")

Q-Q Plot Diagnostics
--------------------

**Quantile-Quantile plot data for visual assessment.**

.. code-block:: python

    # Generate Q-Q plot data
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    print(f"Correlation: {qq_data['correlation']:.4f}")
    print(f"Max residual: {np.max(np.abs(qq_data['residuals'])):.4f}")

**Plot Q-Q:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    
    # Q-Q plot
    plt.subplot(1, 2, 1)
    plt.scatter(qq_data['theoretical'], qq_data['sample'], alpha=0.5)
    plt.plot([qq_data['theoretical'].min(), qq_data['theoretical'].max()],
             [qq_data['theoretical'].min(), qq_data['theoretical'].max()],
             'r--', label='Perfect fit')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f"Q-Q Plot (r={qq_data['correlation']:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(1, 2, 2)
    plt.scatter(qq_data['theoretical'], qq_data['residuals'], alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Q-Q Residuals')
    plt.title('Q-Q Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

**Interpretation:**

- Points on line = good fit
- Correlation close to 1 = excellent fit
- Systematic deviations = poor fit

P-P Plot Diagnostics
--------------------

**Probability-Probability plot.**

.. code-block:: python

    # P-P plot data
    pp_data = Diagnostics.pp_diagnostics(data, dist)
    
    print(f"Max deviation: {pp_data['max_deviation']:.6f}")

**Difference from Q-Q:**

- Q-Q: More sensitive to tails
- P-P: More sensitive to middle

Worm Plot
---------

**Detrended Q-Q plot.**

Removes linear trend to highlight deviations.

.. code-block:: python

    # Worm plot data
    worm_data = Diagnostics.worm_plot_data(data, dist)
    
    print(f"Slope: {worm_data['slope']:.4f}")
    print(f"Intercept: {worm_data['intercept']:.4f}")

**Plot:**

.. code-block:: python

    plt.figure(figsize=(10, 4))
    plt.scatter(worm_data['theoretical'], worm_data['worm_residuals'], alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.axhline(2, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(-2, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized Detrended Residuals')
    plt.title('Worm Plot')
    plt.grid(True, alpha=0.3)
    plt.show()

**Interpretation:**

- Random scatter around 0 = good fit
- Pattern = systematic deviation
- Points outside ±2 = poor fit

Practical Workflow
------------------

**Complete diagnostic pipeline:**

.. code-block:: python

    def comprehensive_diagnostics(data, dist):
        """
        Run full diagnostic suite
        """
        print("=" * 60)
        print("COMPREHENSIVE DIAGNOSTICS")
        print("=" * 60)
        
        # 1. Residuals
        print("\n1. RESIDUAL ANALYSIS")
        print("-" * 60)
        residuals = Diagnostics.residual_analysis(data, dist)
        print(residuals.summary())
        
        # 2. Influence
        print("\n2. INFLUENCE DIAGNOSTICS")
        print("-" * 60)
        influence = Diagnostics.influence_diagnostics(data, dist)
        print(influence.summary())
        
        # 3. Outliers (multiple methods)
        print("\n3. OUTLIER DETECTION")
        print("-" * 60)
        
        for method in ['zscore', 'iqr']:
            outliers = Diagnostics.detect_outliers(data, dist, method=method)
            print(f"\n{method.upper()}:")
            print(f"  Detected: {len(outliers.outlier_indices)}")
            if len(outliers.outlier_indices) > 0:
                print(f"  Indices: {outliers.outlier_indices[:10].tolist()}")
        
        # 4. Q-Q diagnostic
        print("\n4. Q-Q DIAGNOSTICS")
        print("-" * 60)
        qq_data = Diagnostics.qq_diagnostics(data, dist)
        print(f"Correlation: {qq_data['correlation']:.4f}")
        print(f"Interpretation: ", end="")
        if qq_data['correlation'] > 0.99:
            print("✅ Excellent fit")
        elif qq_data['correlation'] > 0.95:
            print("✅ Good fit")
        elif qq_data['correlation'] > 0.90:
            print("⚠️ Acceptable fit")
        else:
            print("❌ Poor fit")
        
        print("\n" + "=" * 60)
    
    # Run it
    comprehensive_diagnostics(data, dist)

Case Studies
------------

**Example 1: Detecting Model Misspecification**

.. code-block:: python

    # Generate gamma data
    gamma_data = np.random.gamma(2, 3, 1000)
    
    # Fit WRONG distribution (Normal)
    dist_wrong = get_distribution('normal')
    dist_wrong.fit(gamma_data)
    
    # Diagnostics will reveal the problem
    qq_data = Diagnostics.qq_diagnostics(gamma_data, dist_wrong)
    print(f"Q-Q correlation: {qq_data['correlation']:.4f}")  # Low!
    
    residuals = Diagnostics.residual_analysis(gamma_data, dist_wrong)
    print(f"Residual skewness: {np.mean(residuals.quantile_residuals**3):.4f}")  # Non-zero!

**Example 2: Quality Control**

.. code-block:: python

    # Manufacturing measurements
    measurements = np.concatenate([
        np.random.normal(100, 2, 980),  # Good products
        np.random.uniform(90, 95, 20)    # Defects
    ])
    
    dist = get_distribution('normal')
    dist.fit(measurements)
    
    # Find defects
    outliers = Diagnostics.detect_outliers(
        measurements, 
        dist, 
        method='zscore',
        threshold=2.5  # Stricter for QC
    )
    
    print(f"Defect rate: {len(outliers.outlier_indices)/len(measurements)*100:.2f}%")
    
    # Identify which are defects
    defect_indices = outliers.outlier_indices
    print(f"Defect values: {measurements[defect_indices[:5]]}")

Best Practices
--------------

1. **Always run diagnostics after fitting**
   
   Don't trust the fit without checking!

2. **Use multiple outlier detection methods**
   
   Consensus = more reliable

3. **Plot residuals**
   
   Visual patterns reveal problems

4. **Check influence for small datasets**
   
   Single points can dominate

5. **Investigate outliers, don't just remove**
   
   They might be real phenomena!

Next Steps
----------

- :doc:`07_weighted_data` - Handle weighted observations
- :doc:`08_visualization` - Visual diagnostics
- :doc:`09_advanced` - Advanced techniques
