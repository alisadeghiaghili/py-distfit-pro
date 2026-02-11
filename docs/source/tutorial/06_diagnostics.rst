Tutorial 6: Enhanced Diagnostics
=================================

Learn how to diagnose problems with fitted distributions.

Why Diagnostics?
----------------

**GOF tests tell you IF the fit is bad.**
**Diagnostics tell you WHY and WHERE.**

**Questions Diagnostics Answer:**

- Where does the fit fail?
- Which observations are problematic?
- Are there outliers?
- Is the model systematically wrong?

Residual Analysis
-----------------

**Residuals = difference between observed and expected.**

Types of Residuals
^^^^^^^^^^^^^^^^^^

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
    
    # Compute residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    print(residuals.summary())

**Output:**

::

    Residual Analysis Summary
    ==================================================
    Quantile Residuals:
      Mean: -0.001234
      Std: 0.998765
      Range: [-3.456, 3.234]
    
    Pearson Residuals:
      Mean: 0.000987
      Std: 1.002341
      
    Deviance Residuals:
      Mean: -0.000543
      Std: 0.997654

**4 Types of Residuals:**

1. **Quantile Residuals** - most useful, should be ~N(0,1)
2. **Pearson Residuals** - (observed - expected) / std
3. **Deviance Residuals** - based on log-likelihood
4. **Standardized Residuals** - normalized residuals

Interpreting Residuals
^^^^^^^^^^^^^^^^^^^^^^

**Good fit:**

- Mean ≈ 0
- Std ≈ 1
- No patterns
- Roughly normal

.. code-block:: python

    # Check residuals
    qres = residuals.quantile_residuals
    
    print(f"Mean: {np.mean(qres):.4f} (should be ≈ 0)")
    print(f"Std: {np.std(qres):.4f} (should be ≈ 1)")
    
    # Plot histogram (if using matplotlib)
    import matplotlib.pyplot as plt
    plt.hist(qres, bins=30, density=True, alpha=0.7)
    plt.title('Quantile Residuals')
    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.show()

Influence Diagnostics
---------------------

**Which observations have large influence on parameter estimates?**

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
      Influential: 23 observations
    
    Influential Indices: [45, 123, 234, 456, 567, 678, 789, 890, 901, 912]
    ... and 13 more

**Metrics:**

1. **Cook's Distance** - overall influence
   
   - > 4/n → influential
   - > 1 → very influential

2. **Leverage** - how "unusual" the observation is

3. **DFFITS** - change in fitted value if observation removed

**Examining Influential Points:**

.. code-block:: python

    # Get influential observations
    influential_idx = influence.influential_indices
    influential_values = data[influential_idx]
    
    print(f"\nInfluential observations:")
    print(f"  Indices: {influential_idx[:10]}...")
    print(f"  Values: {influential_values[:10]}")
    print(f"  Mean: {np.mean(influential_values):.4f}")
    print(f"  Std: {np.std(influential_values):.4f}")
    
    # Compare to overall data
    print(f"\nAll data:")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std: {np.std(data):.4f}")

Outlier Detection
-----------------

**4 Methods for Detecting Outliers**

Z-Score Method
^^^^^^^^^^^^^^

**Most common method.**

.. code-block:: python

    # Z-score outliers (|z| > 3)
    outliers_z = Diagnostics.detect_outliers(
        data=data,
        distribution=dist,
        method='zscore',
        threshold=3.0
    )
    
    print(outliers_z.summary())

**Output:**

::

    Outlier Detection Summary
    ==================================================
    Method: zscore
    Threshold: 3.000000
    Outliers Detected: 3
    
    Outlier Indices: [234, 567, 891]
    
    Score Range: [0.012, 3.456]

**Interpretation:**

- Threshold = 3 → observations > 3 std from mean
- Very strict: only extreme outliers
- Threshold = 2 → more outliers detected

IQR Method
^^^^^^^^^^

**Interquartile Range method (Tukey's rule).**

.. code-block:: python

    # IQR outliers
    outliers_iqr = Diagnostics.detect_outliers(
        data=data,
        distribution=dist,
        method='iqr',
        threshold=1.5  # Standard Tukey multiplier
    )
    
    print(f"Outliers: {len(outliers_iqr.outlier_indices)}")

**Thresholds:**

- 1.5 → standard (mild outliers)
- 3.0 → extreme outliers only

Likelihood Method
^^^^^^^^^^^^^^^^^

**Based on log-likelihood.**

.. code-block:: python

    # Likelihood-based outliers
    outliers_lik = Diagnostics.detect_outliers(
        data=data,
        distribution=dist,
        method='likelihood'
    )
    
    print(f"Outliers: {len(outliers_lik.outlier_indices)}")

**Advantages:**

- Distribution-specific
- Considers full model
- Sensitive to tail behavior

Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^

**Multivariate-inspired (univariate version).**

.. code-block:: python

    # Mahalanobis outliers
    outliers_maha = Diagnostics.detect_outliers(
        data=data,
        distribution=dist,
        method='mahalanobis',
        threshold=3.84  # Chi-square(1, 0.95)
    )

Comparing Methods
^^^^^^^^^^^^^^^^^

.. code-block:: python

    methods = ['zscore', 'iqr', 'likelihood', 'mahalanobis']
    
    print("\nOutlier Detection Comparison:")
    print(f"{'Method':<15} {'Count':<10} {'Indices (first 5)'}")
    print("-" * 50)
    
    for method in methods:
        result = Diagnostics.detect_outliers(data, dist, method=method)
        indices = result.outlier_indices[:5]
        print(f"{method:<15} {len(result.outlier_indices):<10} {indices}")

Q-Q Plot Diagnostics
--------------------

**Quantile-Quantile plot data.**

.. code-block:: python

    # Q-Q diagnostics
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    print(f"Q-Q Correlation: {qq_data['correlation']:.6f}")
    print(f"(Should be close to 1.0 for good fit)")
    
    # Get Q-Q plot data
    theoretical = qq_data['theoretical']
    sample = qq_data['sample']
    residuals = qq_data['residuals']

**Plotting Q-Q (if using matplotlib):**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.scatter(theoretical, sample, alpha=0.5)
    plt.plot([theoretical.min(), theoretical.max()], 
             [theoretical.min(), theoretical.max()], 
             'r--', lw=2)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f'Q-Q Plot (r = {qq_data["correlation"]:.4f})')
    plt.grid(True, alpha=0.3)
    plt.show()

**Interpreting Q-Q Plots:**

- **Points on line** → good fit ✓
- **S-shape** → distribution too light-tailed
- **Reverse S** → distribution too heavy-tailed
- **Points above line (right)** → right tail too heavy
- **Points below line (left)** → left tail too light

P-P Plot Diagnostics
--------------------

**Probability-Probability plot.**

.. code-block:: python

    # P-P diagnostics
    pp_data = Diagnostics.pp_diagnostics(data, dist)
    
    print(f"Max Deviation: {pp_data['max_deviation']:.6f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pp_data['theoretical'], pp_data['empirical'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('Theoretical Probability')
    plt.ylabel('Empirical Probability')
    plt.title('P-P Plot')
    plt.grid(True, alpha=0.3)
    plt.show()

**P-P vs Q-Q:**

- **P-P** → better for overall fit
- **Q-Q** → better for tail behavior
- **Use both** for complete picture

Worm Plot
---------

**Detrended Q-Q plot (easier to see deviations).**

.. code-block:: python

    # Worm plot data
    worm_data = Diagnostics.worm_plot_data(data, dist)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.scatter(worm_data['theoretical'], worm_data['worm_residuals'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Detrended Residuals')
    plt.title('Worm Plot')
    plt.grid(True, alpha=0.3)
    plt.show()

**Interpretation:**

- Points around 0 line → good fit
- Points outside ±2 bands → poor fit
- Patterns indicate systematic problems

Complete Diagnostic Workflow
-----------------------------

**Comprehensive analysis:**

.. code-block:: python

    def complete_diagnostics(data, dist):
        """Run all diagnostic checks"""
        
        print("="*60)
        print("COMPLETE DIAGNOSTIC REPORT")
        print("="*60)
        
        # 1. Residual Analysis
        print("\n1. RESIDUAL ANALYSIS")
        print("-"*60)
        residuals = Diagnostics.residual_analysis(data, dist)
        print(residuals.summary())
        
        # 2. Influence Diagnostics
        print("\n2. INFLUENCE DIAGNOSTICS")
        print("-"*60)
        influence = Diagnostics.influence_diagnostics(data, dist)
        print(influence.summary())
        
        # 3. Outlier Detection (all methods)
        print("\n3. OUTLIER DETECTION")
        print("-"*60)
        methods = ['zscore', 'iqr', 'likelihood']
        for method in methods:
            outliers = Diagnostics.detect_outliers(data, dist, method=method)
            print(f"\n{method.upper()}:")
            print(f"  Detected: {len(outliers.outlier_indices)} outliers")
            if len(outliers.outlier_indices) > 0:
                print(f"  Indices: {outliers.outlier_indices[:10]}")
        
        # 4. Q-Q Diagnostics
        print("\n4. Q-Q DIAGNOSTICS")
        print("-"*60)
        qq_data = Diagnostics.qq_diagnostics(data, dist)
        print(f"Correlation: {qq_data['correlation']:.6f}")
        if qq_data['correlation'] > 0.99:
            print("✓ Excellent fit")
        elif qq_data['correlation'] > 0.95:
            print("✓ Good fit")
        else:
            print("✗ Poor fit")
        
        # 5. Summary
        print("\n5. SUMMARY")
        print("-"*60)
        
        # Check residuals
        qres_mean = np.mean(residuals.quantile_residuals)
        qres_std = np.std(residuals.quantile_residuals)
        
        print(f"Residuals: mean={qres_mean:.4f}, std={qres_std:.4f}")
        
        if abs(qres_mean) < 0.1 and abs(qres_std - 1) < 0.1:
            print("✓ Residuals look good")
        else:
            print("✗ Residuals suggest problems")
        
        # Check influential points
        n_influential = len(influence.influential_indices)
        pct_influential = 100 * n_influential / len(data)
        
        print(f"Influential points: {n_influential} ({pct_influential:.1f}%)")
        
        if pct_influential < 5:
            print("✓ Few influential points")
        else:
            print("⚠ Many influential points - investigate")
        
        print("\n" + "="*60)
    
    # Run complete diagnostics
    complete_diagnostics(data, dist)

Example: Detecting Problems
---------------------------

**Wrong distribution fitted:**

.. code-block:: python

    # Generate gamma data
    data_gamma = np.random.gamma(2, 3, 1000)
    
    # Fit WRONG distribution
    dist_wrong = get_distribution('normal')
    dist_wrong.fit(data_gamma)
    
    # Diagnostics will reveal problems
    residuals = Diagnostics.residual_analysis(data_gamma, dist_wrong)
    print(residuals.summary())
    
    # Residuals won't be N(0,1)
    # Q-Q correlation will be low
    # Many outliers detected

**Contaminated data:**

.. code-block:: python

    # 95% normal + 5% outliers
    data_contaminated = np.concatenate([
        np.random.normal(10, 2, 950),
        np.random.uniform(30, 40, 50)
    ])
    
    dist_cont = get_distribution('normal')
    dist_cont.fit(data_contaminated)
    
    # Outlier detection
    outliers = Diagnostics.detect_outliers(
        data_contaminated, 
        dist_cont, 
        method='zscore',
        threshold=3
    )
    
    print(f"Outliers detected: {len(outliers.outlier_indices)}")
    # Should detect ~50 outliers

Best Practices
--------------

1. **Always run diagnostics after fitting**
   
   Don't trust fit blindly!

2. **Use multiple diagnostic methods**
   
   - Residuals
   - Q-Q plot
   - Outlier detection

3. **Visual inspection is crucial**
   
   Numbers + plots = best understanding

4. **Investigate influential points**
   
   .. code-block:: python
   
       influence = Diagnostics.influence_diagnostics(data, dist)
       if len(influence.influential_indices) > 0:
           print("Investigate these points!")

5. **Compare with GOF tests**
   
   Diagnostics + GOF tests = complete picture

Next Steps
----------

- :doc:`07_weighted_data` - Handle weighted observations
- :doc:`08_visualization` - Visual diagnostics
- :doc:`examples/real_world` - Real data examples
