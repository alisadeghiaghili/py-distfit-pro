Tutorial 6: Diagnostics
=======================

Learn to diagnose problems with fitted distributions.

Why Diagnostics?
----------------

**Fitting is not enough - you need to check:**

- Are there systematic deviations?
- Which observations are influential?
- Are there outliers?
- Does the model fit well everywhere?

Residual Analysis
-----------------

**Residuals show deviations from the fitted model.**

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
      Mean: 0.001234
      Std: 1.003456
      Range: [-3.245, 3.198]
    
    Pearson Residuals:
      Mean: 0.000987
      Std: 0.998765
      
    Deviance Residuals:
      Mean: 0.001456
      Std: 1.002345

**Ideal residuals:**

- Mean ≈ 0
- Std ≈ 1
- Symmetric distribution
- No patterns

Plotting Residuals
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Get residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Quantile residuals histogram
    axes[0, 0].hist(residuals.quantile_residuals, bins=50, 
                    alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Quantile Residuals')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram of Quantile Residuals')
    axes[0, 0].axvline(0, color='red', linestyle='--')
    
    # 2. Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals.quantile_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # 3. Residuals vs fitted values
    fitted_values = dist.cdf(data)
    axes[1, 0].scatter(fitted_values, residuals.quantile_residuals, 
                       alpha=0.5, s=10)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Fitted CDF')
    axes[1, 0].set_ylabel('Quantile Residuals')
    axes[1, 0].set_title('Residuals vs Fitted')
    
    # 4. Residuals vs index (time series check)
    axes[1, 1].scatter(range(len(data)), residuals.quantile_residuals, 
                       alpha=0.5, s=10)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Observation Index')
    axes[1, 1].set_ylabel('Quantile Residuals')
    axes[1, 1].set_title('Residuals vs Order')
    
    plt.tight_layout()
    plt.savefig('residuals_diagnostic.png', dpi=150)
    plt.show()

**What to look for:**

- ✅ Random scatter around zero
- ❌ Patterns (curved, fan-shaped)
- ❌ Outliers
- ❌ Non-normal Q-Q plot

Influence Diagnostics
---------------------

**Find observations that strongly affect parameter estimates.**

.. code-block:: python

    # Compute influence diagnostics
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
    
    Influential Indices: [15, 42, 87, 156, 234, 345, 456, 567, 678, 789]
    ... and 13 more

**Cook's Distance:**

- Measures influence of each observation
- Threshold: 4/n
- Large values indicate influential points

Plotting Influence
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    influence = Diagnostics.influence_diagnostics(data, dist)
    
    plt.figure(figsize=(12, 5))
    
    # Cook's distance plot
    plt.subplot(1, 2, 1)
    plt.stem(range(len(data)), influence.cooks_distance, 
             markerfmt=' ', basefmt=' ')
    threshold = 4 / len(data)
    plt.axhline(threshold, color='red', linestyle='--', 
                label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Observation Index')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance Plot")
    plt.legend()
    
    # Highlight influential points
    influential_mask = influence.cooks_distance > threshold
    plt.scatter(np.where(influential_mask)[0],
                influence.cooks_distance[influential_mask],
                color='red', s=50, zorder=5, label='Influential')
    
    # Leverage vs residuals
    plt.subplot(1, 2, 2)
    residuals = Diagnostics.residual_analysis(data, dist)
    plt.scatter(influence.leverage, 
                residuals.standardized_residuals,
                alpha=0.5)
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals')
    plt.title('Leverage vs Residuals')
    plt.axhline(0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('influence_diagnostic.png', dpi=150)
    plt.show()

Outlier Detection
-----------------

**Four methods to detect outliers:**

1. Z-Score Method
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Detect outliers using z-scores
    outliers_z = Diagnostics.detect_outliers(
        data, 
        dist, 
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
    Outliers Detected: 7
    
    Outlier Indices: [23, 156, 342, 567, 678, 823, 945]
    
    Score Range: [0.012, 4.567]

2. IQR Method
^^^^^^^^^^^^^

.. code-block:: python

    # Interquartile range method
    outliers_iqr = Diagnostics.detect_outliers(
        data,
        dist,
        method='iqr',
        threshold=1.5  # Standard: 1.5 * IQR
    )
    
    print(f"Outliers detected: {len(outliers_iqr.outlier_indices)}")

3. Likelihood Method
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Low-likelihood outliers
    outliers_lik = Diagnostics.detect_outliers(
        data,
        dist,
        method='likelihood'
    )
    
    print(outliers_lik.summary())

4. Mahalanobis Distance
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Mahalanobis distance
    outliers_maha = Diagnostics.detect_outliers(
        data,
        dist,
        method='mahalanobis'
    )
    
    print(outliers_maha.summary())

Comparing Outlier Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    methods = ['zscore', 'iqr', 'likelihood', 'mahalanobis']
    
    results = {}
    for method in methods:
        outliers = Diagnostics.detect_outliers(data, dist, method=method)
        results[method] = {
            'n_outliers': len(outliers.outlier_indices),
            'indices': set(outliers.outlier_indices)
        }
    
    # Print comparison
    print("Outlier Detection Comparison:")
    print(f"{'Method':<15} {'Count':<10}")
    print("-" * 25)
    for method, result in results.items():
        print(f"{method:<15} {result['n_outliers']:<10}")
    
    # Find consensus outliers (detected by all methods)
    consensus = results['zscore']['indices']
    for method in methods[1:]:
        consensus = consensus.intersection(results[method]['indices'])
    
    print(f"\nConsensus outliers: {sorted(list(consensus))}")

Q-Q Plot Diagnostics
--------------------

**Quantile-Quantile plot data:**

.. code-block:: python

    # Get Q-Q data
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    print(f"Q-Q Correlation: {qq_data['correlation']:.6f}")
    
    # Plot Q-Q
    plt.figure(figsize=(8, 8))
    plt.scatter(qq_data['theoretical'], qq_data['sample'], 
                alpha=0.5, s=20)
    
    # Add 45-degree line
    min_val = min(qq_data['theoretical'].min(), qq_data['sample'].min())
    max_val = max(qq_data['theoretical'].max(), qq_data['sample'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2, label='Perfect fit')
    
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f'Q-Q Plot (correlation = {qq_data["correlation"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('qq_plot.png', dpi=150)
    plt.show()

**Interpretation:**

- Points on line → good fit
- Systematic deviation → wrong distribution
- Correlation close to 1 → excellent fit

P-P Plot Diagnostics
--------------------

**Probability-Probability plot:**

.. code-block:: python

    # Get P-P data
    pp_data = Diagnostics.pp_diagnostics(data, dist)
    
    print(f"Max deviation: {pp_data['max_deviation']:.6f}")
    
    # Plot P-P
    plt.figure(figsize=(8, 8))
    plt.scatter(pp_data['theoretical'], pp_data['empirical'],
                alpha=0.5, s=20)
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect fit')
    
    plt.xlabel('Theoretical Probabilities')
    plt.ylabel('Empirical Probabilities')
    plt.title(f'P-P Plot (max dev = {pp_data["max_deviation"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('pp_plot.png', dpi=150)
    plt.show()

Worm Plot
---------

**Detrended Q-Q plot - easier to see deviations:**

.. code-block:: python

    # Get worm plot data
    worm_data = Diagnostics.worm_plot_data(data, dist)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(worm_data['theoretical'], 
                worm_data['worm_residuals'],
                alpha=0.5, s=20)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.axhline(2, color='orange', linestyle=':', alpha=0.5)
    plt.axhline(-2, color='orange', linestyle=':', alpha=0.5)
    
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized Deviations')
    plt.title('Worm Plot')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('worm_plot.png', dpi=150)
    plt.show()

**Interpretation:**

- Points around zero → good fit
- Outside ±2 band → poor fit
- Systematic pattern → wrong distribution

Complete Diagnostic Report
--------------------------

**Generate a full diagnostic report:**

.. code-block:: python

    def diagnostic_report(data, dist):
        """Generate comprehensive diagnostic report"""
        
        print("="*60)
        print("COMPREHENSIVE DIAGNOSTIC REPORT")
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
        
        # 3. Outlier Detection
        print("\n3. OUTLIER DETECTION")
        print("-"*60)
        for method in ['zscore', 'iqr', 'likelihood']:
            outliers = Diagnostics.detect_outliers(data, dist, method=method)
            print(f"\n{method.upper()}: {len(outliers.outlier_indices)} outliers")
        
        # 4. Q-Q Diagnostics
        print("\n4. Q-Q DIAGNOSTICS")
        print("-"*60)
        qq_data = Diagnostics.qq_diagnostics(data, dist)
        print(f"Q-Q Correlation: {qq_data['correlation']:.6f}")
        
        # 5. P-P Diagnostics
        print("\n5. P-P DIAGNOSTICS")
        print("-"*60)
        pp_data = Diagnostics.pp_diagnostics(data, dist)
        print(f"Max Deviation: {pp_data['max_deviation']:.6f}")
        
        # 6. Overall Assessment
        print("\n6. OVERALL ASSESSMENT")
        print("-"*60)
        
        issues = []
        
        # Check residuals
        if abs(np.mean(residuals.quantile_residuals)) > 0.1:
            issues.append("⚠️  Residual mean not close to zero")
        if abs(np.std(residuals.quantile_residuals) - 1.0) > 0.1:
            issues.append("⚠️  Residual std not close to 1")
        
        # Check Q-Q correlation
        if qq_data['correlation'] < 0.98:
            issues.append(f"⚠️  Low Q-Q correlation ({qq_data['correlation']:.4f})")
        
        # Check influential points
        if len(influence.influential_indices) > len(data) * 0.05:
            issues.append(f"⚠️  Many influential points ({len(influence.influential_indices)})")
        
        if issues:
            print("Issues detected:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ No major issues detected!")
            print("Distribution fits well.")
        
        print("\n" + "="*60)
    
    # Run report
    diagnostic_report(data, dist)

Next Steps
----------

- :doc:`07_weighted_data` - Weighted diagnostics
- :doc:`08_visualization` - Visual diagnostics
