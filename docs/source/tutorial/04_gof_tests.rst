Tutorial 4: Goodness-of-Fit Tests
==================================

Learn how to test if a distribution fits your data well.

Why Test Goodness-of-Fit?
--------------------------

Just because you can fit a distribution doesn't mean it's a good fit!

**You need to verify:**

- Does the fitted distribution match the data?
- Are deviations statistically significant?
- Can we trust predictions from this model?

Available Tests
---------------

DistFit Pro provides 4 GOF tests:

1. **Kolmogorov-Smirnov (KS)** - general purpose
2. **Anderson-Darling (AD)** - sensitive to tails
3. **Chi-Square (χ²)** - frequency-based
4. **Cramér-von Mises (CvM)** - middle-focused

Kolmogorov-Smirnov Test
-----------------------

**Most popular GOF test.**

Measures maximum distance between empirical and theoretical CDFs.

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    # KS test
    ks_result = GOFTests.kolmogorov_smirnov(data, dist)
    print(ks_result)

**Output:**

::

    Kolmogorov-Smirnov Test Results
    ==================================================
    Statistic: 0.018234
    P-value: 0.856432
    Reject H0 (α=0.05): False
    
    The data is consistent with the Normal (Gaussian) Distribution.
    Evidence: D = 0.018234 ≤ 0.043011 (critical value)
    No significant evidence against the fitted distribution.

**Interpretation:**

- **P-value > 0.05** → Good fit! ✅
- **P-value < 0.05** → Poor fit ❌

**When to use:**

- Continuous data
- General-purpose testing
- Small to medium samples

Anderson-Darling Test
---------------------

**More sensitive to tails than KS.**

Gives more weight to extreme values.

.. code-block:: python

    # AD test
    ad_result = GOFTests.anderson_darling(data, dist)
    print(ad_result)

**Output:**

::

    Anderson-Darling Test Results
    ==================================================
    Statistic: 0.324567
    P-value: 0.523412
    Reject H0 (α=0.05): False
    
    The data is consistent with the Normal (Gaussian) Distribution.
    Evidence: A² = 0.324567 ≤ 2.492000 (critical value)
    Good fit, including in the tails.

**When to use:**

- Tails are important (e.g., risk analysis)
- Financial data
- Extreme value analysis

Chi-Square Test
---------------

**Compares observed vs expected frequencies.**

.. code-block:: python

    # Chi-square test
    chi_result = GOFTests.chi_square(data, dist, n_bins=20)
    print(chi_result)

**Parameters:**

- ``n_bins``: Number of bins (default: auto-calculated)
- More bins = more detailed, but needs more data

**When to use:**

- Discrete data
- Grouped data
- When you want to see which regions fit poorly

Cramér-von Mises Test
-----------------------

**Similar to KS but uses squared differences.**

More sensitive to middle deviations.

.. code-block:: python

    # CvM test
    cvm_result = GOFTests.cramer_von_mises(data, dist)
    print(cvm_result)

Run All Tests
-------------

**Convenience method to run all 4 tests:**

.. code-block:: python

    # Run all tests
    results = GOFTests.run_all_tests(data, dist, alpha=0.05)
    
    # Summary table
    print(GOFTests.summary_table(results))

**Output:**

::

    ╔═══════════════════════════════════════════════════════════════╗
    ║                 Goodness-of-Fit Test Summary                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Test                  Statistic    P-value    Reject H0      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Kolmogorov-Smirnov      0.018234   0.856432      No        ║
    ║  Anderson-Darling        0.324567   0.523412      No        ║
    ║  Chi-Square             18.234567   0.312456      No        ║
    ║  Cramér-von Mises        0.042345   0.678901      No        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    ✅ All tests passed: Distribution fits well!

Example: Poor Fit
-----------------

**What happens when the fit is bad?**

.. code-block:: python

    # Generate gamma data
    data_gamma = np.random.gamma(2, 3, 1000)
    
    # Fit WRONG distribution (Normal)
    dist_wrong = get_distribution('normal')
    dist_wrong.fit(data_gamma)
    
    # Test
    results = GOFTests.run_all_tests(data_gamma, dist_wrong)
    print(GOFTests.summary_table(results))

**Output:**

::

    ║  Kolmogorov-Smirnov      0.156234   0.000012     Yes        ║
    ║  Anderson-Darling       12.345678   0.000001     Yes        ║
    ║  Chi-Square            156.234567   0.000000     Yes        ║
    ║  Cramér-von Mises        2.345678   0.000023     Yes        ║
    
    ❌ All tests failed: Distribution does NOT fit!

Comparing Distributions
-----------------------

**Find the best distribution for your data:**

.. code-block:: python

    # Try multiple distributions
    candidates = ['normal', 'lognormal', 'gamma', 'weibull']
    
    test_results = {}
    
    for dist_name in candidates:
        dist = get_distribution(dist_name)
        try:
            dist.fit(data)
            
            # Run KS test
            ks_result = GOFTests.kolmogorov_smirnov(data, dist)
            
            test_results[dist_name] = {
                'p_value': ks_result.p_value,
                'statistic': ks_result.statistic,
                'reject': ks_result.reject_null
            }
        except:
            test_results[dist_name] = None
    
    # Print results
    print("\nGOF Test Results:")
    print(f"{'Distribution':<15} {'P-value':<12} {'Reject?':<10}")
    print("-" * 40)
    
    for name, result in test_results.items():
        if result:
            print(f"{name:<15} {result['p_value']:<12.6f} {result['reject']}")
    
    # Best fit = highest p-value (among non-rejected)
    valid_results = {k: v for k, v in test_results.items() 
                     if v and not v['reject']}
    
    if valid_results:
        best = max(valid_results.items(), key=lambda x: x[1]['p_value'])
        print(f"\nBest fit: {best[0]} (p-value = {best[1]['p_value']:.6f})")

Choosing Significance Level
---------------------------

**Default α = 0.05, but you can change it:**

.. code-block:: python

    # Stricter (more likely to reject)
    result_strict = GOFTests.kolmogorov_smirnov(data, dist, alpha=0.01)
    
    # More lenient (less likely to reject)
    result_lenient = GOFTests.kolmogorov_smirnov(data, dist, alpha=0.10)

**Guidelines:**

- α = 0.01: Very strict (99% confidence)
- α = 0.05: Standard (95% confidence) **← recommended**
- α = 0.10: Lenient (90% confidence)

Limitations
-----------

**GOF tests can fail to detect:**

1. **Small sample issues**
   
   .. code-block:: python
   
       # Only 20 points - tests have low power
       small_data = np.random.normal(10, 2, 20)
       
       # May not reject even if fit is poor

2. **Large sample sensitivity**
   
   .. code-block:: python
   
       # 100,000 points - tests very sensitive
       huge_data = np.random.normal(10, 2, 100000)
       
       # May reject even for tiny deviations

3. **Multiple testing**
   
   If you test 20 distributions, expect 1 to fail by chance!

Best Practices
--------------

1. **Always test GOF**
   
   Don't just fit - test!

2. **Use multiple tests**
   
   Different tests catch different problems.

3. **Consider domain knowledge**
   
   Statistical tests + expert knowledge = best choice

4. **Visual inspection**
   
   Always plot! (See :doc:`08_visualization`)

5. **Use with model selection**
   
   Combine GOF tests with AIC/BIC.

Next Steps
----------

- :doc:`05_bootstrap` - Confidence intervals
- :doc:`06_diagnostics` - Detailed diagnostics
- :doc:`08_visualization` - Visual GOF assessment
