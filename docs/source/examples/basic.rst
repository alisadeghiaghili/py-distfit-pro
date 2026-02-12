Basic Examples
==============

Example 1: Analyze Customer Wait Times
---------------------------------------

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    # Customer wait times (minutes)
    wait_times = [2.3, 4.1, 1.8, 3.5, 2.9, 5.2, 1.5, 3.8, 2.7, 4.3]
    
    # Try exponential (common for wait times)
    dist = get_distribution('exponential')
    dist.fit(wait_times)
    
    print("\nFitted Distribution:")
    print(dist.summary())
    
    # Test fit
    results = GOFTests.run_all_tests(wait_times, dist)
    print(GOFTests.summary_table(results))
    
    # Predict: What's the probability of waiting > 5 minutes?
    prob = 1 - dist.cdf(np.array([5.0]))[0]
    print(f"\nP(wait > 5 min) = {prob:.1%}")

Example 2: Product Defect Rates
--------------------------------

.. code-block:: python

    # Defect counts per 100 units
    defects = [0, 1, 0, 2, 1, 0, 1, 3, 0, 1]
    
    # Poisson distribution
    dist = get_distribution('poisson')
    dist.fit(defects)
    
    print(f"Average defect rate: {dist.params['mu']:.2f} per 100 units")
