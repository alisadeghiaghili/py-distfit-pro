Basic Examples
==============

Simple, practical examples for common tasks.

Example 1: Test Scores
----------------------

**Analyzing exam scores:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    # Exam scores (0-100)
    np.random.seed(42)
    scores = np.random.normal(75, 10, 150)
    scores = np.clip(scores, 0, 100)  # Keep in [0, 100]
    
    # Fit normal distribution
    dist = get_distribution('normal')
    dist.fit(scores)
    
    print("Test Score Distribution:")
    print(dist.summary())
    
    # GOF test
    ks_result = GOFTests.kolmogorov_smirnov(scores, dist)
    print(f"\nGOF Test: {ks_result.interpretation}")
    
    # What percentage scored above 85?
    prob_above_85 = 1 - dist.cdf(np.array([85]))[0]
    print(f"\nPercentage scoring > 85: {prob_above_85*100:.1f}%")
    
    # What score is 90th percentile?
    score_90th = dist.ppf(0.90)
    print(f"90th percentile score: {score_90th:.1f}")

Example 2: Product Lifetimes
-----------------------------

**Reliability analysis:**

.. code-block:: python

    # Component lifetimes (hours)
    lifetimes = np.array([
        1234, 1567, 892, 2134, 1789, 1456, 2345, 1678,
        1234, 1890, 2456, 1567, 1345, 2123, 1678, 1987,
        1456, 1789, 2234, 1567, 1345, 1890, 2123, 1678
    ])
    
    # Fit Weibull (common for reliability)
    dist = get_distribution('weibull')
    dist.fit(lifetimes)
    
    print("Weibull Parameters:")
    print(f"  Shape (k): {dist.params['c']:.3f}")
    print(f"  Scale (λ): {dist.params['scale']:.1f}")
    
    # Reliability metrics
    print("\nReliability Analysis:")
    
    # Mean Time To Failure
    mttf = dist.mean_time_to_failure()
    print(f"  MTTF: {mttf:.1f} hours")
    
    # Reliability at 1000 hours
    R_1000 = dist.reliability(1000)
    print(f"  R(1000h): {R_1000:.1%}")
    
    # Median lifetime
    median_life = dist.median()
    print(f"  Median lifetime: {median_life:.1f} hours")

Example 3: Customer Wait Times
-------------------------------

**Service time analysis:**

.. code-block:: python

    # Wait times (minutes)
    wait_times = np.random.exponential(5, 200)
    
    # Fit exponential
    dist = get_distribution('exponential')
    dist.fit(wait_times)
    
    print(f"Average wait time: {dist.mean():.2f} minutes")
    
    # Service level: % served within 10 minutes
    service_level = dist.cdf(np.array([10]))[0]
    print(f"Service level (<10 min): {service_level:.1%}")
    
    # How long to achieve 95% service level?
    target_time = dist.ppf(0.95)
    print(f"Time for 95% service: {target_time:.2f} minutes")

Example 4: Income Distribution
-------------------------------

**Lognormal income data:**

.. code-block:: python

    # Annual income ($1000s)
    incomes = np.random.lognormal(np.log(50), 0.6, 1000)
    
    # Fit lognormal
    dist = get_distribution('lognormal')
    dist.fit(incomes)
    
    print("Income Distribution:")
    print(f"  Median: ${dist.median():.1f}K")
    print(f"  Mean: ${dist.mean():.1f}K")
    print(f"  Std: ${dist.std():.1f}K")
    
    # Income brackets
    print("\nIncome Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        value = dist.ppf(p/100)
        print(f"  {p}th: ${value:.1f}K")

Example 5: Quality Control
--------------------------

**Defect rate analysis:**

.. code-block:: python

    # Number of defects per batch (count data)
    defects = np.array([0, 1, 0, 2, 1, 0, 3, 1, 0, 1, 2, 0, 1, 0, 4])
    
    # Fit Poisson
    dist = get_distribution('poisson')
    dist.fit(defects)
    
    print(f"Average defects per batch: {dist.mean():.2f}")
    
    # Probability of zero defects
    prob_zero = dist.pdf(np.array([0]))[0]
    print(f"P(zero defects): {prob_zero:.1%}")
    
    # Probability of > 3 defects (out of control)
    prob_high = 1 - dist.cdf(np.array([3]))[0]
    print(f"P(>3 defects): {prob_high:.1%}")

Example 6: Survey Response Rates
---------------------------------

**Beta distribution for proportions:**

.. code-block:: python

    # Response rates from different campaigns
    response_rates = np.array([
        0.23, 0.34, 0.28, 0.31, 0.25, 0.29, 0.32, 0.27,
        0.30, 0.26, 0.33, 0.24, 0.28, 0.31, 0.29, 0.27
    ])
    
    # Fit Beta (bounded [0,1])
    dist = get_distribution('beta')
    dist.fit(response_rates)
    
    print("Response Rate Distribution:")
    print(f"  Mean: {dist.mean():.1%}")
    print(f"  Mode: {dist.mode():.1%}")
    
    # Credible interval
    ci_lower = dist.ppf(0.025)
    ci_upper = dist.ppf(0.975)
    print(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

Example 7: Website Traffic
--------------------------

**Page views per hour:**

.. code-block:: python

    # Hourly page views
    page_views = np.array([
        1234, 1456, 1678, 1345, 1567, 1789, 1456, 1890,
        2123, 2345, 2567, 2234, 2456, 2678, 2123, 1890,
        1678, 1456, 1234, 1567, 1789, 1345, 1234, 1456
    ])
    
    # Fit normal
    dist = get_distribution('normal')
    dist.fit(page_views)
    
    print("Traffic Statistics:")
    print(f"  Average: {dist.mean():.0f} views/hour")
    print(f"  Peak (95th): {dist.ppf(0.95):.0f} views/hour")
    
    # Capacity planning: handle 99% of traffic
    capacity = dist.ppf(0.99)
    print(f"\nRecommended capacity: {capacity:.0f} views/hour")
