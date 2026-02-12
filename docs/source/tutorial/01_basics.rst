Tutorial 1: The Basics
======================

Learn the fundamentals of DistFit Pro.

What is Distribution Fitting?
------------------------------

Distribution fitting is the process of finding a probability distribution that best describes your data.

**Why is it useful?**

- Understand data behavior
- Make predictions
- Generate synthetic data
- Risk analysis
- Quality control

Your First Distribution Fit
----------------------------

**Step 1: Import and Generate Data**

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate 1000 samples from Normal(10, 2)
    data = np.random.normal(loc=10, scale=2, size=1000)
    
    print(f"Data shape: {data.shape}")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Std: {np.std(data):.2f}")

**Step 2: Choose a Distribution**

.. code-block:: python

    # Get Normal distribution
    dist = get_distribution('normal')
    
    print(dist.info.display_name)
    print(f"Parameters: {dist.info.parameters}")
    print(f"Support: {dist.info.support}")

**Step 3: Fit the Distribution**

.. code-block:: python

    # Fit using Maximum Likelihood Estimation
    dist.fit(data, method='mle')
    
    print(f"Fitted: {dist.fitted}")
    print(f"Parameters: {dist.params}")

**Step 4: View Summary**

.. code-block:: python

    # Complete statistical summary
    print(dist.summary())

Output::

    ╔══════════════════════════════════════════════════════════════╗
    ║            Normal (Gaussian) Distribution                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  📊 Estimated Parameters                                      ║
    ╚══════════════════════════════════════════════════════════════╝
       μ (mean)                         =        10.017342
       σ (std)                          =         1.991847
    
    ╔══════════════════════════════════════════════════════════════╗
    ║  📍 Location Statistics                                       ║
    ╚══════════════════════════════════════════════════════════════╝
       Mean                             =        10.017342
       Median                           =        10.017342
       Mode                             =        10.017342

Understanding the Output
------------------------

**Parameters Section**

Shows the fitted parameter values:

- ``μ (mean) = 10.017`` - location parameter
- ``σ (std) = 1.992`` - scale parameter

These are very close to the true values (10 and 2)!

**Statistics Sections**

- **Location**: mean, median, mode
- **Spread**: variance, standard deviation
- **Shape**: skewness, kurtosis
- **Quantiles**: key percentiles

Using the Fitted Distribution
------------------------------

**Generate Random Samples**

.. code-block:: python

    # Generate 100 new samples
    samples = dist.rvs(size=100, random_state=42)
    print(f"Generated samples: {samples[:5]}")

**Calculate Probabilities**

.. code-block:: python

    # PDF at x=10
    pdf_val = dist.pdf(np.array([10.0]))[0]
    print(f"PDF at x=10: {pdf_val:.4f}")
    
    # CDF at x=10
    cdf_val = dist.cdf(np.array([10.0]))[0]
    print(f"CDF at x=10: {cdf_val:.4f}")
    
    # P(X <= 12)
    prob = dist.cdf(np.array([12.0]))[0]
    print(f"P(X <= 12) = {prob:.4f}")

**Find Quantiles**

.. code-block:: python

    # 95th percentile
    q95 = dist.ppf(0.95)
    print(f"95th percentile: {q95:.2f}")
    
    # Median
    median = dist.ppf(0.5)
    print(f"Median: {median:.2f}")

List Available Distributions
-----------------------------

.. code-block:: python

    from distfit_pro import list_distributions
    from distfit_pro import list_continuous_distributions
    from distfit_pro import list_discrete_distributions
    
    print(f"All distributions ({len(list_distributions())}):")
    print(list_distributions())
    
    print(f"\nContinuous ({len(list_continuous_distributions())}):")
    print(list_continuous_distributions())
    
    print(f"\nDiscrete ({len(list_discrete_distributions())}):")
    print(list_discrete_distributions())

Explain the Distribution
-------------------------

.. code-block:: python

    # Conceptual explanation
    print(dist.explain())

Output::

    ╔══════════════════════════════════════════════════════════════╗
    ║            Normal (Gaussian) Distribution                    ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📊 Estimated Parameters:
       • μ (mean): 10.0173
       • σ (std): 1.9918
    
    💡 Practical Applications:
       • Measurement errors
       • Heights and weights in populations
       • Test scores
       • Signal noise
    
    🔍 Characteristics:
       • Symmetric bell-shaped curve
       • 68% of data within ±1σ
       • 95% of data within ±2σ

Next Steps
----------

- :doc:`02_distributions` - Explore all 30 distributions
- :doc:`03_fitting_methods` - Different estimation methods
- :doc:`quickstart` - More examples
