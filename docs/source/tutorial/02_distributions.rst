Tutorial 2: Distributions Guide
================================

Complete guide to all 30 distributions.

Continuous Distributions
------------------------

Normal Distribution
^^^^^^^^^^^^^^^^^^^^

**Use when:** Data is symmetric, bell-shaped, no outliers

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    
    # Heights of adult males (cm)
    data = np.random.normal(175, 7, 1000)
    
    dist = get_distribution('normal')
    dist.fit(data)
    print(dist.summary())

**Parameters:**
- ``loc`` (μ): mean/center
- ``scale`` (σ): standard deviation

**When NOT to use:**
- Skewed data
- Heavy tails
- Bounded data (e.g., percentages)

Lognormal Distribution
^^^^^^^^^^^^^^^^^^^^^^

**Use when:** Data is positive, right-skewed

.. code-block:: python

    # Income data ($)
    data = np.random.lognormal(10, 0.5, 1000)
    
    dist = get_distribution('lognormal')
    dist.fit(data)
    print(dist.summary())

**Parameters:**
- ``s`` (σ): shape (log-scale std)
- ``scale`` (exp(μ)): scale

**Common applications:**
- Income/wealth
- Stock prices
- File sizes
- Particle sizes

Weibull Distribution
^^^^^^^^^^^^^^^^^^^^

**Use when:** Modeling time-to-failure, lifetimes

.. code-block:: python

    # Component lifetime (hours)
    data = np.random.weibull(1.5, 1000) * 1000
    
    dist = get_distribution('weibull')
    dist.fit(data)
    print(dist.summary())
    
    # Reliability at t=500 hours
    reliability = dist.reliability(500)
    print(f"Reliability at 500h: {reliability:.4f}")

**Parameters:**
- ``c`` (k): shape
  - k < 1: decreasing failure rate (infant mortality)
  - k = 1: constant failure rate (random failures)
  - k > 1: increasing failure rate (wear-out)
- ``scale`` (λ): scale

**Applications:**
- Reliability engineering
- Failure time analysis
- Wind speed modeling

Gamma Distribution
^^^^^^^^^^^^^^^^^^

**Use when:** Waiting times, sum of exponentials

.. code-block:: python

    # Waiting time for 5 events
    data = np.random.gamma(5, 2, 1000)
    
    dist = get_distribution('gamma')
    dist.fit(data)
    print(dist.summary())

**Parameters:**
- ``a`` (α): shape
- ``scale`` (θ): scale

**Special cases:**
- α = 1: Exponential distribution
- α = k/2, θ = 2: Chi-square with k df

Exponential Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

**Use when:** Time between events (memoryless)

.. code-block:: python

    # Time between arrivals (minutes)
    data = np.random.exponential(5, 1000)
    
    dist = get_distribution('exponential')
    dist.fit(data)
    print(dist.summary())
    
    # Probability of waiting < 3 minutes
    prob = dist.cdf(np.array([3]))[0]
    print(f"P(wait < 3 min) = {prob:.4f}")

**Key property:** Memoryless!

.. code-block:: python

    # P(X > 10 | X > 5) = P(X > 5)
    # Past doesn't affect future

Beta Distribution
^^^^^^^^^^^^^^^^^

**Use when:** Data is bounded between 0 and 1

.. code-block:: python

    # Success rates, percentages
    data = np.random.beta(2, 5, 1000)
    
    dist = get_distribution('beta')
    dist.fit(data)
    print(dist.summary())

**Parameters:**
- ``a`` (α): shape 1
- ``b`` (β): shape 2

**Applications:**
- Conversion rates
- Probabilities
- Proportions
- Bayesian priors

Pareto Distribution
^^^^^^^^^^^^^^^^^^^

**Use when:** Power-law, heavy tails, 80-20 rule

.. code-block:: python

    # Wealth distribution
    data = (np.random.pareto(2, 1000) + 1) * 50000
    
    dist = get_distribution('pareto')
    dist.fit(data)
    print(dist.summary())

**Applications:**
- Wealth/income distribution
- City sizes
- Word frequencies

Student's t Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

**Use when:** Small samples, heavier tails than normal

.. code-block:: python

    # Small sample data
    data = np.random.standard_t(5, 100)
    
    dist = get_distribution('studentt')
    dist.fit(data)
    print(dist.summary())

**Parameters:**
- ``df`` (ν): degrees of freedom
- As df → ∞, approaches Normal

Discrete Distributions
----------------------

Poisson Distribution
^^^^^^^^^^^^^^^^^^^^

**Use when:** Count of rare events in fixed interval

.. code-block:: python

    # Number of calls per hour
    data = np.random.poisson(lam=3.5, size=1000)
    
    dist = get_distribution('poisson')
    dist.fit(data)
    print(dist.summary())
    
    # P(exactly 5 calls)
    prob = dist.pdf(np.array([5]))[0]
    print(f"P(X = 5) = {prob:.4f}")

**Parameter:**
- ``mu`` (λ): rate (mean = variance)

**Applications:**
- Call center arrivals
- Website visitors
- Defects in manufacturing

Binomial Distribution
^^^^^^^^^^^^^^^^^^^^^

**Use when:** n independent yes/no trials

.. code-block:: python

    # 10 coin flips, p=0.5
    data = np.random.binomial(n=10, p=0.5, size=1000)
    
    dist = get_distribution('binomial')
    dist.fit(data)
    print(dist.summary())

**Parameters:**
- ``n``: number of trials
- ``p``: success probability

**Applications:**
- Quality control (pass/fail)
- Survey responses (yes/no)
- A/B testing

Negative Binomial
^^^^^^^^^^^^^^^^^

**Use when:** Overdispersed count data (variance > mean)

.. code-block:: python

    # Overdispersed counts
    data = np.random.negative_binomial(5, 0.5, 1000)
    
    dist = get_distribution('nbinom')
    dist.fit(data)
    print(dist.summary())

**Better than Poisson when:**
- Data shows more variability
- Clustering of events

Complete Distribution List
--------------------------

**Continuous (25):**

1. Normal - symmetric, bell curve
2. Lognormal - positive, right-skewed
3. Weibull - reliability, lifetimes
4. Gamma - waiting times
5. Exponential - time between events
6. Beta - bounded [0,1]
7. Uniform - equal probability
8. Triangular - three-point estimate
9. Logistic - growth models
10. Gumbel - extreme values (max)
11. Frechet - extreme values (positive)
12. Pareto - power law, 80-20
13. Cauchy - undefined mean/variance
14. Student's t - heavy tails
15. Chi-squared - variance tests
16. F - variance ratio
17. Rayleigh - signal processing
18. Laplace - sparse data
19. Inverse Gamma - Bayesian priors
20. Log-Logistic - survival analysis

**Discrete (5):**

1. Poisson - rare event counts
2. Binomial - n trials
3. Negative Binomial - overdispersed
4. Geometric - trials to first success
5. Hypergeometric - sampling without replacement

Choosing the Right Distribution
--------------------------------

**Decision Tree:**

1. **Is data discrete (counts) or continuous?**
   
   - Discrete → Poisson, Binomial, etc.
   - Continuous → continue

2. **Is data bounded?**
   
   - [0, 1] → Beta
   - [a, b] → Uniform, Triangular
   - [0, ∞) → Lognormal, Gamma, Weibull, Exponential
   - (-∞, ∞) → Normal, Logistic, Cauchy, Student's t

3. **Is data skewed?**
   
   - Right-skewed → Lognormal, Gamma, Weibull
   - Symmetric → Normal, Logistic, Student's t
   - Left-skewed → Reflected versions

4. **Heavy tails?**
   
   - Yes → Student's t, Cauchy, Pareto
   - No → Normal, Logistic

5. **Special domain?**
   
   - Reliability → Weibull, Exponential
   - Extreme values → Gumbel, Frechet
   - Survival → Weibull, Log-Logistic

Next Steps
----------

- :doc:`03_fitting_methods` - Different ways to fit
- :doc:`04_gof_tests` - Test if fit is good
