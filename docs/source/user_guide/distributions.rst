Distributions Reference
=======================

Complete reference for all 30 distributions.

Continuous Distributions
------------------------

Normal Distribution
^^^^^^^^^^^^^^^^^^^

**PDF:**

.. math::

   f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

**Parameters:**

- ``loc`` (μ): mean
- ``scale`` (σ): standard deviation (σ > 0)

**Support:** x ∈ (-∞, ∞)

**Properties:**

- Mean = μ
- Variance = σ²
- Skewness = 0
- Kurtosis = 0

**When to use:** Symmetric data, natural phenomena, measurement errors

Weibull Distribution
^^^^^^^^^^^^^^^^^^^^

**PDF:**

.. math::

   f(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k}

**Parameters:**

- ``c`` (k): shape parameter
- ``scale`` (λ): scale parameter

**Support:** x ≥ 0

**Hazard Rate:**

- k < 1: Decreasing (infant mortality)
- k = 1: Constant (exponential)
- k > 1: Increasing (wear-out)

**When to use:** Reliability analysis, failure time data

See :doc:`../tutorial/02_distributions` for more details.
