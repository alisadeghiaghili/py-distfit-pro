Goodness-of-Fit Tests
=====================

Statistical tests for distribution fit quality.

Available Tests
---------------

Kolmogorov-Smirnov
^^^^^^^^^^^^^^^^^^

**Test Statistic:**

.. math::

   D = \max_x |F_n(x) - F(x)|

Where Fn is empirical CDF, F is theoretical CDF.

**Null Hypothesis:** Data comes from the specified distribution

**Interpretation:**

- P-value > 0.05: Accept (good fit)
- P-value ≤ 0.05: Reject (poor fit)

Anderson-Darling
^^^^^^^^^^^^^^^^

More sensitive to tail differences than KS test.

**Test Statistic:**

.. math::

   A^2 = -n - \sum_{i=1}^n \frac{2i-1}{n}[\ln F(x_i) + \ln(1-F(x_{n+1-i}))]

**When to use:** When tails are important (risk analysis, extreme values)

See :doc:`../tutorial/04_gof_tests` for examples.
