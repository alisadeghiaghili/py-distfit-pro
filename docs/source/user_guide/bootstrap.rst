Bootstrap Methods
=================

Uncertainty quantification via resampling.

Parametric Bootstrap
--------------------

**Algorithm:**

1. Fit distribution to data
2. Generate new sample from fitted distribution
3. Refit distribution
4. Repeat N times
5. Calculate percentiles

**Confidence Interval:**

- 95% CI = [2.5th percentile, 97.5th percentile]

**Code:**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    ci = Bootstrap.parametric(data, dist, n_bootstrap=1000)

Non-Parametric Bootstrap
------------------------

**Algorithm:**

1. Resample data with replacement
2. Fit distribution
3. Repeat N times
4. Calculate percentiles

**More robust when model is uncertain.**

See :doc:`../tutorial/05_bootstrap`.
