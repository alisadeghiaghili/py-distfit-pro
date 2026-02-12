Weighted Data
=============

Fitting with observation weights.

When to Use Weights
-------------------

**Survey Data:**

- Sampling weights
- Post-stratification weights
- Inverse probability weights

**Aggregated Data:**

- Frequency counts
- Grouped data

**Quality Indicators:**

- Measurement precision
- Reliability scores

Weighted Fitting
----------------

**Weighted MLE:**

.. code-block:: python

    from distfit_pro.core.weighted import WeightedFitting
    
    params = WeightedFitting.fit_weighted_mle(
        data, weights, dist
    )
    
    dist.params = params
    dist.fitted = True

**Weighted Moments:**

.. code-block:: python

    params = WeightedFitting.fit_weighted_moments(
        data, weights, dist
    )

Effective Sample Size
---------------------

**Formula:**

.. math::

   ESS = \frac{(\sum w_i)^2}{\sum w_i^2}

**Interpretation:** Equivalent unweighted sample size.

**Code:**

.. code-block:: python

    ess = WeightedFitting.effective_sample_size(weights)
    print(f"ESS: {ess:.1f} / {len(weights)}")

See :doc:`../tutorial/07_weighted_data`.
