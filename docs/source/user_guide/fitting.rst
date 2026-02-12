Fitting Guide
=============

Complete guide to parameter estimation.

Estimation Methods
------------------

Maximum Likelihood
^^^^^^^^^^^^^^^^^^

Finds parameters that maximize P(data|parameters).

**Advantages:**

- Asymptotically efficient
- Consistent estimator
- Usually most accurate

**Code:**

.. code-block:: python

    dist.fit(data, method='mle')

Method of Moments
^^^^^^^^^^^^^^^^^

Matches sample moments to theoretical moments.

**Advantages:**

- Simple, fast
- Always converges
- Good initial estimates

**Code:**

.. code-block:: python

    dist.fit(data, method='moments')

See :doc:`../tutorial/03_fitting_methods` for more.
