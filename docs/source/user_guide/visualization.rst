Visualization
=============

Plotting fitted distributions.

Basic Plots
-----------

PDF/PMF Plot
^^^^^^^^^^^^

.. code-block:: python

    from distfit_pro.plotting import plot_fit
    
    plot_fit(dist, data, plot_type='pdf')

**Shows:**

- Data histogram
- Fitted PDF curve
- Distribution name

Q-Q Plot
^^^^^^^^

.. code-block:: python

    plot_fit(dist, data, plot_type='qq')

**Interpretation:**

- Points on diagonal = good fit
- Curve above diagonal = right-skewed
- Curve below diagonal = left-skewed

Customization
-------------

.. code-block:: python

    plot_fit(
        dist, data,
        plot_type='pdf',
        backend='plotly',  # or 'matplotlib'
        bins=30,
        color='blue',
        alpha=0.7
    )

See :doc:`../tutorial/08_visualization`.
