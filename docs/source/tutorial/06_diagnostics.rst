Diagnostics
===========

Analyze fit quality in detail.

Residual Analysis
-----------------

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    residuals = Diagnostics.residual_analysis(data, dist)
    print(residuals.summary())

Outlier Detection
-----------------

.. code-block:: python

    # Z-score method
    outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
    print(f"Found {len(outliers.outlier_indices)} outliers")
    
    # IQR method
    outliers_iqr = Diagnostics.detect_outliers(data, dist, method='iqr')

Influence Diagnostics
---------------------

.. code-block:: python

    influence = Diagnostics.influence_diagnostics(data, dist)
    print(f"Influential points: {len(influence.influential_indices)}")
