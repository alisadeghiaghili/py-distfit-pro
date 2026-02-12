Diagnostic Tools
================

Assess fit quality beyond GOF tests.

Residual Analysis
-----------------

Residual Types
^^^^^^^^^^^^^^

1. **Quantile Residuals** - randomized, should be N(0,1)
2. **Pearson Residuals** - standardized differences
3. **Deviance Residuals** - based on likelihood
4. **Standardized Residuals** - normalized

**Code:**

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    residuals = Diagnostics.residual_analysis(data, dist)
    print(residuals.summary())

Outlier Detection
-----------------

Methods
^^^^^^^

1. **Z-score** - |z| > threshold (default: 3)
2. **IQR** - outside [Q1-1.5*IQR, Q3+1.5*IQR]
3. **Likelihood** - very low PDF value
4. **Mahalanobis** - distance-based

**Code:**

.. code-block:: python

    outliers = Diagnostics.detect_outliers(
        data, dist, 
        method='zscore',
        threshold=2.5
    )
    
    print(f"Found: {len(outliers.outlier_indices)}")

See :doc:`../tutorial/06_diagnostics`.
