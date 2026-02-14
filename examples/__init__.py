"""DistFit-Pro Examples Package.

Comprehensive examples demonstrating all features of distfit-pro.

Package Structure:
-----------------
01_basics/
    - basic_fitting.py: Introduction to distribution fitting
    - common_distributions.py: Standard distributions overview

02_advanced_fitting/
    - maximum_likelihood.py: MLE with custom optimization
    - method_of_moments.py: MoM estimation

03_model_selection/
    - aic_bic_comparison.py: Information criteria
    - cross_validation.py: CV for distribution selection
    - hypothesis_testing.py: Goodness-of-fit tests

04_goodness_of_fit/
    - ks_test.py: Kolmogorov-Smirnov test
    - chi_square_test.py: Chi-square test
    - anderson_darling.py: Anderson-Darling test

05_visualization/
    - pdf_cdf_plots.py: Probability plots
    - qq_pp_plots.py: Quantile-quantile plots
    - interactive_plots.py: Interactive dashboards

06_real_world/
    - finance_analysis.py: Risk analysis, VaR
    - reliability_engineering.py: Failure analysis, Weibull
    - quality_control.py: SPC, process capability

07_advanced_topics/
    - mixture_models.py: Gaussian mixture models
    - bootstrap_confidence.py: Bootstrap CI
    - custom_distributions.py: Create custom distributions

Quick Start:
-----------
    # Basic fitting
    from distfit_pro import get_distribution
    import numpy as np
    
    data = np.random.normal(100, 15, 1000)
    dist = get_distribution('normal')
    dist.fit(data)
    
    print(f"Mean: {dist.mean():.2f}")
    print(f"Std: {dist.std():.2f}")

Learning Path:
-------------
1. Start with 01_basics/ for introduction
2. Move to 03_model_selection/ for choosing distributions
3. Explore 05_visualization/ for understanding fits
4. Study 06_real_world/ for practical applications
5. Master 07_advanced_topics/ for complex scenarios

Author: Ali Sadeghi Aghili
License: MIT
"""

__version__ = '1.0.0'
__author__ = 'Ali Sadeghi Aghili'

# Import key examples for easy access
__all__ = [
    'basics',
    'advanced_fitting',
    'model_selection',
    'goodness_of_fit',
    'visualization',
    'real_world',
    'advanced_topics',
]
