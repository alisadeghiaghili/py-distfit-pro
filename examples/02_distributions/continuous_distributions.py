#!/usr/bin/env python3
"""
Continuous Distributions Gallery
=================================

Demonstrates all 20 continuous distributions available in distfit_pro.

For each distribution, we:
1. Generate synthetic data
2. Fit the distribution
3. Show fitted parameters
4. Display summary statistics
"""

import numpy as np
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("CONTINUOUS DISTRIBUTIONS GALLERY")
print("="*70)
print(f"\nShowing all {20} continuous distributions...\n")

# =============================================================================
# Distribution definitions with parameters for data generation
# =============================================================================
continuous_dists = [
    ('normal', {'loc': 0, 'scale': 1}, "Most common, symmetric, bell-shaped"),
    ('exponential', {'scale': 2.0}, "Memoryless, time between events"),
    ('gamma', {'a': 2, 'scale': 2}, "Waiting times, positive skew"),
    ('beta', {'a': 2, 'b': 5}, "Proportions, bounded [0,1]"),
    ('weibull_min', {'c': 1.5, 'scale': 1}, "Reliability, failure times"),
    ('lognormal', {'s': 0.5, 'scale': np.exp(0)}, "Right-skewed, multiplicative processes"),
    ('chi2', {'df': 5}, "Sum of squared normals"),
    ('t', {'df': 10}, "Heavy tails, small samples"),
    ('f', {'dfn': 5, 'dfd': 10}, "Ratio of variances"),
    ('uniform', {'loc': 0, 'scale': 1}, "Equal probability, simple"),
    ('cauchy', {'loc': 0, 'scale': 1}, "Very heavy tails, no moments"),
    ('laplace', {'loc': 0, 'scale': 1}, "Double exponential"),
    ('logistic', {'loc': 0, 'scale': 1}, "S-shaped CDF, logistic regression"),
    ('pareto', {'b': 2.5}, "Power law, wealth distribution"),
    ('rayleigh', {'scale': 1}, "Wind speed, magnitude of 2D vector"),
    ('triangular', {'c': 0.5, 'loc': 0, 'scale': 1}, "Simple, bounded"),
    ('gumbel_r', {'loc': 0, 'scale': 1}, "Extreme values (maximum)"),
    ('gumbel_l', {'loc': 0, 'scale': 1}, "Extreme values (minimum)"),
    ('frechet_r', {'c': 2, 'scale': 1}, "Extreme value type II"),
    ('invgauss', {'mu': 1, 'scale': 1}, "Inverse Gaussian, Brownian motion"),
]

print(f"Sample size for each: 500 observations\n")

# =============================================================================
# Fit each distribution
# =============================================================================
for i, (name, params, description) in enumerate(continuous_dists, 1):
    print("="*70)
    print(f"{i}. {name.upper()}")
    print("="*70)
    print(f"Description: {description}")
    print("-"*70)
    
    try:
        # Generate data using scipy directly
        from scipy import stats
        scipy_dist = getattr(stats, name)
        data = scipy_dist.rvs(size=500, **params)
        
        # Fit our distribution
        dist = get_distribution(name)
        dist.fit(data)
        
        # Show key results
        print("\nFitted Parameters:")
        for param, value in dist.params.items():
            print(f"  {param}: {value:.6f}")
        
        print("\nSummary Statistics:")
        try:
            print(f"  Mean: {dist.mean():.4f}")
            print(f"  Std: {dist.std():.4f}")
            print(f"  Median: {dist.median():.4f}")
        except:
            print("  (Some statistics not available)")
        
        print("\nGoodness of Fit:")
        print(f"  Log-Likelihood: {dist.log_likelihood():.2f}")
        print(f"  AIC: {dist.aic():.2f}")
        print(f"  BIC: {dist.bic():.2f}")
        
        print("\n✓ Success")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print()

# =============================================================================
# Summary Table
# =============================================================================
print("\n" + "="*70)
print("QUICK REFERENCE GUIDE")
print("="*70)
print()
print(f"{'Distribution':<20} {'Use Case':<45}")
print("-"*70)

for name, _, description in continuous_dists:
    print(f"{name:<20} {description:<45}")

print("\n" + "="*70)
print("NOTES")
print("="*70)
print("""
1. All distributions can be fitted using:
   dist = get_distribution('name')
   dist.fit(data)

2. Some distributions have constraints:
   - Beta: data must be in [0, 1]
   - Exponential/Gamma/Weibull: data must be positive
   - Pareto: data must be >= 1

3. Distribution selection tips:
   - Symmetric data → Normal, t, Cauchy, Logistic
   - Right-skewed → Gamma, Weibull, Lognormal
   - Bounded data → Beta, Uniform, Triangular
   - Heavy tails → t, Cauchy, Pareto
   - Extreme values → Gumbel, Frechet

4. Use AIC/BIC for model comparison (see multiple_distributions.py)
""")
print("="*70)
