#!/usr/bin/env python3
"""
Simple Distribution Fitting Example
===================================

Learn the basics: fit a distribution to your data in 3 lines of code.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution

# ============================================================================
# EXAMPLE 1: Your First Fit (Normal Distribution)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Fitting Normal Distribution to Data")
print("="*70)

# Generate some example data (in real life, this would be your measurements)
data = np.random.normal(loc=100, scale=15, size=1000)

print(f"\n📊 Data: {len(data)} measurements")
print(f"   Mean: {data.mean():.2f}")
print(f"   Std:  {data.std():.2f}")

# Fit a normal distribution (just 2 lines!)
dist = get_distribution('normal')
dist.fit(data)

print("\n✅ Distribution fitted!")
print(dist.summary())


# ============================================================================
# EXAMPLE 2: What Can You Do With a Fitted Distribution?
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Using Your Fitted Distribution")
print("="*70)

# Calculate probabilities
x = 115  # Some value
prob_density = dist.pdf(x)
prob_cumulative = dist.cdf(x)

print(f"\n📈 At x = {x}:")
print(f"   PDF (probability density): {prob_density:.6f}")
print(f"   CDF (cumulative prob):     {prob_cumulative:.6f}")
print(f"   → {prob_cumulative*100:.1f}% of data is below {x}")

# Calculate percentiles (quantiles)
percentile_95 = dist.ppf(0.95)  # 95th percentile
print(f"\n📊 Percentiles:")
print(f"   95th percentile: {percentile_95:.2f}")
print(f"   → 95% of values are below {percentile_95:.2f}")

# Generate random samples from fitted distribution
samples = dist.rvs(size=10, random_state=42)
print(f"\n🎲 Random samples from fitted distribution:")
print(f"   {samples}")


# ============================================================================
# EXAMPLE 3: Try Different Distributions
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Trying Different Distributions")
print("="*70)

# Generate exponential data (e.g., time between events)
data_exp = np.random.exponential(scale=5.0, size=1000)

print(f"\n📊 New data (exponential): {len(data_exp)} measurements")

# Fit exponential distribution
dist_exp = get_distribution('expon')
dist_exp.fit(data_exp)

print("\n✅ Exponential distribution fitted!")
print(f"   Mean (λ⁻¹): {dist_exp.mean():.2f}")
print(f"   Std:        {dist_exp.std():.2f}")

# Compare models using AIC (lower is better)
print("\n🎯 Model Comparison (which fits better?):")
aic_normal = get_distribution('normal').fit(data_exp).aic()
aic_exp = dist_exp.aic()

print(f"   Normal AIC:      {aic_normal:.2f}")
print(f"   Exponential AIC: {aic_exp:.2f}")
print(f"   → {'Exponential' if aic_exp < aic_normal else 'Normal'} fits better!")


print("\n" + "="*70)
print("🎉 Done! You've learned the basics of distfit-pro")
print("="*70)
print("\nNext steps:")
print("  - Check out distribution_comparison.py for model selection")
print("  - See quick_visualization.py for plotting")
print("  - Explore advanced examples in 03_advanced_fitting/")
