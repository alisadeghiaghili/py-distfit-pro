#!/usr/bin/env python3
"""
Simple Distribution Fitting
============================

This example demonstrates the most basic workflow:
1. Generate or load data
2. Fit a distribution
3. View results

Perfect starting point for new users.
"""

import numpy as np
from distfit_pro import get_distribution

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("SIMPLE DISTRIBUTION FITTING EXAMPLE")
print("="*70)

# =============================================================================
# STEP 1: Generate sample data
# =============================================================================
print("\n[1] Generating sample data...")
print("-" * 70)

# Generate 1000 samples from normal distribution
# True parameters: mean=10, std=2
true_mean = 10
true_std = 2
data = np.random.normal(true_mean, true_std, 1000)

print(f"Generated {len(data)} samples")
print(f"True parameters: μ={true_mean}, σ={true_std}")
print(f"Sample mean: {np.mean(data):.4f}")
print(f"Sample std: {np.std(data, ddof=1):.4f}")

# =============================================================================
# STEP 2: Create distribution and fit
# =============================================================================
print("\n[2] Fitting normal distribution...")
print("-" * 70)

# Get a normal distribution object
dist = get_distribution('normal')

# Fit to data using Maximum Likelihood Estimation (default)
dist.fit(data)

print("✓ Fitting complete!")

# =============================================================================
# STEP 3: View results
# =============================================================================
print("\n[3] Fitted parameters:")
print("-" * 70)

# Get fitted parameters
params = dist.params
print(f"Estimated μ (loc): {params['loc']:.4f}")
print(f"Estimated σ (scale): {params['scale']:.4f}")

print(f"\nError in μ: {abs(params['loc'] - true_mean):.4f}")
print(f"Error in σ: {abs(params['scale'] - true_std):.4f}")

# =============================================================================
# STEP 4: Display comprehensive summary
# =============================================================================
print("\n[4] Full distribution summary:")
print("-" * 70)
print()
print(dist.summary())

# =============================================================================
# STEP 5: Use the fitted distribution
# =============================================================================
print("\n[5] Using the fitted distribution:")
print("-" * 70)

# Calculate probabilities
test_point = 12.0
pdf_value = dist.pdf(test_point)
cdf_value = dist.cdf(test_point)

print(f"PDF at x={test_point}: {pdf_value:.6f}")
print(f"CDF at x={test_point}: {cdf_value:.6f}")
print(f"P(X > {test_point}): {1 - cdf_value:.6f}")

# Generate new samples from fitted distribution
new_samples = dist.rvs(5, random_state=123)
print(f"\nGenerated samples from fitted distribution:")
for i, sample in enumerate(new_samples, 1):
    print(f"  Sample {i}: {sample:.4f}")

# Calculate percentiles
percentiles = [25, 50, 75, 95]
print(f"\nPercentiles:")
for p in percentiles:
    value = dist.ppf(p/100)
    print(f"  {p}th percentile: {value:.4f}")

# =============================================================================
# STEP 6: Model quality metrics
# =============================================================================
print("\n[6] Model quality:")
print("-" * 70)

log_likelihood = dist.log_likelihood()
aic = dist.aic()
bic = dist.bic()

print(f"Log-likelihood: {log_likelihood:.2f}")
print(f"AIC: {aic:.2f} (lower is better)")
print(f"BIC: {bic:.2f} (lower is better)")

print("\n" + "="*70)
print("✓ Example complete!")
print("="*70)
print("\nNext steps:")
print("  - Try different distributions: 'exponential', 'gamma', 'weibull'")
print("  - Compare multiple distributions (see multiple_distributions.py)")
print("  - Create visualizations (see visualization_basics.py)")
print("="*70)
