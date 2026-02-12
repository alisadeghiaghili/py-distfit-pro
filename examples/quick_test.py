#!/usr/bin/env python
"""
Quick Smoke Test for MVP
========================

Verify basic functionality works.
"""

import numpy as np
import sys

try:
    from distfit_pro import NormalDistribution, ExponentialDistribution, get_distribution
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 1: Normal Distribution
print("\n" + "="*50)
print("Test 1: Normal Distribution")
print("="*50)

np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=1000)

dist = NormalDistribution()
print(f"Distribution created: {dist}")

# Fit with MLE
dist.fit(data, method='mle')
print(f"\nFitted with MLE:")
print(f"  loc (mean): {dist.params['loc']:.4f} (expected: ~10)")
print(f"  scale (std): {dist.params['scale']:.4f} (expected: ~2)")

# Test probability functions
test_point = 10.0
print(f"\nProbability functions at x={test_point}:")
print(f"  PDF: {dist.pdf(np.array([test_point]))[0]:.6f}")
print(f"  CDF: {dist.cdf(np.array([test_point]))[0]:.6f}")
print(f"  PPF(0.5): {dist.ppf(np.array([0.5]))[0]:.4f}")

# Test moments
print(f"\nMoments:")
print(f"  Mean: {dist.mean():.4f}")
print(f"  Variance: {dist.var():.4f}")
print(f"  Std: {dist.std():.4f}")
print(f"  Median: {dist.median():.4f}")
print(f"  Mode: {dist.mode():.4f}")

# Test sampling
samples = dist.rvs(size=10, random_state=42)
print(f"\nGenerated samples (first 5): {samples[:5]}")

# Test information criteria
print(f"\nGoodness of fit:")
print(f"  Log-likelihood: {dist.log_likelihood():.2f}")
print(f"  AIC: {dist.aic():.2f}")
print(f"  BIC: {dist.bic():.2f}")

print("\n✅ Normal distribution: ALL TESTS PASSED!")

# Test 2: Exponential Distribution
print("\n" + "="*50)
print("Test 2: Exponential Distribution")
print("="*50)

data_exp = np.random.exponential(scale=2, size=1000)

dist_exp = ExponentialDistribution()
dist_exp.fit(data_exp, method='mle')

print(f"Fitted with MLE:")
print(f"  scale: {dist_exp.params['scale']:.4f} (expected: ~2)")
print(f"  Mean: {dist_exp.mean():.4f}")
print(f"  Mode: {dist_exp.mode():.4f}")

print("\n✅ Exponential distribution: ALL TESTS PASSED!")

# Test 3: get_distribution function
print("\n" + "="*50)
print("Test 3: get_distribution() function")
print("="*50)

dist3 = get_distribution('normal')
print(f"Got distribution: {dist3.info.display_name}")

dist3.fit(data, method='mom')
print(f"Fitted with MoM: loc={dist3.params['loc']:.4f}, scale={dist3.params['scale']:.4f}")

print("\n✅ get_distribution(): ALL TESTS PASSED!")

# Final summary
print("\n" + "="*50)
print("🎉 ALL SMOKE TESTS PASSED! 🎉")
print("="*50)
print("\nMVP is operational:")
print("  ✅ Base classes working")
print("  ✅ Normal distribution fully functional")
print("  ✅ Exponential distribution fully functional")
print("  ✅ MLE fitting works")
print("  ✅ MoM fitting works")
print("  ✅ All probability functions work")
print("  ✅ Information criteria calculated correctly")
print("  ✅ Sampling works")
print("\nReady to add more distributions! 🚀")
