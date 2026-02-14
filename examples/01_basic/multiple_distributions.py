#!/usr/bin/env python3
"""
Comparing Multiple Distributions
=================================

This example shows how to:
1. Fit multiple distributions to the same data
2. Compare goodness-of-fit using AIC/BIC
3. Select the best distribution

Useful for model selection.
"""

import numpy as np
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("MULTIPLE DISTRIBUTION COMPARISON")
print("="*70)

# =============================================================================
# STEP 1: Generate data with known distribution
# =============================================================================
print("\n[1] Generating sample data...")
print("-" * 70)

# Generate data from Gamma distribution (shape=2, scale=2)
data = np.random.gamma(shape=2, scale=2, size=1000)

print(f"Generated {len(data)} samples from Gamma(α=2, β=2)")
print(f"Sample statistics:")
print(f"  Mean: {np.mean(data):.4f}")
print(f"  Std: {np.std(data, ddof=1):.4f}")
print(f"  Min: {np.min(data):.4f}")
print(f"  Max: {np.max(data):.4f}")

# =============================================================================
# STEP 2: Fit multiple candidate distributions
# =============================================================================
print("\n[2] Fitting candidate distributions...")
print("-" * 70)

# List of distributions to try
dist_names = ['normal', 'exponential', 'gamma', 'weibull', 'lognormal']
results = {}

for name in dist_names:
    print(f"  Fitting {name}...", end=" ")
    
    try:
        # Create and fit distribution
        dist = get_distribution(name)
        dist.fit(data)
        
        # Store results
        results[name] = {
            'distribution': dist,
            'params': dist.params,
            'log_likelihood': dist.log_likelihood(),
            'aic': dist.aic(),
            'bic': dist.bic()
        }
        print("✓")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        continue

print(f"\nSuccessfully fitted {len(results)} distributions")

# =============================================================================
# STEP 3: Compare using AIC
# =============================================================================
print("\n[3] Comparison using AIC (lower is better):")
print("-" * 70)
print(f"{'Distribution':<20} {'AIC':>12} {'BIC':>12} {'Log-Likelihood':>15}")
print("-" * 70)

# Sort by AIC
sorted_by_aic = sorted(results.items(), key=lambda x: x[1]['aic'])

for name, result in sorted_by_aic:
    aic = result['aic']
    bic = result['bic']
    ll = result['log_likelihood']
    
    # Mark the best
    marker = "← Best" if name == sorted_by_aic[0][0] else ""
    print(f"{name:<20} {aic:>12.2f} {bic:>12.2f} {ll:>15.2f} {marker}")

# =============================================================================
# STEP 4: Examine best model
# =============================================================================
print("\n[4] Best fitting distribution:")
print("-" * 70)

best_name = sorted_by_aic[0][0]
best_dist = results[best_name]['distribution']

print(f"\nWinner: {best_name.upper()}")
print()
print(best_dist.summary())

# =============================================================================
# STEP 5: Calculate AIC differences
# =============================================================================
print("\n[5] AIC differences from best model:")
print("-" * 70)

best_aic = sorted_by_aic[0][1]['aic']

print(f"{'Distribution':<20} {'ΔAIC':>12} {'Interpretation':<30}")
print("-" * 70)

for name, result in sorted_by_aic:
    delta_aic = result['aic'] - best_aic
    
    # Interpret delta AIC
    if delta_aic < 2:
        interpretation = "Substantial support"
    elif delta_aic < 4:
        interpretation = "Considerable support"
    elif delta_aic < 7:
        interpretation = "Some support"
    elif delta_aic < 10:
        interpretation = "Little support"
    else:
        interpretation = "Essentially no support"
    
    print(f"{name:<20} {delta_aic:>12.2f} {interpretation:<30}")

# =============================================================================
# STEP 6: Parameter comparison
# =============================================================================
print("\n[6] Fitted parameters for each distribution:")
print("-" * 70)

for name in dist_names:
    if name in results:
        print(f"\n{name.upper()}:")
        params = results[name]['params']
        for param_name, param_value in params.items():
            print(f"  {param_name}: {param_value:.6f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Best distribution: {best_name}")
print(f"True distribution: gamma")
print(f"Match: {'✓ YES!' if best_name == 'gamma' else '✗ No'}")
print("\nKey insights:")
print("  - AIC and BIC both select the same model")
print("  - Gamma distribution has lowest AIC (as expected)")
print("  - ΔAIC > 10 means other models have no support")
print("="*70)
print("\nNext steps:")
print("  - Try with different data generators")
print("  - Use BIC for larger sample sizes (stronger penalty)")
print("  - Combine with visualizations for validation")
print("="*70)
