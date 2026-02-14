#!/usr/bin/env python3
"""
Discrete Distributions Gallery
===============================

Demonstrates all 5 discrete distributions available in distfit_pro.

Discrete distributions are for count data:
- Number of events
- Integer-valued outcomes
- Success/failure trials
"""

import numpy as np
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("DISCRETE DISTRIBUTIONS GALLERY")
print("="*70)
print(f"\nShowing all 5 discrete distributions...\n")

# =============================================================================
# Distribution definitions
# =============================================================================
discrete_dists = [
    ('poisson', 
     {'mu': 5},
     "Number of events in fixed time/space",
     "Rare events, call center arrivals, website visits"),
    
    ('binom', 
     {'n': 20, 'p': 0.3},
     "Number of successes in n trials",
     "Quality control, survey responses, coin flips"),
    
    ('nbinom', 
     {'n': 5, 'p': 0.4},
     "Number of failures before r successes",
     "Insurance claims, repeat purchases"),
    
    ('geom', 
     {'p': 0.2},
     "Number of trials to first success",
     "Time to first sale, first failure"),
    
    ('hypergeom',
     {'M': 50, 'n': 10, 'N': 15},
     "Sampling without replacement",
     "Quality inspection, lottery, defect detection"),
]

print(f"Sample size for each: 500 observations\n")

# =============================================================================
# Fit each distribution
# =============================================================================
for i, (name, params, short_desc, use_cases) in enumerate(discrete_dists, 1):
    print("="*70)
    print(f"{i}. {name.upper()}")
    print("="*70)
    print(f"Description: {short_desc}")
    print(f"Use cases: {use_cases}")
    print("-"*70)
    
    try:
        # Generate data
        from scipy import stats
        scipy_dist = getattr(stats, name)
        data = scipy_dist.rvs(size=500, **params)
        
        # Fit distribution
        dist = get_distribution(name)
        dist.fit(data)
        
        # Show key results
        print("\nFitted Parameters:")
        for param, value in dist.params.items():
            print(f"  {param}: {value:.6f}")
        
        print("\nData Characteristics:")
        print(f"  Range: [{int(data.min())}, {int(data.max())}]")
        print(f"  Sample mean: {np.mean(data):.4f}")
        print(f"  Sample variance: {np.var(data, ddof=1):.4f}")
        print(f"  Var/Mean ratio: {np.var(data, ddof=1)/np.mean(data):.4f}")
        
        print("\nFitted Distribution Statistics:")
        try:
            fitted_mean = dist.mean()
            fitted_var = dist.var()
            print(f"  Mean: {fitted_mean:.4f}")
            print(f"  Variance: {fitted_var:.4f}")
            print(f"  Std: {dist.std():.4f}")
            
            # For discrete, show mode as most probable value
            try:
                mode_val = dist.mode()
                print(f"  Mode: {int(mode_val)}")
            except:
                pass
        except:
            print("  (Some statistics not available)")
        
        print("\nProbability Mass Function (PMF) Examples:")
        # Show PMF for a few values
        unique_values = np.sort(np.unique(data))[:5]  # First 5 unique values
        for val in unique_values:
            pmf_val = dist.pdf(val)  # For discrete, pdf returns PMF
            print(f"  P(X = {int(val)}) = {pmf_val:.6f}")
        
        print("\nGoodness of Fit:")
        print(f"  Log-Likelihood: {dist.log_likelihood():.2f}")
        print(f"  AIC: {dist.aic():.2f}")
        print(f"  BIC: {dist.bic():.2f}")
        
        print("\n✓ Success")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print()

# =============================================================================
# Variance-to-Mean Ratio Interpretation
# =============================================================================
print("\n" + "="*70)
print("VARIANCE-TO-MEAN RATIO GUIDE")
print("="*70)
print("""
The variance-to-mean ratio helps select discrete distributions:

1. Ratio ≈ 1 (Equidispersion):
   → Use POISSON distribution
   - Events occur independently
   - Constant rate

2. Ratio > 1 (Overdispersion):
   → Use NEGATIVE BINOMIAL
   - More variability than Poisson
   - Events are clustered

3. Ratio < 1 (Underdispersion):
   → Use BINOMIAL
   - Less variability than Poisson
   - Fixed number of trials

4. Sampling without replacement:
   → Use HYPERGEOMETRIC
   - Finite population
   - No replacement

5. Waiting time to event:
   → Use GEOMETRIC
   - Memoryless property
   - First success
""")

# =============================================================================
# Quick Reference
# =============================================================================
print("\n" + "="*70)
print("QUICK REFERENCE")
print("="*70)
print()
print(f"{'Distribution':<15} {'Parameters':<25} {'Mean':<20}")
print("-"*70)
print(f"{'Poisson':<15} {'λ (rate)':<25} {'λ':<20}")
print(f"{'Binomial':<15} {'n (trials), p (prob)':<25} {'n·p':<20}")
print(f"{'Negative Binom':<15} {'r (successes), p':<25} {'r(1-p)/p':<20}")
print(f"{'Geometric':<15} {'p (success prob)':<25} {'1/p':<20}")
print(f"{'Hypergeometric':<15} {'N, K, n (pop, good, draw)':<25} {'n·K/N':<20}")

print("\n" + "="*70)
print("EXAMPLES")
print("="*70)
print("""
1. Poisson → Number of emails per hour
   data = [3, 5, 4, 6, 2, 4, 5, 3, ...]
   dist = get_distribution('poisson')
   dist.fit(data)

2. Binomial → Number of defective items in batch of 100
   data = [2, 1, 3, 0, 2, 1, ...]
   dist = get_distribution('binom')
   dist.fit(data)

3. Negative Binomial → Sales attempts before 3 successes
   data = [5, 8, 6, 10, 7, ...]
   dist = get_distribution('nbinom')
   dist.fit(data)
""")
print("="*70)
