#!/usr/bin/env python3
"""
Comparing Multiple Distributions
================================

Find the best distribution for your data by comparing candidates.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution

# ============================================================================
# Generate Example Data (Right-Skewed)
# ============================================================================

np.random.seed(42)

# Simulate delivery times (always positive, right-skewed)
data = np.random.lognormal(mean=3.0, sigma=0.5, size=1000)

print("="*70)
print("📦 EXAMPLE: Delivery Time Analysis")
print("="*70)
print(f"\nData: {len(data)} delivery times (hours)")
print(f"  Mean:    {data.mean():.2f} hours")
print(f"  Median:  {np.median(data):.2f} hours")
print(f"  Std:     {data.std():.2f} hours")
print(f"  Range:   [{data.min():.2f}, {data.max():.2f}]")


# ============================================================================
# Try Multiple Distribution Candidates
# ============================================================================

print("\n" + "="*70)
print("🔍 Testing Different Distribution Models")
print("="*70)

candidates = [
    'normal',      # Symmetric (Gaussian)
    'lognormal',   # Right-skewed (for positive data)
    'gamma',       # Flexible right-skewed
    'weibull_min', # Reliability/survival
    'expon',       # Simple exponential
]

results = []

for name in candidates:
    try:
        dist = get_distribution(name)
        dist.fit(data)
        
        # Calculate fit quality metrics
        aic = dist.aic()
        bic = dist.bic()
        ll = dist.log_likelihood()
        
        results.append({
            'name': name,
            'dist': dist,
            'aic': aic,
            'bic': bic,
            'log_likelihood': ll,
        })
        
        print(f"\n✓ {name:15s} AIC={aic:8.2f}  BIC={bic:8.2f}")
        
    except Exception as e:
        print(f"\n✗ {name:15s} Failed: {e}")


# ============================================================================
# Find Best Model (Lowest AIC/BIC)
# ============================================================================

print("\n" + "="*70)
print("🏆 Model Selection Results")
print("="*70)

# Sort by AIC (lower is better)
results_sorted = sorted(results, key=lambda x: x['aic'])

print("\n📊 Ranking by AIC (lower = better fit):")
print("\n   Rank  Distribution      AIC        BIC        Log-Likelihood")
print("   " + "-"*65)

for rank, result in enumerate(results_sorted, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"   {marker} {rank}.  {result['name']:15s}  "
          f"{result['aic']:8.2f}   {result['bic']:8.2f}   {result['log_likelihood']:10.2f}")

best = results_sorted[0]
print(f"\n✅ Best fit: {best['name']} (AIC = {best['aic']:.2f})")


# ============================================================================
# Analyze Best Distribution
# ============================================================================

print("\n" + "="*70)
print(f"📈 Best Model Analysis: {best['name'].upper()}")
print("="*70)

best_dist = best['dist']

print("\n" + best_dist.summary())

# Business insights
median_delivery = best_dist.median()
p95_delivery = best_dist.ppf(0.95)

print("\n" + "="*70)
print("💼 Business Insights")
print("="*70)
print(f"\n  📦 Typical delivery time (median): {median_delivery:.2f} hours")
print(f"  ⏱️  95% of deliveries within:     {p95_delivery:.2f} hours")
print(f"  🎯 Expected value (mean):          {best_dist.mean():.2f} hours")

# Calculate probability of late delivery (> 30 hours)
late_prob = 1 - best_dist.cdf(30)
print(f"\n  ⚠️  Probability of delay (>30h):    {late_prob*100:.2f}%")


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print(f"""
1. Always test multiple distributions for your data type
2. Use AIC/BIC to compare models objectively
3. The '{best['name']}' distribution fits this data best
4. Lower AIC = better balance of fit quality and simplicity

Next: See quick_visualization.py to plot these distributions!
""")
