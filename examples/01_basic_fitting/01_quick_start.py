"""Quick Start - 5 Minute Introduction to distfit-pro

This is the fastest way to get started with distfit-pro.
You'll learn how to:
1. Fit a distribution to data
2. Get parameter estimates
3. Generate predictions
4. Create visualizations

Perfect for: First-time users, quick prototyping
Time: ~5 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

# Generate some example data (normal distribution)
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)

print("="*70)
print("🚀 QUICK START: distfit-pro in 5 Minutes")
print("="*70)

# ============================================================================
# STEP 1: Create a distribution object
# ============================================================================
print("\n📊 Step 1: Create a distribution object")
print("-" * 70)

dist = get_distribution('normal')
print(f"✓ Created: {dist.info.display_name}")
print(f"  Description: {dist.info.description}")

# ============================================================================
# STEP 2: Fit to your data
# ============================================================================
print("\n📈 Step 2: Fit distribution to data")
print("-" * 70)

dist.fit(data)
print("✓ Fitting completed!")
print(f"  Sample size: {len(data)} points")
print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")

# ============================================================================
# STEP 3: View results
# ============================================================================
print("\n📋 Step 3: View fitted parameters")
print("-" * 70)

params = dist.params
print("\nEstimated parameters:")
for name, value in params.items():
    print(f"  {name:>10} = {value:>10.4f}")

print("\nCompare with true values:")
print(f"  {'True μ':>10} = {100:>10.4f}")
print(f"  {'True σ':>10} = {15:>10.4f}")

# ============================================================================
# STEP 4: Make predictions
# ============================================================================
print("\n🎯 Step 4: Make predictions")
print("-" * 70)

# Probability that a value is less than 110
prob = dist.cdf(110)
print(f"\nP(X ≤ 110) = {prob:.4f} ({prob*100:.2f}%)")

# What value has 95% of data below it?
quantile_95 = dist.ppf(0.95)
print(f"95th percentile = {quantile_95:.2f}")

# Generate new random samples
samples = dist.rvs(size=5, random_state=42)
print(f"\n5 random samples: {samples.round(2)}")

# ============================================================================
# STEP 5: Visualize
# ============================================================================
print("\n📊 Step 5: Create visualization")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Histogram + PDF
ax1 = axes[0]
ax1.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', 
         edgecolor='black', label='Observed Data')

# Plot fitted PDF
x = np.linspace(data.min(), data.max(), 200)
ax1.plot(x, dist.pdf(x), 'r-', linewidth=2, label='Fitted Normal')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('Histogram + Fitted Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q Plot
ax2 = axes[1]
from scipy import stats as sp_stats
sp_stats.probplot(data, dist='norm', sparams=(params['loc'], params['scale']), plot=ax2)
ax2.set_title('Q-Q Plot (Normality Check)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_start_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: quick_start_results.png")

# ============================================================================
# BONUS: Model Quality Assessment
# ============================================================================
print("\n✅ Bonus: Model Quality Metrics")
print("-" * 70)

print(f"\nLog-Likelihood: {dist.log_likelihood():.2f}")
print(f"AIC: {dist.aic():.2f}")
print(f"BIC: {dist.bic():.2f}")
print("\n(Lower AIC/BIC = better fit)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 SUCCESS! You've completed the quick start.")
print("="*70)
print("\nWhat you learned:")
print("  ✓ Create distribution objects with get_distribution()")
print("  ✓ Fit distributions with .fit(data)")
print("  ✓ Access parameters with .params")
print("  ✓ Make predictions with .cdf(), .ppf(), .rvs()")
print("  ✓ Create visualizations with matplotlib")
print("  ✓ Assess model quality with AIC/BIC")
print("\nNext steps:")
print("  → Try 02_basic_fitting.py for detailed workflow")
print("  → Try 03_comparing_distributions.py for model selection")
print("  → Check docs/ for advanced features")
print("="*70)
