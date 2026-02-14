#!/usr/bin/env python3
"""
MLE vs Method of Moments Comparison
===================================

Compare two parameter estimation methods:
  - Maximum Likelihood Estimation (MLE): Most accurate
  - Method of Moments (MoM): Faster, more robust

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
import time

np.random.seed(42)

print("="*70)
print("⚔️ MLE vs METHOD OF MOMENTS")
print("="*70)


# ============================================================================
# Theory Overview
# ============================================================================

print("\n" + "="*70)
print("📚 Method Overview")
print("="*70)
print("""
──────────────────────────────────────────────────────
MLE (Maximum Likelihood Estimation):
──────────────────────────────────────────────────────
  • Finds parameters that maximize P(data | parameters)
  • Statistically optimal (minimum variance estimator)
  • Requires optimization (slower)
  • Can fail on difficult data (outliers, small samples)
  
  ✅ Best for: Large datasets, well-behaved data
  ⚠️  Watch out: Can be slow, may not converge

──────────────────────────────────────────────────────
Method of Moments (MoM):
──────────────────────────────────────────────────────
  • Matches sample moments (mean, variance) to theoretical
  • Simple closed-form formulas (very fast)
  • Always converges (no optimization)
  • Less efficient (higher variance)
  
  ✅ Best for: Quick estimates, small samples, outliers
  ⚠️  Watch out: Less accurate than MLE
  
──────────────────────────────────────────────────────
""")


# ============================================================================
# Example 1: Normal Distribution (Large Sample)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Normal Distribution - Large Sample")
print("="*70)

n_large = 10000
data_large = np.random.normal(loc=100, scale=15, size=n_large)

print(f"\n📊 Data: {n_large} samples from N(100, 15)")

# MLE
print("\n1️⃣ MLE (Maximum Likelihood):")
start = time.time()
dist_mle = get_distribution('normal')
dist_mle.fit(data_large, method='mle')
time_mle = time.time() - start

print(f"  Time: {time_mle*1000:.2f} ms")
print(f"  μ (loc):   {dist_mle.params['loc']:.4f}")
print(f"  σ (scale): {dist_mle.params['scale']:.4f}")
print(f"  Log-likelihood: {dist_mle.log_likelihood():.2f}")

# MoM
print("\n2️⃣ Method of Moments:")
start = time.time()
dist_mom = get_distribution('normal')
dist_mom.fit(data_large, method='mom')
time_mom = time.time() - start

print(f"  Time: {time_mom*1000:.2f} ms")
print(f"  μ (loc):   {dist_mom.params['loc']:.4f}")
print(f"  σ (scale): {dist_mom.params['scale']:.4f}")
print(f"  Log-likelihood: {dist_mom.log_likelihood():.2f}")

# Comparison
print(f"\n🎯 Comparison (large sample):")
print(f"  Speed: MoM is {time_mle/time_mom:.1f}x faster")
print(f"  Accuracy: MLE log-lik = {dist_mle.log_likelihood():.2f}")
print(f"           MoM log-lik = {dist_mom.log_likelihood():.2f}")
print(f"  → For normal, both are nearly identical!")


# ============================================================================
# Example 2: Small Sample (n=30)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Small Sample (n=30)")
print("="*70)

n_small = 30
data_small = np.random.gamma(shape=2, scale=3, size=n_small)

print(f"\n📊 Data: {n_small} samples from Gamma(2, 3)")

# MLE
print("\n1️⃣ MLE:")
try:
    start = time.time()
    dist_mle_small = get_distribution('gamma')
    dist_mle_small.fit(data_small, method='mle')
    time_mle_small = time.time() - start
    
    print(f"  Time: {time_mle_small*1000:.2f} ms")
    print(f"  Parameters: {dist_mle_small.params}")
    print(f"  Mean: {dist_mle_small.mean():.2f}")
    mle_success = True
except Exception as e:
    print(f"  ⚠️  Failed: {e}")
    mle_success = False

# MoM
print("\n2️⃣ Method of Moments:")
start = time.time()
dist_mom_small = get_distribution('gamma')
dist_mom_small.fit(data_small, method='mom')
time_mom_small = time.time() - start

print(f"  Time: {time_mom_small*1000:.2f} ms")
print(f"  Parameters: {dist_mom_small.params}")
print(f"  Mean: {dist_mom_small.mean():.2f}")

if mle_success:
    print(f"\n🎯 Comparison (small sample):")
    print(f"  Both converged successfully")
    print(f"  MLE is more accurate but MoM is faster and more robust")
else:
    print(f"\n🎯 Comparison (small sample):")
    print(f"  MLE failed, but MoM still works!")
    print(f"  → MoM is more robust for small/difficult samples")


# ============================================================================
# Example 3: Data with Outliers
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Data with Outliers")
print("="*70)

# Generate data with outliers
data_clean = np.random.normal(50, 10, 950)
outliers = np.random.uniform(150, 200, 50)  # 5% outliers
data_contaminated = np.concatenate([data_clean, outliers])
np.random.shuffle(data_contaminated)

print(f"\n📊 Data: 950 clean + 50 outliers (5% contamination)")
print(f"  True mean: 50")
print(f"  Observed mean: {data_contaminated.mean():.2f}")

# MLE
print("\n1️⃣ MLE (sensitive to outliers):")
dist_mle_out = get_distribution('normal')
dist_mle_out.fit(data_contaminated, method='mle')

print(f"  Estimated mean: {dist_mle_out.mean():.2f}")
print(f"  Error: {abs(dist_mle_out.mean() - 50):.2f}")

# MoM
print("\n2️⃣ MoM (also sensitive, but faster to try robust alternatives):")
dist_mom_out = get_distribution('normal')
dist_mom_out.fit(data_contaminated, method='mom')

print(f"  Estimated mean: {dist_mom_out.mean():.2f}")
print(f"  Error: {abs(dist_mom_out.mean() - 50):.2f}")

print(f"\n💡 Note: Both affected by outliers!")
print(f"  → Consider robust alternatives (trimmed mean, etc.)")
print(f"  → Or use distributions with heavier tails (t, Cauchy)")


# ============================================================================
# Example 4: Speed Comparison Across Sample Sizes
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Speed Comparison")
print("="*70)

sample_sizes = [50, 100, 500, 1000, 5000, 10000]
mle_times = []
mom_times = []

print(f"\n🕒 Benchmarking... (Normal distribution)\n")
print("  Sample Size    MLE (ms)    MoM (ms)    Speedup")
print("  " + "-"*50)

for n in sample_sizes:
    data_bench = np.random.normal(0, 1, n)
    
    # MLE
    start = time.time()
    d = get_distribution('normal')
    d.fit(data_bench, method='mle')
    t_mle = (time.time() - start) * 1000
    mle_times.append(t_mle)
    
    # MoM
    start = time.time()
    d = get_distribution('normal')
    d.fit(data_bench, method='mom')
    t_mom = (time.time() - start) * 1000
    mom_times.append(t_mom)
    
    speedup = t_mle / t_mom
    print(f"  {n:>10d}     {t_mle:>7.2f}     {t_mom:>7.2f}     {speedup:>5.1f}x")


# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Visualizations...")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fit quality comparison
ax = axes[0]
ax.hist(data_large, bins=50, density=True, alpha=0.3, color='gray',
        label='Data', edgecolor='black')

x = np.linspace(data_large.min(), data_large.max(), 200)
y_mle = dist_mle.pdf(x)
y_mom = dist_mom.pdf(x)

ax.plot(x, y_mle, 'r-', linewidth=2, label='MLE fit', alpha=0.8)
ax.plot(x, y_mom, 'b--', linewidth=2, label='MoM fit', alpha=0.8)

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Fit Quality: MLE vs MoM (n=10000)', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Speed comparison
ax = axes[1]
ax.plot(sample_sizes, mle_times, 'ro-', linewidth=2, markersize=8, 
        label='MLE', alpha=0.7)
ax.plot(sample_sizes, mom_times, 'bo-', linewidth=2, markersize=8, 
        label='MoM', alpha=0.7)

ax.set_xlabel('Sample Size', fontsize=11)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Computation Speed Comparison', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()

print("\n✅ Visualizations created!")
plt.show()


print("\n" + "="*70)
print("🎓 Decision Guide: MLE vs MoM")
print("="*70)
print("""
┌────────────────────────────────────────────────────┐
│ USE MLE WHEN:                                            │
├────────────────────────────────────────────────────┤
│  ✅ You need maximum accuracy                            │
│  ✅ Large sample size (n > 100)                         │
│  ✅ Data is well-behaved (no extreme outliers)         │
│  ✅ Computational time is not critical                  │
│  ✅ Publishing research (statistically optimal)         │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│ USE MoM WHEN:                                            │
├────────────────────────────────────────────────────┤
│  ✅ You need quick, rough estimates                      │
│  ✅ Small sample size (n < 50)                          │
│  ✅ MLE fails to converge                               │
│  ✅ Speed is critical (real-time applications)          │
│  ✅ Initial parameter guess for MLE optimization        │
│  ✅ Simple distributions (Normal, Exponential)          │
└────────────────────────────────────────────────────┘

RECOMMENDATION:
  🎯 Default to MLE for final analysis
  🎯 Use MoM for exploration and debugging
  🎯 For critical applications, compare both!

Next: See 04_model_selection/ for choosing the best distribution!
""")
