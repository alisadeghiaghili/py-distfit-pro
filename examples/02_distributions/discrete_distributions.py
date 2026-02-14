#!/usr/bin/env python3
"""
Discrete Distributions Showcase
===============================

Explore discrete distributions for count data.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("🔢 DISCRETE DISTRIBUTIONS SHOWCASE")
print("="*70)


# ============================================================================
# 1. POISSON - Events Per Time Period
# ============================================================================

print("\n" + "="*70)
print("1. POISSON DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Counting events in fixed time/space
  ✓ Website visits per hour
  ✓ Calls to support center per day
  ✓ Defects per product
  ✓ Rate parameter λ = mean = variance
""")

# Simulate: Average 5 customer calls per hour
data_poisson = np.random.poisson(lam=5.0, size=1000)
dist_poisson = get_distribution('poisson')
dist_poisson.fit(data_poisson)

print("Fitted parameters:")
for param, val in dist_poisson.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean (λ):  {dist_poisson.mean():.2f}")
print(f"Variance: {dist_poisson.var():.2f}")
print(f"\n📊 Example: Probability of exactly 3 calls in an hour:")
print(f"   P(X=3) = {dist_poisson.pdf(3):.4f} or {dist_poisson.pdf(3)*100:.2f}%")


# ============================================================================
# 2. BINOMIAL - Success/Failure Trials
# ============================================================================

print("\n" + "="*70)
print("2. BINOMIAL DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Fixed number of independent trials (n)
  ✓ Each trial has two outcomes (success/failure)
  ✓ Constant probability of success (p)
  ✓ Quality control (defective items)
  ✓ Survey responses (yes/no)
""")

# Simulate: 10 coin flips, 60% heads probability, repeated 1000 times
data_binom = np.random.binomial(n=10, p=0.6, size=1000)

# Note: binomial distribution fitting is tricky (n must be known)
# For demo, we'll analyze the data
print(f"Data: {len(data_binom)} trials of 10 flips each")
print(f"  Mean successes per trial: {data_binom.mean():.2f}")
print(f"  Std: {data_binom.std():.2f}")
print(f"  Range: [{data_binom.min()}, {data_binom.max()}]")

# Create theoretical distribution for comparison
theoretical_binom = stats.binom(n=10, p=0.6)
print(f"\nTheoretical (n=10, p=0.6):")
print(f"  Mean: {theoretical_binom.mean():.2f}")
print(f"  P(X=6) = {theoretical_binom.pmf(6):.4f}")


# ============================================================================
# 3. GEOMETRIC - Trials Until First Success
# ============================================================================

print("\n" + "="*70)
print("3. GEOMETRIC DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Number of trials until first success
  ✓ Probability of success is constant (p)
  ✓ Memoryless property
  ✓ Customer acquisition attempts
  ✓ Equipment testing until failure
""")

# Simulate: Keep trying until success (p=0.2)
data_geom = np.random.geometric(p=0.2, size=1000)

# Fit using continuous approximation (geometric is discrete)
print(f"Data: {len(data_geom)} trials until success")
print(f"  Mean trials: {data_geom.mean():.2f}")
print(f"  Median: {np.median(data_geom):.2f}")
print(f"  Max: {data_geom.max()}")

theoretical_geom = stats.geom(p=0.2)
print(f"\nTheoretical (p=0.2):")
print(f"  Mean: {theoretical_geom.mean():.2f}")
print(f"  P(X=5) = {theoretical_geom.pmf(5):.4f}")


# ============================================================================
# 4. NEGATIVE BINOMIAL - Trials Until r Successes
# ============================================================================

print("\n" + "="*70)
print("4. NEGATIVE BINOMIAL DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Number of trials until r-th success
  ✓ Overdispersed count data (variance > mean)
  ✓ Generalization of geometric (r=1)
  ✓ Insurance claims
  ✓ Ecology (species counts)
""")

# Simulate: Trials until 3rd success (p=0.3)
data_nbinom = np.random.negative_binomial(n=3, p=0.3, size=1000)

print(f"Data: {len(data_nbinom)} repetitions")
print(f"  Mean: {data_nbinom.mean():.2f}")
print(f"  Variance: {data_nbinom.var():.2f}")
print(f"  Variance > Mean? {'Yes (overdispersed)' if data_nbinom.var() > data_nbinom.mean() else 'No'}")

theoretical_nbinom = stats.nbinom(n=3, p=0.3)
print(f"\nTheoretical (r=3, p=0.3):")
print(f"  Mean: {theoretical_nbinom.mean():.2f}")
print(f"  Variance: {theoretical_nbinom.var():.2f}")


# ============================================================================
# Visual Comparison
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Visual Comparison...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Discrete Distributions Gallery', fontsize=16, fontweight='bold')

# 1. Poisson
ax = axes[0, 0]
ax.hist(data_poisson, bins=range(0, max(data_poisson)+2), density=True, 
        alpha=0.7, color='skyblue', edgecolor='black', label='Observed')
x_poi = np.arange(0, max(data_poisson)+1)
y_poi = dist_poisson.pdf(x_poi)
ax.plot(x_poi, y_poi, 'ro-', linewidth=2, markersize=6, label='Fitted Poisson')
ax.set_title('Poisson Distribution', fontweight='bold')
ax.set_xlabel('Count')
ax.set_ylabel('Probability')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Binomial
ax = axes[0, 1]
ax.hist(data_binom, bins=range(0, 12), density=True, 
        alpha=0.7, color='lightgreen', edgecolor='black', label='Observed')
x_binom = np.arange(0, 11)
y_binom = theoretical_binom.pmf(x_binom)
ax.plot(x_binom, y_binom, 'ro-', linewidth=2, markersize=6, label='Theoretical')
ax.set_title('Binomial Distribution (n=10, p=0.6)', fontweight='bold')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Probability')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Geometric
ax = axes[1, 0]
ax.hist(data_geom, bins=range(1, min(max(data_geom)+2, 30)), density=True, 
        alpha=0.7, color='salmon', edgecolor='black', label='Observed')
x_geom = np.arange(1, min(max(data_geom)+1, 30))
y_geom = theoretical_geom.pmf(x_geom)
ax.plot(x_geom, y_geom, 'bo-', linewidth=2, markersize=6, label='Theoretical')
ax.set_title('Geometric Distribution (p=0.2)', fontweight='bold')
ax.set_xlabel('Trials Until Success')
ax.set_ylabel('Probability')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Negative Binomial
ax = axes[1, 1]
ax.hist(data_nbinom, bins=range(0, min(max(data_nbinom)+2, 30)), density=True, 
        alpha=0.7, color='plum', edgecolor='black', label='Observed')
x_nbinom = np.arange(0, min(max(data_nbinom)+1, 30))
y_nbinom = theoretical_nbinom.pmf(x_nbinom)
ax.plot(x_nbinom, y_nbinom, 'go-', linewidth=2, markersize=6, label='Theoretical')
ax.set_title('Negative Binomial (r=3, p=0.3)', fontweight='bold')
ax.set_xlabel('Failures Before 3rd Success')
ax.set_ylabel('Probability')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Visual comparison created!")
print("   Close the plot window to continue...")

plt.show()

print("\n" + "="*70)
print("🎓 Summary")
print("="*70)
print("""
Discrete Distribution Selection Guide:

🔢 Poisson:
   - Events in fixed time/space
   - Mean = Variance
   - Examples: calls/hour, defects/unit

🎲 Binomial:
   - Fixed trials (n), constant success probability (p)
   - Examples: quality control, A/B testing

⏳ Geometric:
   - Trials until first success
   - Memoryless property
   - Examples: customer acquisition, first failure

🔄 Negative Binomial:
   - Trials until r-th success
   - Allows overdispersion (variance > mean)
   - Examples: insurance claims, ecology counts

Next: Check custom_parameters.py to manually set distribution params!
""")
