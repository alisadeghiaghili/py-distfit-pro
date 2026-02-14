#!/usr/bin/env python3
"""
Weighted Fitting Example
=======================

Fit distributions when data points have different importance/reliability.

Use cases:
  - Survey data with sampling weights
  - Aggregated data with frequency counts
  - Heteroscedastic measurement errors
  - Stratified sampling

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
from distfit_pro.core.weighted import WeightedFitting
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("⚖️ WEIGHTED DISTRIBUTION FITTING")
print("="*70)


# ============================================================================
# Example 1: Survey Data with Sampling Weights
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Survey with Sampling Weights")
print("="*70)
print("""
Scenario: Income survey with stratified sampling
  - Different regions sampled at different rates
  - Each response has a sampling weight
  - Need to account for weights when fitting distribution
""")

# Generate synthetic survey data
n_samples = 1000
data = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples)

# Simulate sampling weights (1.0 = normal, >1.0 = overrepresented)
weights = np.random.uniform(0.5, 2.0, size=n_samples)

print(f"\n📊 Data: {n_samples} survey responses")
print(f"  Mean income: ${data.mean()/1000:.1f}k")
print(f"  Median:      ${np.median(data)/1000:.1f}k")
print(f"\n🎯 Weights:")
print(f"  Range: [{weights.min():.2f}, {weights.max():.2f}]")
print(f"  Mean:  {weights.mean():.2f}")

# Fit WITHOUT weights (naive approach)
print("\n1️⃣ Fitting WITHOUT weights (unweighted):")
dist_unweighted = get_distribution('lognormal')
dist_unweighted.fit(data)

print(f"  Parameters: {dist_unweighted.params}")
print(f"  Fitted mean: ${dist_unweighted.mean()/1000:.1f}k")

# Fit WITH weights (correct approach)
print("\n2️⃣ Fitting WITH weights (weighted MLE):")
dist_weighted = get_distribution('lognormal')
params_weighted = WeightedFitting.fit_weighted_mle(data, weights, dist_weighted)
dist_weighted.params = params_weighted
dist_weighted.fitted = True

print(f"  Parameters: {dist_weighted.params}")
print(f"  Fitted mean: ${dist_weighted.mean()/1000:.1f}k")

# Compare
print("\n⚖️ Comparison:")
print(f"  Unweighted mean: ${dist_unweighted.mean()/1000:.1f}k")
print(f"  Weighted mean:   ${dist_weighted.mean()/1000:.1f}k")
print(f"  Difference:      ${abs(dist_weighted.mean() - dist_unweighted.mean())/1000:.1f}k")


# ============================================================================
# Example 2: Aggregated Frequency Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Aggregated Frequency Data")
print("="*70)
print("""
Scenario: Customer age data aggregated into bins
  - Data: age bins and their frequencies
  - Need to fit distribution to bin centers with frequencies as weights
""")

# Aggregated data: (bin_center, frequency)
age_bins = np.array([25, 35, 45, 55, 65, 75])
frequencies = np.array([150, 300, 250, 200, 80, 20])

print("\n📊 Aggregated data:")
print("  Age Bin    Frequency")
print("  " + "-"*25)
for age, freq in zip(age_bins, frequencies):
    print(f"    {age:3d}        {freq:4d}")

total_customers = frequencies.sum()
print(f"\n  Total customers: {total_customers}")

# Fit distribution using bin centers and frequencies
print("\n🎯 Fitting normal distribution to aggregated data...")

dist_freq = get_distribution('normal')
params_freq = WeightedFitting.fit_weighted_mle(age_bins, frequencies, dist_freq)
dist_freq.params = params_freq
dist_freq.fitted = True

print(f"\n✅ Fitted parameters:")
for param, val in dist_freq.params.items():
    print(f"  {param}: {val:.2f}")

print(f"\n  Mean age: {dist_freq.mean():.1f} years")
print(f"  Std:      {dist_freq.std():.1f} years")

# Calculate business insights
under_40 = dist_freq.cdf(40)
over_60 = 1 - dist_freq.cdf(60)

print(f"\n💼 Business insights:")
print(f"  % customers under 40: {under_40*100:.1f}%")
print(f"  % customers over 60:  {over_60*100:.1f}%")


# ============================================================================
# Example 3: Measurement Uncertainty
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Data with Varying Measurement Precision")
print("="*70)
print("""
Scenario: Temperature measurements from different sensors
  - High-precision sensors (weight = high)
  - Low-precision sensors (weight = low)
  - Want to trust precise measurements more
""")

# Generate data with varying precision
n = 500
true_mean = 25.0  # True temperature

# Half from precise sensors (σ=0.5), half from imprecise (σ=2.0)
precise_data = np.random.normal(true_mean, 0.5, n//2)
imprecise_data = np.random.normal(true_mean, 2.0, n//2)

data_mixed = np.concatenate([precise_data, imprecise_data])

# Assign weights based on precision (inversely proportional to variance)
# Precise: weight = 1/(0.5²) = 4.0
# Imprecise: weight = 1/(2.0²) = 0.25
weights_mixed = np.concatenate([
    np.full(n//2, 4.0),    # Precise sensors
    np.full(n//2, 0.25),   # Imprecise sensors
])

print(f"\n🌡️  Temperature data: {n} measurements")
print(f"  {n//2} from precise sensors (weight=4.0)")
print(f"  {n//2} from imprecise sensors (weight=0.25)")
print(f"  True mean: {true_mean}°C")

# Unweighted fit
dist_unweighted_temp = get_distribution('normal')
dist_unweighted_temp.fit(data_mixed)

print(f"\n1️⃣ Unweighted estimate:")
print(f"  Mean: {dist_unweighted_temp.mean():.2f}°C")
print(f"  Std:  {dist_unweighted_temp.std():.2f}°C")
print(f"  Error: {abs(dist_unweighted_temp.mean() - true_mean):.2f}°C")

# Weighted fit (trust precise measurements more)
dist_weighted_temp = get_distribution('normal')
params_weighted_temp = WeightedFitting.fit_weighted_mle(
    data_mixed, weights_mixed, dist_weighted_temp
)
dist_weighted_temp.params = params_weighted_temp
dist_weighted_temp.fitted = True

print(f"\n2️⃣ Weighted estimate (precision-based):")
print(f"  Mean: {dist_weighted_temp.mean():.2f}°C")
print(f"  Std:  {dist_weighted_temp.std():.2f}°C")
print(f"  Error: {abs(dist_weighted_temp.mean() - true_mean):.2f}°C")

print(f"\n✅ Weighted estimate is closer to true value!")


# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Comparison Visualizations...")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Survey data
ax = axes[0]
ax.hist(data, bins=50, density=True, alpha=0.3, color='gray', 
        label='Raw data', edgecolor='black')

# Create weighted histogram
ax.hist(data, bins=50, weights=weights, density=True, alpha=0.5, 
        color='skyblue', label='Weighted data', edgecolor='black')

x = np.linspace(data.min(), data.max(), 200)
y_unweighted = dist_unweighted.pdf(x)
y_weighted = dist_weighted.pdf(x)

ax.plot(x, y_unweighted, 'r--', linewidth=2, label='Unweighted fit')
ax.plot(x, y_weighted, 'b-', linewidth=2, label='Weighted fit')

ax.set_xlabel('Income', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Survey Data: Weighted vs Unweighted Fitting', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Temperature measurements
ax = axes[1]
ax.hist(data_mixed, bins=40, density=True, alpha=0.5, color='skyblue', 
        label='Mixed precision data', edgecolor='black')

x_temp = np.linspace(data_mixed.min(), data_mixed.max(), 200)
y_unw_temp = dist_unweighted_temp.pdf(x_temp)
y_w_temp = dist_weighted_temp.pdf(x_temp)

ax.plot(x_temp, y_unw_temp, 'r--', linewidth=2, label='Unweighted fit')
ax.plot(x_temp, y_w_temp, 'g-', linewidth=2, label='Weighted fit (precision-based)')
ax.axvline(true_mean, color='black', linestyle=':', linewidth=2, label='True mean')

ax.set_xlabel('Temperature (°C)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Measurement Data: Accounting for Precision', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Visualizations created!")
plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Use weighted fitting when:
   • Data points have different importance/reliability
   • Survey data with sampling weights
   • Aggregated frequency data
   • Measurements with varying precision

2. Implementation:
   from distfit_pro.core.weighted import WeightedFitting
   params = WeightedFitting.fit_weighted_mle(data, weights, dist)
   dist.params = params
   dist.fitted = True

3. Weights represent relative importance (not probabilities)

4. Ignoring weights can lead to biased estimates!

Next: See censored_data.py for survival analysis with incomplete data!
""")
