#!/usr/bin/env python3
"""
Custom Distributions
===================

Create and use custom probability distributions:
  - Extend scipy distributions
  - Define custom PDF/CDF
  - Integration with distfit-pro
  - Use cases

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats, integrate

np.random.seed(42)

print("="*70)
print("🔧 CUSTOM DISTRIBUTIONS")
print("="*70)


# ============================================================================
# Example 1: Custom Continuous Distribution (Triangular)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Custom Triangular Distribution")
print("="*70)

print("""
Scenario: Create custom triangular distribution
  - Parameters: a (min), c (mode), b (max)
  - Used in project management (PERT)
""")

class TriangularDist(stats.rv_continuous):
    """Custom triangular distribution.
    
    Parameters:
    -----------
    a : float
        Minimum value
    c : float
        Mode (peak)
    b : float
        Maximum value
    """
    
    def _pdf(self, x, c):
        """PDF for standard triangular on [0, 1] with mode c."""
        # Scaled to [0, 1], mode at c
        return np.where(x < c,
                       2*x/c,
                       2*(1-x)/(1-c))
    
    def _cdf(self, x, c):
        """CDF for standard triangular."""
        return np.where(x < c,
                       x**2/c,
                       1 - (1-x)**2/(1-c))
    
    def _ppf(self, q, c):
        """Inverse CDF (quantile function)."""
        return np.where(q < (c),
                       np.sqrt(q * c),
                       1 - np.sqrt((1-q)*(1-c)))

# Create instance
triangular = TriangularDist(a=0, b=1, name='triangular_custom')

print("\n🔨 Custom Triangular Distribution Created!")
print("  Domain: [0, 1]")
print("  Parameter: c (mode position)")

# Example with c=0.3 (left-skewed)
c_param = 0.3

print(f"\n📊 Using c = {c_param} (mode at {c_param})")
print(f"  Mean: {triangular.mean(c_param):.3f}")
print(f"  Median: {triangular.median(c_param):.3f}")
print(f"  Std: {triangular.std(c_param):.3f}")

# Generate sample
sample = triangular.rvs(c_param, size=1000)

print(f"\n  Sample statistics:")
print(f"    Sample mean: {sample.mean():.3f}")
print(f"    Sample std: {sample.std():.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 1, 300)

# Plot 1: PDF
ax = axes[0]
ax.hist(sample, bins=40, density=True, alpha=0.6, color='skyblue',
        edgecolor='black', label='Sample')
ax.plot(x, triangular.pdf(x, c_param), 'r-', linewidth=2.5,
        label=f'Triangular(c={c_param})')
ax.axvline(c_param, color='green', linestyle='--', linewidth=2,
           label=f'Mode = {c_param}')
ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Custom Triangular Distribution - PDF', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: CDF
ax = axes[1]
sample_sorted = np.sort(sample)
empirical_cdf = np.arange(1, len(sample_sorted)+1) / len(sample_sorted)
ax.plot(sample_sorted, empirical_cdf, 'o', markersize=2, alpha=0.4,
        label='Empirical CDF')
ax.plot(x, triangular.cdf(x, c_param), 'r-', linewidth=2.5,
        label='Theoretical CDF')
ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax.set_title('Custom Triangular Distribution - CDF', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
print("\n📊 Custom triangular distribution plots created!")
plt.savefig('/tmp/custom_triangular.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: Custom Distribution from Data (Kernel Density)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Non-Parametric Distribution (KDE)")
print("="*70)

print("""
Scenario: Data doesn't fit standard distributions
  - Use Kernel Density Estimation (KDE)
  - Non-parametric approach
""")

# Generate bimodal data
data_bimodal = np.concatenate([
    np.random.normal(20, 3, 300),
    np.random.normal(40, 4, 200)
])
np.random.shuffle(data_bimodal)

print(f"\n📊 Data: {len(data_bimodal)} observations (bimodal)")
print(f"  Mean: {data_bimodal.mean():.2f}")
print(f"  Std: {data_bimodal.std():.2f}")

# Try standard normal (will fail)
print("\n1️⃣ Fitting standard normal (will be poor):")
dist_normal = get_distribution('normal')
dist_normal.fit(data_bimodal)
print(f"  AIC: {dist_normal.aic():.2f}")

# Use KDE
print("\n2️⃣ Using Kernel Density Estimation (KDE):")
kde = stats.gaussian_kde(data_bimodal)
print(f"  ✅ KDE fitted (bandwidth = {kde.factor:.4f})")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x_range = np.linspace(data_bimodal.min(), data_bimodal.max(), 300)

# Plot 1: Normal vs KDE
ax1.hist(data_bimodal, bins=40, density=True, alpha=0.5, color='gray',
         edgecolor='black', label='Data')
ax1.plot(x_range, dist_normal.pdf(x_range), 'r--', linewidth=2,
         label='Normal (poor fit)')
ax1.plot(x_range, kde(x_range), 'b-', linewidth=2.5,
         label='KDE (good fit)')
ax1.set_xlabel('Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('Normal vs KDE', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: KDE sampling
sample_kde = kde.resample(500).flatten()
ax2.hist(data_bimodal, bins=40, density=True, alpha=0.4, color='gray',
         edgecolor='black', label='Original Data')
ax2.hist(sample_kde, bins=40, density=True, alpha=0.4, color='blue',
         edgecolor='black', label='KDE Sample')
ax2.plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
ax2.set_xlabel('Value', fontsize=11, fontweight='bold')
ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
ax2.set_title('KDE: Sampling from Fitted Distribution', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 KDE comparison plots created!")
plt.savefig('/tmp/custom_kde.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 3: Truncated/Modified Distribution
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Truncated Normal Distribution")
print("="*70)

print("""
Scenario: Normal distribution but values must be in [0, 100]
  - Truncated distribution
  - Common in real applications (e.g., test scores)
""")

# Parameters
mu, sigma = 70, 15
lower, upper = 0, 100

print(f"\n📈 Truncated Normal({mu}, {sigma}²) on [{lower}, {upper}]")

# Create truncated normal using scipy
a_truncated = (lower - mu) / sigma
b_truncated = (upper - mu) / sigma
trunc_norm = stats.truncnorm(a_truncated, b_truncated, loc=mu, scale=sigma)

print(f"  Mean: {trunc_norm.mean():.2f}")
print(f"  Std: {trunc_norm.std():.2f}")

# Generate samples
sample_truncated = trunc_norm.rvs(1000)

print(f"\n  Sample statistics:")
print(f"    Sample mean: {sample_truncated.mean():.2f}")
print(f"    Sample std: {sample_truncated.std():.2f}")
print(f"    Min: {sample_truncated.min():.2f}")
print(f"    Max: {sample_truncated.max():.2f}")

# Compare with regular normal
regular_normal = stats.norm(mu, sigma)
sample_regular = regular_normal.rvs(1000)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_plot = np.linspace(-10, 110, 300)

# Plot 1: PDF comparison
ax = axes[0]
ax.hist(sample_regular, bins=40, density=True, alpha=0.4, color='red',
        edgecolor='black', label='Regular Normal')
ax.hist(sample_truncated, bins=40, density=True, alpha=0.6, color='blue',
        edgecolor='black', label='Truncated Normal')

x_regular = x_plot
x_truncated = x_plot[(x_plot >= lower) & (x_plot <= upper)]

ax.plot(x_regular, regular_normal.pdf(x_regular), 'r--', linewidth=2,
        label='Regular PDF')
ax.plot(x_truncated, trunc_norm.pdf(x_truncated), 'b-', linewidth=2.5,
        label='Truncated PDF')

ax.axvline(lower, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(upper, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.axvspan(lower, upper, alpha=0.1, color='green', label='Valid Range')

ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Regular vs Truncated Normal', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim([lower-20, upper+20])

# Plot 2: CDF comparison
ax = axes[1]
ax.plot(x_plot, regular_normal.cdf(x_plot), 'r--', linewidth=2,
        label='Regular CDF')
ax.plot(x_plot, trunc_norm.cdf(x_plot), 'b-', linewidth=2.5,
        label='Truncated CDF')
ax.axvline(lower, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.axvline(upper, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.axvspan(lower, upper, alpha=0.1, color='green')
ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax.set_title('CDF Comparison', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([lower-20, upper+20])
ax.set_ylim([0, 1.05])

plt.tight_layout()
print("\n📊 Truncated distribution plots created!")
plt.savefig('/tmp/custom_truncated.png', dpi=150, bbox_inches='tight')

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways - Custom Distributions")
print("="*70)
print("""
1. WHY CUSTOM DISTRIBUTIONS?
   • Data doesn't fit standard distributions
   • Domain-specific requirements
   • Truncated/bounded ranges
   • Mixture of distributions

2. THREE APPROACHES:
   a) Extend scipy.stats.rv_continuous
      - Define _pdf, _cdf, _ppf methods
      - Full control, most flexible
      - Requires mathematical knowledge
   
   b) Kernel Density Estimation (KDE)
      - Non-parametric
      - No distributional assumptions
      - Good for complex, multimodal data
   
   c) Truncated/Modified Standard Distributions
      - Use scipy.truncnorm, etc.
      - Keep familiar properties
      - Enforce constraints (bounds)

3. SCIPY CUSTOM DISTRIBUTION:
   class MyDist(stats.rv_continuous):
       def _pdf(self, x, *params):
           # Define PDF
           return ...
       
       def _cdf(self, x, *params):
           # Define CDF
           return ...
   
   dist = MyDist(a=0, b=1)  # Domain [0, 1]

4. KERNEL DENSITY ESTIMATION:
   from scipy.stats import gaussian_kde
   
   kde = gaussian_kde(data)
   density = kde(x)  # Evaluate at x
   sample = kde.resample(n)  # Generate samples

5. TRUNCATED DISTRIBUTIONS:
   from scipy.stats import truncnorm
   
   a = (lower - mu) / sigma
   b = (upper - mu) / sigma
   dist = truncnorm(a, b, loc=mu, scale=sigma)

6. USE CASES:
   • Triangular: Project management (PERT)
   • Beta: Probabilities, proportions [0,1]
   • Truncated Normal: Test scores, constrained data
   • KDE: Complex real-world data
   • Custom: Industry-specific distributions

7. BEST PRACTICES:
   ✓ Start with standard distributions
   ✓ Use custom only when necessary
   ✓ Validate with Q-Q plots
   ✓ Document assumptions clearly
   ✓ Test edge cases (bounds, tails)

8. LIMITATIONS:
   • Custom distributions harder to communicate
   • Less theoretical support
   • May not have closed-form formulas
   • KDE requires sufficient data

Congratulations! You've mastered advanced distribution fitting! 🎉
""")
