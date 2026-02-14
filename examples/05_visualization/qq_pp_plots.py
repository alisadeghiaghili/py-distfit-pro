#!/usr/bin/env python3
"""
Q-Q and P-P Plots
================

Quantile-Quantile and Probability-Probability plots for
assessing goodness-of-fit visually.

Q-Q Plot:
  - Compares quantiles (ordered values)
  - More sensitive to tails
  - Most common for normality testing

P-P Plot:
  - Compares cumulative probabilities
  - More sensitive to center
  - Better for comparing distribution shapes

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("📊 Q-Q AND P-P PLOTS FOR GOODNESS-OF-FIT")
print("="*70)


# ============================================================================
# Theory: Q-Q vs P-P Plots
# ============================================================================

print("\n" + "="*70)
print("📚 Theory: Q-Q vs P-P Plots")
print("="*70)

theory = """
1. Q-Q PLOT (Quantile-Quantile):
   • X-axis: Theoretical quantiles from distribution
   • Y-axis: Empirical quantiles from data
   • Perfect fit: Points on 45° line
   • Deviations show where fit is poor
   • More sensitive to tail behavior
   • Most popular for visual assessment

2. P-P PLOT (Probability-Probability):
   • X-axis: Theoretical cumulative probabilities
   • Y-axis: Empirical cumulative probabilities
   • Perfect fit: Points on 45° line
   • More sensitive to center of distribution
   • Good for comparing distribution shapes

3. INTERPRETATION:
   • Points on line: Good fit
   • Points above line: Data > theoretical (right tail heavy)
   • Points below line: Data < theoretical (left tail heavy)
   • S-shape: Wrong distribution family
   • Systematic pattern: Systematic misfit

4. WHEN TO USE:
   • Q-Q: General goodness-of-fit, especially tails
   • P-P: When center fit is more important
   • Both: Most robust assessment
"""

print(theory)


# ============================================================================
# Example 1: Perfect Fit (Q-Q Plot)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Q-Q Plot - Good Fit")
print("="*70)

# Generate normal data
data_good = np.random.normal(loc=50, scale=10, size=1000)

print(f"\n📊 Data: {len(data_good)} samples from N(50, 10²)")

# Fit normal distribution
dist_good = get_distribution('normal')
dist_good.fit(data_good)

print(f"\n🎨 Creating Q-Q plot...")

fig, ax = plt.subplots(figsize=(8, 8))

# Calculate quantiles
percentiles = np.linspace(0.01, 0.99, len(data_good))
theoretical_quantiles = dist_good.ppf(percentiles)
empirical_quantiles = np.sort(data_good)

# Q-Q plot
ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=20, 
           color='blue', edgecolors='black', linewidth=0.5, label='Data')

# Perfect fit line (45°)
min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
        label='Perfect Fit (45° line)')

ax.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
ax.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
ax.set_title('Q-Q Plot: Good Fit Example', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal', adjustable='box')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation
ax.text(0.05, 0.95, 'Points close to line\n→ Good fit', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
print("✅ Q-Q plot created (good fit)")
plt.savefig('/tmp/qq_good_fit.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: Poor Fit (Q-Q Plot)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Q-Q Plot - Poor Fit")
print("="*70)

# Generate exponential data (skewed)
data_poor = np.random.exponential(scale=10, size=1000)

print(f"\n📊 Data: {len(data_poor)} samples from Exp(λ=1/10)")

# WRONGLY fit normal distribution
dist_wrong = get_distribution('normal')
dist_wrong.fit(data_poor)

print(f"\n🎨 Creating Q-Q plot (wrong distribution)...")

fig, ax = plt.subplots(figsize=(8, 8))

# Calculate quantiles
percentiles = np.linspace(0.01, 0.99, len(data_poor))
theoretical_q_wrong = dist_wrong.ppf(percentiles)
empirical_q_poor = np.sort(data_poor)

# Q-Q plot
ax.scatter(theoretical_q_wrong, empirical_q_poor, alpha=0.6, s=20, 
           color='red', edgecolors='black', linewidth=0.5, label='Data')

# Perfect fit line
min_v = min(theoretical_q_wrong.min(), empirical_q_poor.min())
max_v = max(theoretical_q_wrong.max(), empirical_q_poor.max())
ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2.5, 
        label='Perfect Fit Line')

ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
ax.set_title('Q-Q Plot: Poor Fit Example', fontsize=14, fontweight='bold', color='red')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal', adjustable='box')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation
ax.text(0.05, 0.95, 'Points deviate from line\n→ Poor fit!', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
print("✅ Q-Q plot created (poor fit)")
plt.savefig('/tmp/qq_poor_fit.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 3: P-P Plot
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: P-P Plot")
print("="*70)

# Use the good data
print(f"\n🎨 Creating P-P plot...")

fig, ax = plt.subplots(figsize=(8, 8))

# Calculate probabilities
empirical_cdf = np.arange(1, len(data_good) + 1) / len(data_good)
data_sorted = np.sort(data_good)
theoretical_cdf = dist_good.cdf(data_sorted)

# P-P plot
ax.scatter(theoretical_cdf, empirical_cdf, alpha=0.6, s=20, 
           color='green', edgecolors='black', linewidth=0.5, label='Data')

# Perfect fit line
ax.plot([0, 1], [0, 1], 'r--', linewidth=2.5, label='Perfect Fit')

ax.set_xlabel('Theoretical Cumulative Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Empirical Cumulative Probability', fontsize=12, fontweight='bold')
ax.set_title('P-P Plot: Probability-Probability', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal', adjustable='box')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("✅ P-P plot created")
plt.savefig('/tmp/pp_plot.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 4: Q-Q and P-P Side-by-Side
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Q-Q and P-P Comparison")
print("="*70)

print(f"\n🎨 Creating side-by-side comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Q-Q Plot
ax1.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=15, 
            color='blue', edgecolors='black', linewidth=0.5)
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax1.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
ax1.set_title('A) Q-Q Plot', fontsize=12, fontweight='bold', loc='left')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: P-P Plot
ax2.scatter(theoretical_cdf, empirical_cdf, alpha=0.6, s=15, 
            color='green', edgecolors='black', linewidth=0.5)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2)
ax2.set_xlabel('Theoretical Probability', fontsize=11, fontweight='bold')
ax2.set_ylabel('Empirical Probability', fontsize=11, fontweight='bold')
ax2.set_title('B) P-P Plot', fontsize=12, fontweight='bold', loc='left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_aspect('equal', adjustable='box')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Goodness-of-Fit Assessment: Q-Q vs P-P', 
             fontsize=14, fontweight='bold', y=1.00)

plt.tight_layout()
print("✅ Side-by-side comparison created")
plt.savefig('/tmp/qq_pp_comparison.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 5: Multiple Distributions Q-Q Comparison
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Compare Multiple Distributions (Q-Q)")
print("="*70)

# Generate slightly heavy-tailed data
data_tails = np.random.standard_t(df=5, size=800)

print(f"\n📊 Data: {len(data_tails)} samples from t(df=5)")
print(f"   Testing: Normal, t-distribution, Logistic")

# Fit multiple distributions
dists_compare = [
    ('Normal', 'normal', 'blue'),
    ('t (df=5)', 't', 'red'),
    ('Logistic', 'logistic', 'green'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, dist_name, color) in enumerate(dists_compare):
    ax = axes[idx]
    
    # Fit distribution
    d = get_distribution(dist_name)
    d.fit(data_tails)
    
    # Q-Q plot
    p = np.linspace(0.01, 0.99, len(data_tails))
    theoretical = d.ppf(p)
    empirical = np.sort(data_tails)
    
    ax.scatter(theoretical, empirical, alpha=0.6, s=10, color=color, 
               edgecolors='black', linewidth=0.5)
    
    min_v = min(theoretical.min(), empirical.min())
    max_v = max(theoretical.max(), empirical.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2)
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax.set_ylabel('Sample Quantiles', fontsize=10)
    ax.set_title(f'{name}\nAIC={d.aic():.1f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Q-Q Plots: Which Distribution Fits Best?', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
print("✅ Multi-distribution Q-Q comparison created")
plt.savefig('/tmp/qq_multi_comparison.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 6: Q-Q with Confidence Bands
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Q-Q Plot with Confidence Bands")
print("="*70)

print(f"\n🎨 Creating Q-Q plot with confidence bands...")

fig, ax = plt.subplots(figsize=(8, 8))

# Q-Q plot
ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=20, 
           color='blue', edgecolors='black', linewidth=0.5, label='Data', zorder=3)

# Perfect fit line
ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2.5, 
        label='Perfect Fit', zorder=2)

# Approximate 95% confidence bands (simple approach)
# More rigorous: use order statistics theory
n = len(data_good)
std_error = dist_good.std() * np.sqrt(percentiles * (1 - percentiles) / n)
band_width = 1.96 * std_error  # 95% CI

upper_band = theoretical_quantiles + band_width * 3  # Scaled for visibility
lower_band = theoretical_quantiles - band_width * 3

ax.fill_between(theoretical_quantiles, lower_band, upper_band, 
                alpha=0.2, color='gray', label='~95% Confidence Band', zorder=1)

ax.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
ax.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
ax.set_title('Q-Q Plot with Confidence Bands', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal', adjustable='box')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.05, 0.95, 'Points within bands\n→ Consistent with distribution', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
print("✅ Q-Q plot with confidence bands created")
plt.savefig('/tmp/qq_confidence_bands.png', dpi=150, bbox_inches='tight')


print("\n" + "="*70)
print("📁 Files Saved")
print("="*70)
print("""
Plots saved to /tmp/:
  1. qq_good_fit.png
  2. qq_poor_fit.png
  3. pp_plot.png
  4. qq_pp_comparison.png
  5. qq_multi_comparison.png
  6. qq_confidence_bands.png
""")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Q-Q PLOT INTERPRETATION:
   ✓ Points on 45° line: Good fit
   ❌ Points systematically above/below: Wrong scale parameter
   ❌ S-shaped pattern: Wrong distribution family
   ❌ Heavy deviation in tails: Tail behavior mismatch

2. P-P PLOT INTERPRETATION:
   • Similar to Q-Q but focuses on probabilities
   • More sensitive to center of distribution
   • Less sensitive to tail behavior
   • Good complement to Q-Q plot

3. Q-Q vs P-P:
   • Q-Q: More popular, better for tails
   • P-P: Better for center, less used
   • Use both for complete picture

4. COMMON PATTERNS:
   • Linear but shifted: Wrong location parameter
   • Different slope: Wrong scale parameter
   • Curved: Wrong distribution family
   • More scatter in tails: Heavy/light tail issues

5. BEST PRACTICES:
   ✓ Use with other diagnostics (KS test, AIC)
   ✓ Check both tails carefully
   ✓ Compare multiple distributions
   ✓ Use confidence bands for uncertainty
   ✓ Consider sample size (small n = more scatter)

6. SCIPY ALTERNATIVE:
   from scipy import stats
   stats.probplot(data, dist='norm', plot=ax)  # Q-Q plot

7. IN PRACTICE:
   • Q-Q plot is #1 visual diagnostic
   • Quick, intuitive assessment
   • Complements statistical tests
   • Reveals specific issues (tails, center)

Next: See interactive_plots.py for Plotly interactive visualizations!
""")
