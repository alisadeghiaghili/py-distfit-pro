#!/usr/bin/env python3
"""
Basic Visualization for Distribution Fitting
=============================================

This example demonstrates:
1. Histogram with fitted PDF overlay
2. QQ-plot for goodness-of-fit
3. PP-plot for probability comparison
4. CDF comparison

Note: Requires matplotlib. Install with: pip install matplotlib
"""

import numpy as np
from distfit_pro import get_distribution

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("ERROR: matplotlib not installed")
    print("Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False
    exit(1)

np.random.seed(42)

print("="*70)
print("VISUALIZATION BASICS")
print("="*70)

# =============================================================================
# STEP 1: Generate and fit data
# =============================================================================
print("\n[1] Generating and fitting data...")
print("-" * 70)

# Generate data from normal distribution
true_mean, true_std = 10, 2
data = np.random.normal(true_mean, true_std, 1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data)

print(f"Generated {len(data)} samples")
print(f"Fitted: μ={dist.params['loc']:.4f}, σ={dist.params['scale']:.4f}")

# =============================================================================
# STEP 2: Create visualizations
# =============================================================================
print("\n[2] Creating visualizations...")
print("-" * 70)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution Fitting Diagnostics', fontsize=16, fontweight='bold')

# -----------------------------------------------------------------------------
# Plot 1: Histogram with PDF overlay
# -----------------------------------------------------------------------------
ax1 = axes[0, 0]

# Plot histogram
ax1.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', 
         edgecolor='black', label='Data histogram')

# Plot fitted PDF
x_range = np.linspace(data.min(), data.max(), 200)
pdf_values = dist.pdf(x_range)
ax1.plot(x_range, pdf_values, 'r-', linewidth=2, label='Fitted PDF')

ax1.set_xlabel('Value', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Histogram with Fitted PDF', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add text with parameters
params_text = f"μ = {dist.params['loc']:.3f}\nσ = {dist.params['scale']:.3f}"
ax1.text(0.02, 0.98, params_text, transform=ax1.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5), fontsize=10)

# -----------------------------------------------------------------------------
# Plot 2: QQ-plot
# -----------------------------------------------------------------------------
ax2 = axes[0, 1]

# Sort data
sorted_data = np.sort(data)
n = len(sorted_data)

# Theoretical quantiles
theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, n))

# Plot QQ
ax2.scatter(theoretical_quantiles, sorted_data, alpha=0.5, s=20, 
            color='blue', edgecolors='black', linewidths=0.5)

# Add 45-degree reference line
min_val = min(theoretical_quantiles.min(), sorted_data.min())
max_val = max(theoretical_quantiles.max(), sorted_data.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
         linewidth=2, label='Perfect fit')

ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
ax2.set_ylabel('Sample Quantiles', fontsize=11)
ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Plot 3: PP-plot
# -----------------------------------------------------------------------------
ax3 = axes[1, 0]

# Empirical CDF
empirical_cdf = np.arange(1, n+1) / n

# Theoretical CDF
theoretical_cdf = dist.cdf(sorted_data)

# Plot PP
ax3.scatter(theoretical_cdf, empirical_cdf, alpha=0.5, s=20,
            color='green', edgecolors='black', linewidths=0.5)

# Add 45-degree reference line
ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')

ax3.set_xlabel('Theoretical Cumulative Probability', fontsize=11)
ax3.set_ylabel('Empirical Cumulative Probability', fontsize=11)
ax3.set_title('P-P Plot', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# Plot 4: CDF comparison
# -----------------------------------------------------------------------------
ax4 = axes[1, 1]

# Empirical CDF (step function)
ax4.step(sorted_data, empirical_cdf, where='post', 
         label='Empirical CDF', linewidth=1.5, color='blue')

# Theoretical CDF
x_cdf = np.linspace(data.min(), data.max(), 500)
y_cdf = dist.cdf(x_cdf)
ax4.plot(x_cdf, y_cdf, 'r-', linewidth=2, label='Fitted CDF')

ax4.set_xlabel('Value', fontsize=11)
ax4.set_ylabel('Cumulative Probability', fontsize=11)
ax4.set_title('CDF Comparison', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Add goodness-of-fit metrics
metrics_text = f"AIC: {dist.aic():.1f}\nBIC: {dist.bic():.1f}"
ax4.text(0.98, 0.02, metrics_text, transform=ax4.transAxes,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
         fontsize=10)

# =============================================================================
# Finalize and save
# =============================================================================
plt.tight_layout()

# Save figure
output_file = 'distribution_fit_diagnostics.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_file}")

# Show plot
print("\nDisplaying plot...")
plt.show()

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print("""
1. Histogram with PDF:
   - Should overlay closely if model fits well
   - Check for systematic deviations

2. Q-Q Plot:
   - Points should fall on red line for good fit
   - S-shape indicates heavy/light tails
   - Systematic curve indicates wrong distribution

3. P-P Plot:
   - Similar to Q-Q but focuses on probabilities
   - More sensitive to center of distribution
   - Points on line = good fit

4. CDF Comparison:
   - Empirical (step) should match theoretical (smooth)
   - Large gaps indicate poor fit
   - Use with Kolmogorov-Smirnov test
""")
print("="*70)
print("\nNext steps:")
print("  - Try different distributions to see poor fits")
print("  - Combine with statistical tests (see 03_gof_tests/)")
print("  - Export plots for reports/papers")
print("="*70)
