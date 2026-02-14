#!/usr/bin/env python3
"""
Quick Visualization Examples
===========================

Create beautiful plots to visualize your fitted distributions.

Author: Ali Sadeghi Aghili
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

# Set style for prettier plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# ============================================================================
# Generate Sample Data
# ============================================================================

np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data)

print("="*70)
print("📊 Creating Visualizations...")
print("="*70)


# ============================================================================
# PLOT 1: Histogram + PDF Overlay
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram of data
ax.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', 
        edgecolor='black', label='Observed Data')

# Overlay fitted PDF
x = np.linspace(data.min(), data.max(), 200)
y_pdf = dist.pdf(x)
ax.plot(x, y_pdf, 'r-', linewidth=2, label=f'Fitted {dist.info.display_name}')

# Add mean line
mean_val = dist.mean()
ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
           label=f'Mean = {mean_val:.2f}')

ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Distribution Fitting: Histogram + PDF', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n✅ Plot 1: Histogram + PDF created")
print("   → Shows how well the distribution fits your data")


# ============================================================================
# PLOT 2: CDF (Cumulative Distribution Function)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Empirical CDF from data
data_sorted = np.sort(data)
empirical_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
ax.plot(data_sorted, empirical_cdf, 'o', markersize=2, alpha=0.5, 
        label='Empirical CDF', color='blue')

# Theoretical CDF from fitted distribution
x = np.linspace(data.min(), data.max(), 500)
y_cdf = dist.cdf(x)
ax.plot(x, y_cdf, 'r-', linewidth=2, label='Fitted CDF')

# Add percentile markers
for p in [0.25, 0.5, 0.75]:
    val = dist.ppf(p)
    ax.axvline(val, color='gray', linestyle=':', alpha=0.6)
    ax.axhline(p, color='gray', linestyle=':', alpha=0.6)
    ax.plot(val, p, 'go', markersize=8)
    ax.text(val, p + 0.03, f'p{int(p*100)}', ha='center', fontsize=9)

ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n✅ Plot 2: CDF created")
print("   → Shows cumulative probabilities (percentiles)")


# ============================================================================
# PLOT 3: Q-Q Plot (Quantile-Quantile)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Theoretical quantiles
percentiles = np.linspace(0.01, 0.99, len(data))
theoretical_quantiles = dist.ppf(percentiles)

# Empirical quantiles (sorted data)
empirical_quantiles = np.sort(data)

# Q-Q plot
ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

# Perfect fit line (45-degree)
min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
        label='Perfect Fit Line')

ax.set_xlabel('Theoretical Quantiles', fontsize=12)
ax.set_ylabel('Empirical Quantiles', fontsize=12)
ax.set_title('Q-Q Plot: Goodness of Fit Check', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
print("\n✅ Plot 3: Q-Q Plot created")
print("   → Points close to red line = good fit")


# ============================================================================
# PLOT 4: Multi-Panel Dashboard
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Panel 1: PDF
ax1 = plt.subplot(2, 2, 1)
ax1.hist(data, bins=40, density=True, alpha=0.7, color='skyblue', edgecolor='black')
x_range = np.linspace(data.min(), data.max(), 200)
ax1.plot(x_range, dist.pdf(x_range), 'r-', linewidth=2)
ax1.set_title('PDF (Probability Density)', fontweight='bold')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.grid(True, alpha=0.3)

# Panel 2: CDF
ax2 = plt.subplot(2, 2, 2)
ax2.plot(np.sort(data), np.arange(1, len(data)+1)/len(data), 
         'o', markersize=2, alpha=0.5, label='Empirical')
ax2.plot(x_range, dist.cdf(x_range), 'r-', linewidth=2, label='Fitted')
ax2.set_title('CDF (Cumulative)', fontweight='bold')
ax2.set_xlabel('Value')
ax2.set_ylabel('Cumulative Probability')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Box Plot
ax3 = plt.subplot(2, 2, 3)
box = ax3.boxplot([data], vert=True, patch_artist=True)
for patch in box['boxes']:
    patch.set_facecolor('lightblue')
ax3.set_title('Box Plot (Data Distribution)', fontweight='bold')
ax3.set_ylabel('Value')
ax3.set_xticklabels(['Data'])
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Summary Statistics
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

summary_text = f"""
DISTRIBUTION SUMMARY
{'='*30}

Distribution: {dist.info.display_name}

Fitted Parameters:
{'-'*30}
"""

for param, value in dist.params.items():
    summary_text += f"  {param:10s} = {value:>12.4f}\n"

summary_text += f"""
{'='*30}

Statistics:
{'-'*30}
  Mean       = {dist.mean():>12.2f}
  Median     = {dist.median():>12.2f}
  Std Dev    = {dist.std():>12.2f}
  
Model Fit:
{'-'*30}
  AIC        = {dist.aic():>12.2f}
  BIC        = {dist.bic():>12.2f}
  Log-Lik    = {dist.log_likelihood():>12.2f}
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Distribution Fitting Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

print("\n✅ Plot 4: Multi-panel dashboard created")
print("   → Complete overview in one figure")


print("\n" + "="*70)
print("🎨 All Plots Created!")
print("="*70)
print("""
Tip: Close plot windows or use plt.show() to view them interactively.

Next steps:
  - Explore 02_distributions/ for more distribution types
  - Check 05_visualization/ for advanced plotting techniques
  - See docs/guides/visualization_guide.md for best practices
""")

plt.show()
