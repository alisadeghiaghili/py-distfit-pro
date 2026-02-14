#!/usr/bin/env python3
"""
PDF and CDF Plotting
===================

Create publication-quality probability plots.

Author: Ali Sadeghi Aghili
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("📈 PDF & CDF VISUALIZATION")
print("="*70)

# Generate data
data = np.random.gamma(shape=2, scale=3, size=1000)

# Fit distribution
dist = get_distribution('gamma')
dist.fit(data)

print(f"\n✅ Fitted Gamma distribution")
print(f"   Mean: {dist.mean():.2f}")
print(f"   Std:  {dist.std():.2f}")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))

# Plot 1: PDF with histogram
ax1 = plt.subplot(2, 3, 1)
ax1.hist(data, bins=40, density=True, alpha=0.6, color='skyblue', 
         edgecolor='black', label='Data')

x = np.linspace(0, data.max(), 200)
y_pdf = dist.pdf(x)
ax1.plot(x, y_pdf, 'r-', linewidth=2.5, label='Fitted PDF')

# Add mean and median
ax1.axvline(dist.mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Mean={dist.mean():.1f}')
ax1.axvline(dist.median(), color='orange', linestyle=':', linewidth=2,
            label=f'Median={dist.median():.1f}')

ax1.set_xlabel('Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax1.set_title('PDF: Probability Density Function', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: CDF
ax2 = plt.subplot(2, 3, 2)
data_sorted = np.sort(data)
empirical_cdf = np.arange(1, len(data)+1) / len(data)

ax2.plot(data_sorted, empirical_cdf, 'o', markersize=2, alpha=0.4,
         label='Empirical CDF', color='blue')

y_cdf = dist.cdf(x)
ax2.plot(x, y_cdf, 'r-', linewidth=2.5, label='Theoretical CDF')

# Add percentiles
for p in [0.25, 0.5, 0.75, 0.95]:
    val = dist.ppf(p)
    ax2.plot([0, val, val], [p, p, 0], 'k:', alpha=0.3, linewidth=1)
    ax2.plot(val, p, 'go', markersize=6)
    ax2.text(val*1.02, p, f'p{int(p*100)}', fontsize=8, va='center')

ax2.set_xlabel('Value', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax2.set_title('CDF: Cumulative Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])

# Plot 3: Survival Function
ax3 = plt.subplot(2, 3, 3)
y_sf = dist.sf(x)  # Survival = 1 - CDF
ax3.plot(x, y_sf, 'b-', linewidth=2.5, label='Survival Function')

ax3.set_xlabel('Value', fontsize=11, fontweight='bold')
ax3.set_ylabel('Survival Probability', fontsize=11, fontweight='bold')
ax3.set_title('SF: Survival Function (1 - CDF)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.05])

# Plot 4: Log-scale PDF
ax4 = plt.subplot(2, 3, 4)
ax4.hist(data, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax4.plot(x, y_pdf, 'r-', linewidth=2.5)
ax4.set_xlabel('Value', fontsize=11, fontweight='bold')
ax4.set_ylabel('Probability Density (log)', fontsize=11, fontweight='bold')
ax4.set_title('PDF: Log Scale Y-axis', fontsize=12, fontweight='bold')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# Plot 5: Hazard Function
ax5 = plt.subplot(2, 3, 5)
y_hazard = y_pdf / y_sf
ax5.plot(x, y_hazard, 'purple', linewidth=2.5, label='Hazard Rate')

ax5.set_xlabel('Value', fontsize=11, fontweight='bold')
ax5.set_ylabel('Hazard Rate', fontsize=11, fontweight='bold')
ax5.set_title('Hazard Function h(x) = f(x)/S(x)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Probability summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary = f"""
PROBABILITY SUMMARY
{'='*35}

Distribution: {dist.info.display_name}

Location Statistics:
{'-'*35}
  Mean:       {dist.mean():>12.2f}
  Median:     {dist.median():>12.2f}
  Mode:       {'N/A':>12s}
  Std Dev:    {dist.std():>12.2f}
  
Percentiles:
{'-'*35}
  P25:        {dist.ppf(0.25):>12.2f}
  P50:        {dist.ppf(0.50):>12.2f}
  P75:        {dist.ppf(0.75):>12.2f}
  P95:        {dist.ppf(0.95):>12.2f}
  
Tail Probabilities:
{'-'*35}
  P(X > mean):  {dist.sf(dist.mean())*100:>8.1f}%
  P(X > P95):   {dist.sf(dist.ppf(0.95))*100:>8.1f}%
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Complete Probability Distribution Visualization', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

print("\n✅ Comprehensive probability plots created!")
plt.show()

print("\nNext: See qq_pp_plots.py for diagnostic plots!")
