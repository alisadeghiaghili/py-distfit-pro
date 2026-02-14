#!/usr/bin/env python3
"""
PDF and CDF Plotting
===================

Create publication-quality probability plots:
  - PDF (Probability Density Function)
  - CDF (Cumulative Distribution Function)
  - Survival Function (1 - CDF)
  - Hazard Function
  - Multiple distributions comparison

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

print("="*70)
print("📊 PDF AND CDF PLOTTING GUIDE")
print("="*70)


# ============================================================================
# Example 1: Basic PDF Plot
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Basic PDF Plot")
print("="*70)

# Generate and fit data
data = np.random.normal(loc=100, scale=15, size=1000)
dist = get_distribution('normal')
dist.fit(data)

print(f"\n📊 Creating basic PDF plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of data
ax.hist(data, bins=50, density=True, alpha=0.6, color='skyblue', 
        edgecolor='black', label='Data Histogram')

# Fitted PDF
x = np.linspace(data.min(), data.max(), 300)
y_pdf = dist.pdf(x)
ax.plot(x, y_pdf, 'r-', linewidth=2.5, label='Fitted PDF')

# Add mean line
mean_val = dist.mean()
ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
           label=f'Mean = {mean_val:.2f}')

# Shade area under curve for probability
shade_start = mean_val - dist.std()
shade_end = mean_val + dist.std()
x_shade = x[(x >= shade_start) & (x <= shade_end)]
y_shade = dist.pdf(x_shade)
ax.fill_between(x_shade, 0, y_shade, alpha=0.3, color='yellow', 
                label='±1 std (68%)')

ax.set_xlabel('Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax.set_title('Probability Density Function (PDF)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("✅ Basic PDF plot created")
plt.savefig('/tmp/pdf_basic.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: CDF Plot with Percentiles
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: CDF with Percentiles")
print("="*70)

print(f"\n📊 Creating CDF plot with percentile markers...")

fig, ax = plt.subplots(figsize=(10, 6))

# Empirical CDF
data_sorted = np.sort(data)
empirical_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
ax.plot(data_sorted, empirical_cdf, 'o', markersize=2, alpha=0.3, 
        label='Empirical CDF', color='blue')

# Theoretical CDF
x_cdf = np.linspace(data.min(), data.max(), 500)
y_cdf = dist.cdf(x_cdf)
ax.plot(x_cdf, y_cdf, 'r-', linewidth=2.5, label='Theoretical CDF')

# Mark important percentiles
percentiles = [0.25, 0.5, 0.75, 0.95]
colors_p = ['orange', 'green', 'purple', 'red']

for p, color in zip(percentiles, colors_p):
    val = dist.ppf(p)
    ax.plot([val, val], [0, p], color=color, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot([data.min(), val], [p, p], color=color, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot(val, p, 'o', color=color, markersize=10, 
            label=f'P{int(p*100)} = {val:.1f}')

ax.set_xlabel('Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.05])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("✅ CDF plot with percentiles created")
plt.savefig('/tmp/cdf_percentiles.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 3: PDF + CDF Side-by-Side
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: PDF and CDF Side-by-Side")
print("="*70)

print(f"\n📊 Creating combined PDF/CDF plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: PDF
ax1.hist(data, bins=40, density=True, alpha=0.6, color='lightcoral', 
         edgecolor='black')
ax1.plot(x, dist.pdf(x), 'darkred', linewidth=2.5, label='PDF')
ax1.axvline(dist.mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
ax1.fill_between(x, 0, dist.pdf(x), alpha=0.2, color='red')
ax1.set_xlabel('Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('Probability Density Function', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right: CDF
ax2.plot(data_sorted, empirical_cdf, 'o', markersize=2, alpha=0.3, color='blue')
ax2.plot(x_cdf, y_cdf, 'darkblue', linewidth=2.5, label='CDF')
ax2.axhline(0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
ax2.axvline(dist.median(), color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Value', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
print("✅ Combined PDF/CDF plot created")
plt.savefig('/tmp/pdf_cdf_combined.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 4: Compare Multiple Distributions
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Compare Multiple Distributions")
print("="*70)

# Generate data from mixture
data_mixed = np.concatenate([
    np.random.normal(100, 10, 500),
    np.random.normal(130, 15, 300)
])

print(f"\n📊 Comparing multiple distribution fits...")

# Fit multiple distributions
dists_to_compare = [
    ('Normal', 'normal', 'blue'),
    ('t-distribution', 't', 'red'),
    ('Logistic', 'logistic', 'green'),
]

fitted_dists = []
for name, dist_name, color in dists_to_compare:
    d = get_distribution(dist_name)
    d.fit(data_mixed)
    fitted_dists.append((name, d, color))
    print(f"   ✓ Fitted {name}: AIC = {d.aic():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF comparison
ax = axes[0]
ax.hist(data_mixed, bins=50, density=True, alpha=0.4, color='gray', 
        edgecolor='black', label='Data')

x_comp = np.linspace(data_mixed.min(), data_mixed.max(), 300)
for name, d, color in fitted_dists:
    ax.plot(x_comp, d.pdf(x_comp), color=color, linewidth=2, 
            label=f'{name} (AIC={d.aic():.1f})')

ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('PDF Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# CDF comparison
ax = axes[1]
data_mixed_sorted = np.sort(data_mixed)
emp_cdf = np.arange(1, len(data_mixed_sorted) + 1) / len(data_mixed_sorted)
ax.plot(data_mixed_sorted, emp_cdf, 'o', markersize=2, alpha=0.3, 
        color='gray', label='Empirical')

for name, d, color in fitted_dists:
    ax.plot(x_comp, d.cdf(x_comp), color=color, linewidth=2, label=name)

ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
ax.set_title('CDF Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("✅ Multi-distribution comparison created")
plt.savefig('/tmp/multi_dist_comparison.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 5: Survival Function (Reliability)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Survival Function (1 - CDF)")
print("="*70)

# Generate failure time data
failure_data = np.random.weibull(1.5, 500) * 100
dist_weibull = get_distribution('weibull_min')
dist_weibull.fit(failure_data)

print(f"\n📊 Creating survival function plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Survival function
x_surv = np.linspace(0, failure_data.max(), 300)
y_surv = dist_weibull.sf(x_surv)  # Survival function = 1 - CDF

ax.plot(x_surv, y_surv, 'b-', linewidth=2.5, label='Survival Function S(t)')

# Mark important reliability milestones
for reliability in [0.9, 0.5, 0.1]:
    time_at_reliability = dist_weibull.ppf(1 - reliability)
    ax.plot([time_at_reliability, time_at_reliability], [0, reliability], 
            'r:', linewidth=1.5, alpha=0.7)
    ax.plot([0, time_at_reliability], [reliability, reliability], 
            'r:', linewidth=1.5, alpha=0.7)
    ax.plot(time_at_reliability, reliability, 'ro', markersize=8)
    ax.text(time_at_reliability, reliability + 0.05, 
            f'B{int((1-reliability)*100)}: {time_at_reliability:.1f}', 
            fontsize=9, ha='center')

ax.fill_between(x_surv, 0, y_surv, alpha=0.2, color='blue')

ax.set_xlabel('Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Reliability (Survival Probability)', fontsize=12, fontweight='bold')
ax.set_title('Survival Function: P(Survive > t)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("✅ Survival function plot created")
plt.savefig('/tmp/survival_function.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 6: Four-Panel Dashboard
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Complete Four-Panel Dashboard")
print("="*70)

print(f"\n📊 Creating comprehensive dashboard...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel 1: PDF
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax1.plot(x, dist.pdf(x), 'r-', linewidth=2)
ax1.axvline(dist.mean(), color='green', linestyle='--', linewidth=2)
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_title('A) Probability Density Function', fontsize=11, fontweight='bold', loc='left')
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel 2: CDF
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(data_sorted, empirical_cdf, 'o', markersize=1.5, alpha=0.4, color='blue')
ax2.plot(x_cdf, y_cdf, 'r-', linewidth=2)
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Cumulative Probability', fontsize=10)
ax2.set_title('B) Cumulative Distribution Function', fontsize=11, fontweight='bold', loc='left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.05])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel 3: Survival Function
ax3 = fig.add_subplot(gs[1, 0])
y_sf = dist.sf(x_cdf)
ax3.plot(x_cdf, y_sf, 'b-', linewidth=2)
ax3.fill_between(x_cdf, 0, y_sf, alpha=0.2, color='blue')
ax3.set_xlabel('Value', fontsize=10)
ax3.set_ylabel('Survival Probability', fontsize=10)
ax3.set_title('C) Survival Function (1 - CDF)', fontsize=11, fontweight='bold', loc='left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.05])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Panel 4: Statistics Summary
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = f"""
DISTRIBUTION SUMMARY
{'='*35}

Distribution: {dist.info.display_name}

Parameters:
{'-'*35}
"""

for param, value in dist.params.items():
    summary_text += f"  {param:12s} = {value:10.4f}\n"

summary_text += f"""
{'='*35}

Statistics:
{'-'*35}
  Mean         = {dist.mean():10.2f}
  Median       = {dist.median():10.2f}
  Std Dev      = {dist.std():10.2f}
  Variance     = {dist.var():10.2f}
  
Percentiles:
{'-'*35}
  25th         = {dist.ppf(0.25):10.2f}
  50th (Med)   = {dist.ppf(0.50):10.2f}
  75th         = {dist.ppf(0.75):10.2f}
  95th         = {dist.ppf(0.95):10.2f}

Model Fit:
{'-'*35}
  AIC          = {dist.aic():10.2f}
  BIC          = {dist.bic():10.2f}
  Log-Lik      = {dist.log_likelihood():10.2f}
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle('Complete Distribution Analysis Dashboard', 
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('/tmp/complete_dashboard.png', dpi=150, bbox_inches='tight')
print("✅ Complete dashboard created")


print("\n" + "="*70)
print("📁 Files Saved")
print("="*70)
print("""
Plots saved to /tmp/:
  1. pdf_basic.png
  2. cdf_percentiles.png
  3. pdf_cdf_combined.png
  4. multi_dist_comparison.png
  5. survival_function.png
  6. complete_dashboard.png
""")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. PDF (Probability Density Function):
   • Shows likelihood of values
   • Area under curve = probability
   • Best for: Understanding distribution shape

2. CDF (Cumulative Distribution Function):
   • Shows P(X ≤ x)
   • Always increases from 0 to 1
   • Best for: Finding percentiles, probabilities

3. Survival Function:
   • S(t) = 1 - CDF = P(X > t)
   • Used in reliability analysis
   • Shows probability of surviving past time t

4. VISUALIZATION BEST PRACTICES:
   ✓ Use histogram + fitted PDF together
   ✓ Mark important points (mean, percentiles)
   ✓ Show both empirical and theoretical curves
   ✓ Use clear labels and legends
   ✓ Remove unnecessary spines
   ✓ Add grid for readability

5. CUSTOMIZATION:
   • Colors: Use colorblind-friendly palettes
   • Transparency: alpha=0.3-0.6 for overlays
   • Line width: 2-2.5 for main curves
   • DPI: 150-300 for publication quality

6. COMMON PLOTS:
   • PDF alone: Show distribution shape
   • PDF + CDF: Complete picture
   • Multiple distributions: Model comparison
   • Dashboard: Comprehensive analysis

Next: See qq_pp_plots.py for goodness-of-fit visualization!
""")
