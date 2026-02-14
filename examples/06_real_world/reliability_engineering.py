#!/usr/bin/env python3
"""
Reliability Engineering: Failure Time Analysis
==============================================

Real-world reliability examples:
  - Component failure analysis
  - Weibull distribution (gold standard)
  - Mean Time Between Failures (MTBF)
  - Reliability prediction
  - Maintenance planning

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("⚙️ RELIABILITY ENGINEERING: FAILURE TIME ANALYSIS")
print("="*70)


# ============================================================================
# Example 1: Component Failure Analysis (Weibull)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Bearing Failure Analysis")
print("="*70)

print("""
Scenario: Analyze bearing failures in manufacturing equipment
  - Time to failure data (hours)
  - Fit Weibull distribution
  - Calculate reliability metrics
  - Plan maintenance schedule
""")

# Simulate realistic bearing failure times
# Weibull with shape > 1 indicates wear-out failures
shape_true = 2.5  # Wear-out regime
scale_true = 5000  # Characteristic life (hours)
failure_times = np.random.weibull(shape_true, 200) * scale_true

print(f"\n📊 Failure Data: {len(failure_times)} bearings tested")
print(f"  Mean time to failure: {failure_times.mean():.0f} hours")
print(f"  Median failure time:  {np.median(failure_times):.0f} hours")
print(f"  Min failure:          {failure_times.min():.0f} hours")
print(f"  Max failure:          {failure_times.max():.0f} hours")

# Fit Weibull distribution
print("\n🔬 Fitting Weibull distribution...")
dist_weibull = get_distribution('weibull_min')
dist_weibull.fit(failure_times)

print(f"\n📈 Fitted Weibull Parameters:")
shape_fitted = dist_weibull.params['c']
scale_fitted = dist_weibull.params['scale']
print(f"  Shape (β):  {shape_fitted:.3f}")
print(f"  Scale (η):  {scale_fitted:.0f} hours")
print(f"  Location:   {dist_weibull.params.get('loc', 0):.0f}")

print(f"\n🔍 Interpretation:")
if shape_fitted < 1:
    print(f"  β < 1: INFANT MORTALITY (decreasing failure rate)")
    print(f"         Early failures, quality issues")
elif shape_fitted > 1:
    print(f"  β > 1: WEAR-OUT (increasing failure rate)")
    print(f"         Component aging, replace before failure")
else:
    print(f"  β ≈ 1: RANDOM FAILURES (constant failure rate)")
    print(f"         No predictable pattern")

# Calculate reliability metrics
print(f"\n⚡ Reliability Metrics:")

# MTBF (Mean Time Between Failures)
mtbf = dist_weibull.mean()
print(f"\n  MTBF: {mtbf:.0f} hours")

# B-life: time by which X% will fail
for pct in [10, 50, 90]:
    b_life = dist_weibull.ppf(pct/100)
    print(f"  B{pct:02d}:  {b_life:.0f} hours ({pct}% will fail by this time)")

# Reliability at specific times
print(f"\n  Reliability at key times:")
for t in [1000, 3000, 5000, 7000]:
    reliability = dist_weibull.sf(t)  # Survival function
    print(f"    t={t:5d}h: R(t) = {reliability:.3f} ({reliability*100:.1f}%)")

# Hazard rate (instantaneous failure rate)
print(f"\n  Hazard Rate h(t) = (β/η) * (t/η)^(β-1):")
for t in [1000, 3000, 5000, 7000]:
    hazard = (shape_fitted / scale_fitted) * (t / scale_fitted)**(shape_fitted - 1)
    print(f"    t={t:5d}h: h(t) = {hazard:.6f} failures/hour")

# Maintenance recommendation
print(f"\n📋 Maintenance Recommendation:")
target_reliability = 0.90  # Want 90% reliability
maintenance_interval = dist_weibull.ppf(1 - target_reliability)
print(f"  Replace bearings every {maintenance_interval:.0f} hours")
print(f"  to maintain {target_reliability*100:.0f}% reliability")
print(f"  (Based on B10 = {dist_weibull.ppf(0.10):.0f} hours)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bearing Reliability Analysis (Weibull)', fontsize=16, fontweight='bold')

# Plot 1: PDF with failure data
ax = axes[0, 0]
ax.hist(failure_times, bins=30, density=True, alpha=0.6, color='skyblue',
        edgecolor='black', label='Failure Data')
x = np.linspace(0, failure_times.max(), 300)
ax.plot(x, dist_weibull.pdf(x), 'r-', linewidth=2.5, label='Weibull Fit')
ax.axvline(mtbf, color='green', linestyle='--', linewidth=2, label=f'MTBF = {mtbf:.0f}h')
ax.set_xlabel('Time to Failure (hours)', fontsize=11, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax.set_title('Failure Time Distribution', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Reliability Function R(t)
ax = axes[0, 1]
t_plot = np.linspace(0, failure_times.max(), 300)
reliability_curve = dist_weibull.sf(t_plot)
ax.plot(t_plot, reliability_curve, 'b-', linewidth=2.5)
ax.axhline(0.90, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
           label='90% Reliability')
ax.axvline(maintenance_interval, color='red', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Maintenance at {maintenance_interval:.0f}h')
ax.fill_between(t_plot, 0, reliability_curve, alpha=0.2, color='blue')
ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax.set_ylabel('Reliability R(t)', fontsize=11, fontweight='bold')
ax.set_title('Reliability Function', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 3: Hazard Rate h(t)
ax = axes[1, 0]
hazard_curve = (shape_fitted / scale_fitted) * (t_plot / scale_fitted)**(shape_fitted - 1)
ax.plot(t_plot, hazard_curve, 'r-', linewidth=2.5)
ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax.set_ylabel('Hazard Rate h(t)', fontsize=11, fontweight='bold')
ax.set_title(f'Hazard Rate (β={shape_fitted:.2f} > 1: Wear-Out)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.fill_between(t_plot, 0, hazard_curve, alpha=0.2, color='red')

# Add text annotation
if shape_fitted > 1:
    ax.text(0.5, 0.95, 'Increasing hazard rate\n→ Preventive maintenance needed',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            ha='center')

# Plot 4: Q-Q Plot
ax = axes[1, 1]
percentiles = np.linspace(0.01, 0.99, len(failure_times))
theoretical_q = dist_weibull.ppf(percentiles)
empirical_q = np.sort(failure_times)
ax.scatter(theoretical_q, empirical_q, alpha=0.6, s=20, color='green')
min_v, max_v = min(theoretical_q.min(), empirical_q.min()), \
               max(theoretical_q.max(), empirical_q.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
ax.set_xlabel('Theoretical Quantiles (hours)', fontsize=10)
ax.set_ylabel('Sample Quantiles (hours)', fontsize=10)
ax.set_title('Q-Q Plot: Goodness of Fit', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 Reliability analysis plots created!")
plt.savefig('/tmp/reliability_weibull_analysis.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: System Reliability (Series vs Parallel)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: System Reliability Configuration")
print("="*70)

print("""
Scenario: Compare system configurations
  - Series: System fails if ANY component fails
  - Parallel: System fails if ALL components fail
""")

# Component reliability
component_reliability = 0.90  # Each component has 90% reliability
n_components = 5

# Series system: R_system = R1 × R2 × ... × Rn
R_series = component_reliability ** n_components

# Parallel system: R_system = 1 - (1-R1) × (1-R2) × ... × (1-Rn)
R_parallel = 1 - (1 - component_reliability) ** n_components

print(f"\n⚙️ System Configuration Analysis:")
print(f"  Component reliability: {component_reliability:.2f} ({component_reliability*100:.0f}%)")
print(f"  Number of components:  {n_components}")
print(f"\n  Series system:   R = {R_series:.4f} ({R_series*100:.2f}%)")
print(f"  Parallel system: R = {R_parallel:.6f} ({R_parallel*100:.4f}%)")

print(f"\n💡 Insight:")
print(f"  Series reduces reliability (weakest link)")
print(f"  Parallel increases reliability (redundancy)")
print(f"  Parallel is {R_parallel/R_series:.1f}x more reliable!")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

n_range = np.arange(1, 11)
R_series_curve = component_reliability ** n_range
R_parallel_curve = 1 - (1 - component_reliability) ** n_range

ax.plot(n_range, R_series_curve, 'r-o', linewidth=2.5, markersize=8,
        label='Series Configuration', markerfacecolor='red', markeredgecolor='black')
ax.plot(n_range, R_parallel_curve, 'b-s', linewidth=2.5, markersize=8,
        label='Parallel Configuration', markerfacecolor='blue', markeredgecolor='black')

ax.axhline(component_reliability, color='green', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Single Component ({component_reliability:.0%})')

ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax.set_ylabel('System Reliability', fontsize=12, fontweight='bold')
ax.set_title('System Reliability: Series vs Parallel', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("\n📊 System configuration comparison plotted!")
plt.savefig('/tmp/reliability_system_config.png', dpi=150, bbox_inches='tight')

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways - Reliability Engineering")
print("="*70)
print("""
1. WEIBULL DISTRIBUTION:
   • Gold standard for reliability analysis
   • Shape parameter β indicates failure mode:
     - β < 1: Infant mortality (early failures)
     - β = 1: Random failures (exponential)
     - β > 1: Wear-out (aging)
   • Scale parameter η: Characteristic life

2. KEY METRICS:
   • MTBF: Mean Time Between Failures
   • B10, B50, etc.: Time by which X% fail
   • R(t): Reliability at time t
   • h(t): Hazard rate (instantaneous failure rate)

3. MAINTENANCE STRATEGY:
   • β < 1: Improve quality, burn-in testing
   • β ≈ 1: Run to failure, reactive maintenance
   • β > 1: Preventive maintenance critical!
   • Schedule replacement before B10

4. SYSTEM RELIABILITY:
   • Series: Reduces reliability
   • Parallel: Increases reliability (redundancy)
   • Cost vs reliability trade-off

5. PRACTICAL APPLICATIONS:
   • Manufacturing equipment maintenance
   • Product warranty analysis
   • Spare parts inventory planning
   • Design for reliability

6. BEST PRACTICES:
   ✓ Use Weibull for time-to-failure data
   ✓ Check shape parameter for failure mode
   ✓ Set maintenance based on target reliability
   ✓ Use Q-Q plot to validate fit
   ✓ Update models with new failure data

Next: See quality_control.py!
""")
