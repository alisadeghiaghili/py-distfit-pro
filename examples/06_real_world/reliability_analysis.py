#!/usr/bin/env python3
"""
Reliability Engineering Example
===============================

Analyze component lifetimes and failure rates.
Common in: Manufacturing, Aerospace, Automotive.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("🔧 RELIABILITY ENGINEERING: Component Lifetime Analysis")
print("="*70)
print("""
Scenario: Analyzing electronic component failures
  - Company manufactures LED drivers
  - Testing shows varying failure times
  - Need to: estimate warranty costs, improve reliability
""")


# ============================================================================
# Data: Component Failure Times (hours)
# ============================================================================

# Simulated failure times (Weibull is common for reliability)
np.random.seed(123)
failure_times = np.random.weibull(a=2.5, size=200) * 10000

print(f"\n📊 Test Data: {len(failure_times)} components tested to failure")
print(f"  Mean lifetime: {failure_times.mean():.0f} hours")
print(f"  Median:        {np.median(failure_times):.0f} hours")
print(f"  Min/Max:       [{failure_times.min():.0f}, {failure_times.max():.0f}] hours")


# ============================================================================
# Fit Weibull Distribution
# ============================================================================

print("\n" + "="*70)
print("📊 Fitting Weibull Distribution")
print("="*70)
print("""
Weibull is THE standard for reliability engineering because:
  • Shape parameter (k) indicates failure mode:
    - k < 1: Infant mortality (early failures)
    - k = 1: Random failures (constant hazard)
    - k > 1: Wear-out failures (aging)
  • Scale parameter (λ) relates to characteristic life
""")

dist = get_distribution('weibull_min')
dist.fit(failure_times)

print("\n" + dist.summary())

# Extract shape parameter
shape_param = dist.params.get('c', dist.params.get('shape', 1))
scale_param = dist.params.get('scale', 1)

print(f"\n🔍 Interpretation:")
print(f"  Shape (k) = {shape_param:.2f}")
if shape_param < 1:
    print("    → k < 1: INFANT MORTALITY (early failures)")
    print("       Recommendation: Improve manufacturing, add burn-in testing")
elif shape_param > 1:
    print("    → k > 1: WEAR-OUT failures (aging/degradation)")
    print("       Recommendation: Preventive maintenance, design improvements")
else:
    print("    → k ≈ 1: RANDOM failures (exponential-like)")
    print("       Recommendation: Monitor for external factors")

print(f"\n  Scale (λ) = {scale_param:.0f} hours")
print(f"    → Characteristic life: 63.2% fail by {scale_param:.0f} hours")


# ============================================================================
# Reliability Metrics
# ============================================================================

print("\n" + "="*70)
print("📈 Key Reliability Metrics")
print("="*70)

# 1. Mean Time To Failure (MTTF)
mttf = dist.mean()
print(f"\n1️⃣  MTTF (Mean Time To Failure): {mttf:.0f} hours")
print(f"    Expected lifetime before failure")

# 2. Reliability at specific times
times_of_interest = [5000, 8000, 10000, 15000]  # hours
print(f"\n2️⃣  Reliability Function R(t) = P(T > t):")
for t in times_of_interest:
    reliability = dist.sf(t)  # Survival function = 1 - CDF
    print(f"    R({t:5d}h) = {reliability:.4f}  ({reliability*100:.2f}% survive)")

# 3. Failure rate (hazard function)
print(f"\n3️⃣  Failure Rate (Hazard):")
if shape_param < 1:
    print(f"    DECREASING (shape={shape_param:.2f} < 1)")
    print(f"    → Failure risk decreases over time (debugging effect)")
elif shape_param > 1:
    print(f"    INCREASING (shape={shape_param:.2f} > 1)")
    print(f"    → Failure risk increases over time (wear-out)")
else:
    print(f"    CONSTANT (shape={shape_param:.2f} ≈ 1)")
    print(f"    → Memoryless (exponential behavior)")

# 4. Warranty period analysis
warranty_period = 8000  # hours
failures_in_warranty = dist.cdf(warranty_period)
print(f"\n4️⃣  Warranty Analysis ({warranty_period} hours):")
print(f"    Failure rate in warranty: {failures_in_warranty*100:.2f}%")
print(f"    If selling 10,000 units: expect {int(failures_in_warranty * 10000)} warranty claims")

# 5. Percentile lifetimes
percentiles = [0.01, 0.05, 0.10, 0.50, 0.90]
print(f"\n5️⃣  Lifetime Percentiles:")
for p in percentiles:
    lifetime = dist.ppf(p)
    print(f"    {p*100:5.1f}% fail by {lifetime:8.0f} hours")


# ============================================================================
# Business Recommendations
# ============================================================================

print("\n" + "="*70)
print("💼 Business Recommendations")
print("="*70)

# Calculate optimal warranty period (balance cost vs customer satisfaction)
warranty_options = [5000, 8000, 10000, 12000]
print("\nWarranty Period Options:")
print("\n  Period (h)  Failure Rate  Expected Claims (per 10k units)")
print("  " + "-"*60)
for w in warranty_options:
    fail_rate = dist.cdf(w)
    claims = int(fail_rate * 10000)
    print(f"    {w:5d}       {fail_rate*100:5.2f}%              {claims:5d}")

print(f"\n✅ Recommendation:")
print(f"  - Current MTTF: {mttf:.0f} hours")
print(f"  - For <5% failure rate: Warranty should be < {dist.ppf(0.05):.0f} hours")
print(f"  - For <10% failure rate: Warranty should be < {dist.ppf(0.10):.0f} hours")

if shape_param > 1:
    print(f"\n  ⚠️  Warning: Shape > 1 indicates wear-out failures")
    print(f"     Consider: Preventive replacement at {dist.ppf(0.90):.0f} hours")


# ============================================================================
# Visualization: Reliability Functions
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Reliability Dashboard...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. PDF (Failure Density)
ax = axes[0, 0]
x = np.linspace(0, failure_times.max(), 500)
y_pdf = dist.pdf(x)
ax.hist(failure_times, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')
ax.plot(x, y_pdf, 'r-', linewidth=2, label='Weibull fit')
ax.axvline(mttf, color='green', linestyle='--', linewidth=2, label=f'MTTF={mttf:.0f}h')
ax.set_xlabel('Time to Failure (hours)', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.set_title('PDF: Failure Time Distribution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Reliability Function R(t) = 1 - F(t)
ax = axes[0, 1]
y_reliability = dist.sf(x)  # Survival function
ax.plot(x, y_reliability, 'b-', linewidth=2)
for t in [5000, 10000]:
    r = dist.sf(t)
    ax.plot(t, r, 'ro', markersize=8)
    ax.text(t, r+0.05, f'{r*100:.1f}%', ha='center')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Reliability R(t)', fontsize=11)
ax.set_title('Reliability Function: P(survive beyond t)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# 3. Cumulative Failure CDF
ax = axes[1, 0]
y_cdf = dist.cdf(x)
ax.plot(x, y_cdf, 'g-', linewidth=2)
ax.axhline(failures_in_warranty, color='red', linestyle='--', 
           label=f'Warranty threshold ({failures_in_warranty*100:.1f}%)')
ax.axvline(warranty_period, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Cumulative Failure Probability', fontsize=11)
ax.set_title('CDF: Fraction Failed by Time t', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Hazard Function (approximate)
ax = axes[1, 1]
# Hazard h(t) = f(t) / R(t)
y_hazard = y_pdf / (y_reliability + 1e-10)  # avoid division by zero
ax.plot(x, y_hazard, 'purple', linewidth=2)
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Hazard Rate h(t)', fontsize=11)
ax.set_title(f'Hazard Function (shape={shape_param:.2f})', fontweight='bold')
ax.grid(True, alpha=0.3)

if shape_param < 1:
    ax.text(0.5, 0.9, 'Decreasing hazard\n(infant mortality)', 
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
elif shape_param > 1:
    ax.text(0.5, 0.9, 'Increasing hazard\n(wear-out)', 
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

plt.suptitle('Reliability Analysis Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()

print("\n✅ Dashboard created!")
plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Weibull distribution is THE standard for reliability engineering

2. Shape parameter (k) reveals failure mode:
   k < 1: Early failures (quality issues)
   k = 1: Random failures  
   k > 1: Wear-out (aging)

3. Key metrics:
   - MTTF: Average lifetime
   - R(t): Probability of surviving to time t
   - h(t): Instantaneous failure rate

4. Use for:
   - Warranty period optimization
   - Preventive maintenance scheduling
   - Design improvement priorities

Next: See financial_risk.py for Value-at-Risk calculations!
""")
