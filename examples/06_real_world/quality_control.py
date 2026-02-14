#!/usr/bin/env python3
"""
Quality Control: Statistical Process Control (SPC)
==================================================

Real-world quality control examples:
  - Process capability analysis (Cp, Cpk)
  - Control charts
  - Six Sigma methodology
  - Defect rate estimation

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("🏭 QUALITY CONTROL: STATISTICAL PROCESS CONTROL")
print("="*70)


# ============================================================================
# Example 1: Process Capability Analysis
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Process Capability Analysis")
print("="*70)

print("""
Scenario: Manufacturing bolt diameter
  - Specification: 10.0 ± 0.5 mm
  - Measure process capability
  - Is process meeting specifications?
""")

# Simulate manufacturing process
# Slightly off-center process
target = 10.0
USL = 10.5  # Upper Specification Limit
LSL = 9.5   # Lower Specification Limit

# Generate process data (slightly off-target)
process_mean = 10.05  # Slightly high
process_std = 0.12
measurements = np.random.normal(process_mean, process_std, 500)

print(f"\n📊 Process Data: {len(measurements)} measurements")
print(f"\n  Specifications:")
print(f"    Target:       {target:.2f} mm")
print(f"    USL:          {USL:.2f} mm")
print(f"    LSL:          {LSL:.2f} mm")
print(f"    Tolerance:    ±{(USL-target):.2f} mm")

print(f"\n  Process Statistics:")
print(f"    Mean:         {measurements.mean():.3f} mm")
print(f"    Std Dev:      {measurements.std():.3f} mm")
print(f"    Min:          {measurements.min():.3f} mm")
print(f"    Max:          {measurements.max():.3f} mm")

# Fit normal distribution
print(f"\n🔬 Fitting normal distribution...")
dist = get_distribution('normal')
dist.fit(measurements)

mean_fit = dist.mean()
std_fit = dist.std()

print(f"\n  Fitted parameters:")
print(f"    μ (mean):     {mean_fit:.4f} mm")
print(f"    σ (std dev):  {std_fit:.4f} mm")

# Calculate Process Capability Indices
print(f"\n📈 Process Capability Indices:")

# Cp: Process Capability (assumes centered process)
Cp = (USL - LSL) / (6 * std_fit)
print(f"\n  Cp  = (USL - LSL) / (6σ) = {Cp:.3f}")
if Cp >= 2.0:
    print(f"       ✅ EXCELLENT (Six Sigma capable)")
elif Cp >= 1.33:
    print(f"       ✅ GOOD (Process capable)")
elif Cp >= 1.0:
    print(f"       ⚠️  MARGINAL (Process barely capable)")
else:
    print(f"       ❌ POOR (Process not capable)")

# Cpk: Process Capability accounting for centering
Cpk_upper = (USL - mean_fit) / (3 * std_fit)
Cpk_lower = (mean_fit - LSL) / (3 * std_fit)
Cpk = min(Cpk_upper, Cpk_lower)

print(f"\n  Cpk = min[(USL - μ)/(3σ), (μ - LSL)/(3σ)] = {Cpk:.3f}")
if Cpk >= 1.67:
    print(f"       ✅ EXCELLENT (Six Sigma capable)")
elif Cpk >= 1.33:
    print(f"       ✅ GOOD (Process capable)")
elif Cpk >= 1.0:
    print(f"       ⚠️  MARGINAL (Process barely capable)")
else:
    print(f"       ❌ POOR (Process not capable)")

print(f"\n  Centering Index:")
centering = Cpk / Cp if Cp > 0 else 0
print(f"    Cpk/Cp = {centering:.3f}")
if centering >= 0.95:
    print(f"       ✅ Well centered")
else:
    print(f"       ⚠️  Process off-center (adjust mean!)")

# Calculate defect rate
print(f"\n🚨 Defect Rate Estimation:")

pct_below_LSL = dist.cdf(LSL) * 100
pct_above_USL = (1 - dist.cdf(USL)) * 100
pct_out_of_spec = pct_below_LSL + pct_above_USL

print(f"  Below LSL:     {pct_below_LSL:.4f}% ({pct_below_LSL*10000:.1f} PPM)")
print(f"  Above USL:     {pct_above_USL:.4f}% ({pct_above_USL*10000:.1f} PPM)")
print(f"  Total defects: {pct_out_of_spec:.4f}% ({pct_out_of_spec*10000:.1f} PPM)")

# Sigma level (Six Sigma methodology)
z_score = min((USL - mean_fit) / std_fit, (mean_fit - LSL) / std_fit)
sigma_level = z_score  # Simplified

print(f"\n  Sigma Level: {sigma_level:.2f}σ")
if sigma_level >= 6:
    print(f"       ⭐ Six Sigma quality (3.4 PPM)")
elif sigma_level >= 5:
    print(f"       ✅ Five Sigma (233 PPM)")
elif sigma_level >= 4:
    print(f"       🟡 Four Sigma (6,210 PPM)")
elif sigma_level >= 3:
    print(f"       🟠 Three Sigma (66,807 PPM)")
else:
    print(f"       🔴 Below Three Sigma")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Process Capability Analysis', fontsize=16, fontweight='bold')

# Plot 1: Process distribution with specs
ax = axes[0, 0]
ax.hist(measurements, bins=40, density=True, alpha=0.6, color='skyblue',
        edgecolor='black', label='Measurements')

x = np.linspace(measurements.min(), measurements.max(), 300)
ax.plot(x, dist.pdf(x), 'r-', linewidth=2.5, label='Fitted Distribution')

# Specification limits
ax.axvline(target, color='green', linestyle='-', linewidth=2, label='Target')
ax.axvline(USL, color='red', linestyle='--', linewidth=2, label='USL')
ax.axvline(LSL, color='red', linestyle='--', linewidth=2, label='LSL')
ax.axvspan(LSL, USL, alpha=0.1, color='green', label='Spec Range')

# Out of spec regions
ax.axvspan(measurements.min(), LSL, alpha=0.2, color='red')
ax.axvspan(USL, measurements.max(), alpha=0.2, color='red')

ax.set_xlabel('Measurement (mm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
ax.set_title('Process Distribution vs Specifications', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Capability summary
ax = axes[0, 1]
ax.axis('off')

summary_text = f"""
PROCESS CAPABILITY SUMMARY
{'='*45}

Specifications:
  Target:       {target:.2f} mm
  USL:          {USL:.2f} mm
  LSL:          {LSL:.2f} mm
  Tolerance:    ±{(USL-target):.2f} mm

Process Performance:
  Mean (μ):     {mean_fit:.4f} mm
  Std Dev (σ):  {std_fit:.4f} mm
  
Capability Indices:
  Cp:           {Cp:.3f}
  Cpk:          {Cpk:.3f}
  Centering:    {centering:.1%}
  
Defect Rate:
  Out of spec:  {pct_out_of_spec:.4f}%
                ({pct_out_of_spec*10000:.1f} PPM)
  Sigma Level:  {sigma_level:.2f}σ

Recommendation:
"""

if Cpk < 1.33:
    summary_text += "  ⚠️  IMPROVE PROCESS:"
    if centering < 0.95:
        summary_text += "\n  - Adjust process mean"
    if Cp < 1.33:
        summary_text += "\n  - Reduce process variation"
else:
    summary_text += "  ✅ Process is capable"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Plot 3: Individual measurements (run chart)
ax = axes[1, 0]
ax.plot(measurements, 'b-', linewidth=1, alpha=0.7, marker='o', markersize=3)
ax.axhline(mean_fit, color='green', linestyle='-', linewidth=2, label='Mean')
ax.axhline(USL, color='red', linestyle='--', linewidth=2, label='USL')
ax.axhline(LSL, color='red', linestyle='--', linewidth=2, label='LSL')
ax.axhline(mean_fit + 3*std_fit, color='orange', linestyle=':', linewidth=1.5, 
           alpha=0.7, label='+3σ')
ax.axhline(mean_fit - 3*std_fit, color='orange', linestyle=':', linewidth=1.5,
           alpha=0.7, label='-3σ')

ax.set_xlabel('Sample Number', fontsize=11, fontweight='bold')
ax.set_ylabel('Measurement (mm)', fontsize=11, fontweight='bold')
ax.set_title('Run Chart (Individual Measurements)', fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 4: Normal probability plot (Q-Q)
ax = axes[1, 1]
percentiles = np.linspace(0.01, 0.99, len(measurements))
theoretical_q = dist.ppf(percentiles)
empirical_q = np.sort(measurements)

ax.scatter(theoretical_q, empirical_q, alpha=0.6, s=15, color='blue')
min_v, max_v = min(theoretical_q.min(), empirical_q.min()), \
               max(theoretical_q.max(), empirical_q.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)

ax.set_xlabel('Theoretical Quantiles (mm)', fontsize=10)
ax.set_ylabel('Sample Quantiles (mm)', fontsize=10)
ax.set_title('Normal Probability Plot', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 Process capability analysis plots created!")
plt.savefig('/tmp/quality_process_capability.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: Control Charts (X-bar and R charts)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Control Charts (SPC)")
print("="*70)

print("""
Scenario: Monitor process over time using control charts
  - X-bar chart: Monitors process mean
  - R chart: Monitors process variability
""")

# Simulate subgroup data (5 samples per subgroup, 25 subgroups)
subgroup_size = 5
n_subgroups = 25

subgroups = []
for i in range(n_subgroups):
    # Add slight drift after subgroup 15
    if i > 15:
        subgroup = np.random.normal(10.1, 0.12, subgroup_size)  # Process shifted
    else:
        subgroup = np.random.normal(10.0, 0.12, subgroup_size)
    subgroups.append(subgroup)

subgroups = np.array(subgroups)

print(f"\n📊 Control Chart Data:")
print(f"  Subgroup size: {subgroup_size}")
print(f"  Number of subgroups: {n_subgroups}")
print(f"  Total samples: {subgroup_size * n_subgroups}")

# Calculate X-bar (mean) and R (range) for each subgroup
xbar = subgroups.mean(axis=1)
R = subgroups.max(axis=1) - subgroups.min(axis=1)

# Control limits
# X-bar chart
xbar_mean = xbar.mean()
R_mean = R.mean()

# Control chart constants (for n=5)
A2 = 0.577  # for X-bar chart
D3 = 0.0    # for R chart (lower)
D4 = 2.114  # for R chart (upper)

UCL_xbar = xbar_mean + A2 * R_mean
LCL_xbar = xbar_mean - A2 * R_mean

UCL_R = D4 * R_mean
LCL_R = D3 * R_mean

print(f"\n📊 X-bar Chart (Process Mean):")
print(f"  Center line (X-bar-bar): {xbar_mean:.4f} mm")
print(f"  UCL: {UCL_xbar:.4f} mm")
print(f"  LCL: {LCL_xbar:.4f} mm")

print(f"\n📊 R Chart (Process Variation):")
print(f"  Center line (R-bar): {R_mean:.4f} mm")
print(f"  UCL: {UCL_R:.4f} mm")
print(f"  LCL: {LCL_R:.4f} mm")

# Check for out-of-control points
out_of_control_xbar = (xbar > UCL_xbar) | (xbar < LCL_xbar)
out_of_control_R = (R > UCL_R) | (R < LCL_R)

print(f"\n🚨 Control Status:")
if np.any(out_of_control_xbar):
    ooc_indices = np.where(out_of_control_xbar)[0] + 1
    print(f"  ⚠️  X-bar chart: OUT OF CONTROL at subgroups {ooc_indices.tolist()}")
else:
    print(f"  ✅ X-bar chart: IN CONTROL")

if np.any(out_of_control_R):
    ooc_indices = np.where(out_of_control_R)[0] + 1
    print(f"  ⚠️  R chart: OUT OF CONTROL at subgroups {ooc_indices.tolist()}")
else:
    print(f"  ✅ R chart: IN CONTROL")

# Visualization: Control Charts
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Statistical Process Control Charts', fontsize=16, fontweight='bold')

# X-bar Chart
ax1.plot(range(1, n_subgroups + 1), xbar, 'b-o', linewidth=2, markersize=6,
         label='Subgroup Mean')
ax1.axhline(xbar_mean, color='green', linestyle='-', linewidth=2, label='Center Line')
ax1.axhline(UCL_xbar, color='red', linestyle='--', linewidth=2, label='UCL')
ax1.axhline(LCL_xbar, color='red', linestyle='--', linewidth=2, label='LCL')

# Mark out-of-control points
if np.any(out_of_control_xbar):
    ax1.scatter(np.where(out_of_control_xbar)[0] + 1, 
                xbar[out_of_control_xbar],
                color='red', s=150, marker='x', linewidths=3, zorder=5,
                label='Out of Control')

ax1.fill_between(range(1, n_subgroups + 1), LCL_xbar, UCL_xbar, 
                 alpha=0.1, color='green')
ax1.set_ylabel('X-bar (mm)', fontsize=11, fontweight='bold')
ax1.set_title('X-bar Chart: Process Mean', fontweight='bold', loc='left')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# R Chart
ax2.plot(range(1, n_subgroups + 1), R, 'b-o', linewidth=2, markersize=6,
         label='Subgroup Range')
ax2.axhline(R_mean, color='green', linestyle='-', linewidth=2, label='Center Line')
ax2.axhline(UCL_R, color='red', linestyle='--', linewidth=2, label='UCL')
ax2.axhline(LCL_R, color='red', linestyle='--', linewidth=2, label='LCL')

# Mark out-of-control points
if np.any(out_of_control_R):
    ax2.scatter(np.where(out_of_control_R)[0] + 1,
                R[out_of_control_R],
                color='red', s=150, marker='x', linewidths=3, zorder=5,
                label='Out of Control')

ax2.fill_between(range(1, n_subgroups + 1), LCL_R, UCL_R,
                 alpha=0.1, color='green')
ax2.set_xlabel('Subgroup Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('Range (mm)', fontsize=11, fontweight='bold')
ax2.set_title('R Chart: Process Variation', fontweight='bold', loc='left')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 Control charts created!")
plt.savefig('/tmp/quality_control_charts.png', dpi=150, bbox_inches='tight')

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways - Quality Control")
print("="*70)
print("""
1. PROCESS CAPABILITY INDICES:
   • Cp:  Measures potential capability (assumes centered)
   • Cpk: Measures actual capability (accounts for centering)
   • Target values:
     - Cpk ≥ 1.67: Six Sigma capable
     - Cpk ≥ 1.33: Process capable
     - Cpk < 1.0:  Process not capable

2. CENTERING:
   • Cpk/Cp ratio indicates centering
   • If Cp >> Cpk: Process off-center (adjust mean)
   • If Cp ≈ Cpk: Process well-centered

3. DEFECT RATES:
   • PPM = Parts Per Million
   • Six Sigma: 3.4 PPM (99.99966% good)
   • Three Sigma: 66,807 PPM (93.3% good)

4. CONTROL CHARTS:
   • X-bar chart: Monitors process mean
   • R chart: Monitors process variability
   • Both must be in control!
   • Check R chart first (variation affects mean chart)

5. OUT-OF-CONTROL SIGNALS:
   • Point beyond control limits
   • 7+ consecutive points on one side of center
   • Trends, cycles, patterns
   • Investigate and take corrective action

6. IMPROVEMENT ACTIONS:
   If Cpk too low:
     1. Check centering (adjust mean if needed)
     2. Reduce variation (improve process)
     3. Widen specifications (last resort!)
   
   If out of control:
     1. Identify special causes
     2. Eliminate or control them
     3. Update control limits after improvement

7. BEST PRACTICES:
   ✓ Verify normality assumption (Q-Q plot)
   ✓ Use rational subgroups
   ✓ Monitor both mean and variation
   ✓ React to signals promptly
   ✓ Update limits periodically
   ✓ Focus on prevention, not detection

8. REAL-WORLD IMPACT:
   • Reduce defects and rework
   • Lower costs
   • Increase customer satisfaction
   • Competitive advantage

Next: See 07_advanced_topics/ for advanced techniques!
""")
