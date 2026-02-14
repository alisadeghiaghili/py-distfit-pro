#!/usr/bin/env python3
"""
Quality Control & SPC Example
============================

Statistical Process Control (SPC) for manufacturing.
Common in: Manufacturing, Production, Six Sigma.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("🏭 QUALITY CONTROL: Statistical Process Control (SPC)")
print("="*70)
print("""
Scenario: Manufacturing precision parts
  - Target dimension: 50.00 mm
  - Tolerance: 50.00 ± 0.30 mm (Upper/Lower Spec Limits)
  - Goal: Minimize defects, ensure process capability
""")


# ============================================================================
# Data: Measurement Samples from Production Line
# ============================================================================

# Simulate measurements (normal process with slight shift)
n_samples = 500
target = 50.0
tolerance = 0.30

# Process is slightly off-center (50.05 instead of 50.00)
measurements = np.random.normal(loc=50.05, scale=0.10, size=n_samples)

USL = target + tolerance  # Upper Specification Limit
LSL = target - tolerance  # Lower Specification Limit

print(f"\n📊 Production Data: {n_samples} measurements")
print(f"  Target:      {target:.2f} mm")
print(f"  Tolerance:   ±{tolerance:.2f} mm")
print(f"  LSL:         {LSL:.2f} mm")
print(f"  USL:         {USL:.2f} mm")
print(f"\n  Sample mean: {measurements.mean():.4f} mm")
print(f"  Sample std:  {measurements.std():.4f} mm")
print(f"  Range:       [{measurements.min():.4f}, {measurements.max():.4f}]")

# Count defects
defects = ((measurements < LSL) | (measurements > USL)).sum()
print(f"\n  ⚠️  Defects: {defects}/{n_samples} ({defects/n_samples*100:.2f}%)")


# ============================================================================
# Fit Normal Distribution
# ============================================================================

print("\n" + "="*70)
print("📊 Fitting Process Distribution")
print("="*70)

dist = get_distribution('normal')
dist.fit(measurements)

print(dist.summary())

process_mean = dist.mean()
process_std = dist.std()

print(f"\nProcess parameters:")
print(f"  μ (mean): {process_mean:.4f} mm")
print(f"  σ (std):  {process_std:.4f} mm")

# Calculate process centering
shift = abs(process_mean - target)
print(f"\n  Process shift from target: {shift:.4f} mm")
if shift > tolerance / 10:
    print(f"  ⚠️  WARNING: Process is off-center!")


# ============================================================================
# Process Capability Analysis (Cp, Cpk)
# ============================================================================

print("\n" + "="*70)
print("🎯 Process Capability Indices")
print("="*70)
print("""
Process Capability Metrics:

1. Cp (Process Capability):
   - Cp = (USL - LSL) / (6σ)
   - Measures potential capability (if perfectly centered)
   - Cp ≥ 1.33 is typically required (4σ quality)
   - Cp ≥ 2.00 is Six Sigma level

2. Cpk (Process Capability Index):
   - Cpk = min(CPU, CPL)
   - CPU = (USL - μ) / (3σ)
   - CPL = (μ - LSL) / (3σ)
   - Accounts for process centering
   - Cpk < Cp means process is off-center
""")

# Calculate Cp
Cp = (USL - LSL) / (6 * process_std)

# Calculate Cpk
CPU = (USL - process_mean) / (3 * process_std)
CPL = (process_mean - LSL) / (3 * process_std)
Cpk = min(CPU, CPL)

print(f"\nCapability Indices:")
print(f"  Cp  = {Cp:.3f}  (potential capability if centered)")
print(f"  Cpk = {Cpk:.3f}  (actual capability with current centering)")
print(f"  CPU = {CPU:.3f}  (upper capability)")
print(f"  CPL = {CPL:.3f}  (lower capability)")

print(f"\n📈 Interpretation:")
if Cpk < 1.0:
    print(f"  ❌ Cpk < 1.0: Process is NOT capable!")
    print(f"     Significant defects expected. Immediate action required.")
elif Cpk < 1.33:
    print(f"  ⚠️  1.0 ≤ Cpk < 1.33: Marginally capable")
    print(f"     Some defects expected. Process improvement needed.")
elif Cpk < 1.67:
    print(f"  ✅ 1.33 ≤ Cpk < 1.67: Capable (4σ quality)")
    print(f"     Low defect rate. Industry standard.")
elif Cpk < 2.0:
    print(f"  ⭐ 1.67 ≤ Cpk < 2.0: Highly capable (5σ quality)")
    print(f"     Very low defect rate. Excellent process.")
else:
    print(f"  🏆 Cpk ≥ 2.0: World class (Six Sigma)")
    print(f"     Negligible defects. Best-in-class process.")

if Cp - Cpk > 0.2:
    print(f"\n  📊 Gap between Cp and Cpk: {Cp - Cpk:.3f}")
    print(f"     Process is off-center! Re-centering would improve quality.")


# ============================================================================
# Defect Rate Calculation
# ============================================================================

print(f"\n" + "="*70)
print("📉 Defect Rate Analysis")
print("="*70)

# Calculate expected defect rates
defect_rate_lower = dist.cdf(LSL)  # Below LSL
defect_rate_upper = 1 - dist.cdf(USL)  # Above USL
defect_rate_total = defect_rate_lower + defect_rate_upper

parts_per_million = defect_rate_total * 1_000_000

print(f"\nExpected Defect Rates (based on fitted distribution):")
print(f"  Below LSL: {defect_rate_lower*100:.4f}% ({defect_rate_lower*1_000_000:.1f} ppm)")
print(f"  Above USL: {defect_rate_upper*100:.4f}% ({defect_rate_upper*1_000_000:.1f} ppm)")
print(f"  Total:     {defect_rate_total*100:.4f}% ({parts_per_million:.1f} ppm)")

print(f"\nFor production of 100,000 parts:")
print(f"  Expected defects: {int(defect_rate_total * 100_000)}")

# Sigma level
sigma_level = (USL - process_mean) / process_std
print(f"\nProcess Sigma Level:")
print(f"  Distance to USL: {sigma_level:.2f}σ")
if sigma_level >= 6:
    print(f"  → Six Sigma process! (< 3.4 defects per million)")
elif sigma_level >= 5:
    print(f"  → Five Sigma process (< 233 defects per million)")
elif sigma_level >= 4:
    print(f"  → Four Sigma process (< 6,210 defects per million)")
elif sigma_level >= 3:
    print(f"  → Three Sigma process (< 66,807 defects per million)")


# ============================================================================
# Recommendations for Process Improvement
# ============================================================================

print(f"\n" + "="*70)
print("🔧 Process Improvement Recommendations")
print("="*70)

if abs(process_mean - target) > 0.01:
    shift_needed = target - process_mean
    print(f"\n1️⃣  RE-CENTER THE PROCESS:")
    print(f"    Current mean: {process_mean:.4f} mm")
    print(f"    Target:       {target:.4f} mm")
    print(f"    → Adjust by {shift_needed:+.4f} mm")
    
    # Calculate Cpk if centered
    cpk_if_centered = (USL - target) / (3 * process_std)
    improvement = cpk_if_centered - Cpk
    print(f"    → Cpk would improve from {Cpk:.3f} to {cpk_if_centered:.3f} (+{improvement:.3f})")

if process_std > (USL - LSL) / 8:
    print(f"\n2️⃣  REDUCE PROCESS VARIATION:")
    print(f"    Current σ: {process_std:.4f} mm")
    target_std = (USL - LSL) / 8  # For Cp = 1.33
    print(f"    Target σ:  {target_std:.4f} mm (for Cp = 1.33)")
    reduction = (1 - target_std/process_std) * 100
    print(f"    → Need to reduce variation by {reduction:.1f}%")
    print(f"    \nMethods:")
    print(f"      - Improve machine calibration")
    print(f"      - Better operator training")
    print(f"      - Use higher quality raw materials")
    print(f"      - Environmental controls (temperature, humidity)")

if Cpk < 1.33:
    print(f"\n3️⃣  IMMEDIATE ACTIONS REQUIRED:")
    print(f"    - Implement 100% inspection until Cpk ≥ 1.33")
    print(f"    - Root cause analysis (5 Whys, Fishbone diagram)")
    print(f"    - Consider tighter process controls (SPC charts)")


# ============================================================================
# Visualization: SPC Dashboard
# ============================================================================

print("\n" + "="*70)
print("📊 Creating SPC Dashboard...")
print("="*70)

fig = plt.figure(figsize=(14, 10))

# 1. Process Distribution with Spec Limits
ax1 = plt.subplot(2, 2, 1)
ax1.hist(measurements, bins=40, density=True, alpha=0.6, color='skyblue', edgecolor='black')

x = np.linspace(measurements.min(), measurements.max(), 500)
y = dist.pdf(x)
ax1.plot(x, y, 'r-', linewidth=2, label='Fitted distribution')

ax1.axvline(LSL, color='red', linestyle='--', linewidth=2, label='LSL/USL')
ax1.axvline(USL, color='red', linestyle='--', linewidth=2)
ax1.axvline(target, color='green', linestyle=':', linewidth=2, label='Target')
ax1.axvline(process_mean, color='blue', linestyle='-.', linewidth=2, label='Process mean')

# Shade defect regions
ax1.axvspan(measurements.min(), LSL, alpha=0.2, color='red')
ax1.axvspan(USL, measurements.max(), alpha=0.2, color='red')

ax1.set_xlabel('Measurement (mm)', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Process Distribution & Specification Limits', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Control Chart (X-bar chart)
ax2 = plt.subplot(2, 2, 2)

# Simulate subgroups
subgroup_size = 5
subgroup_means = [measurements[i:i+subgroup_size].mean() 
                  for i in range(0, len(measurements), subgroup_size)]

ax2.plot(subgroup_means, 'o-', linewidth=1.5, markersize=4)
ax2.axhline(target, color='green', linestyle=':', linewidth=2, label='Target')
ax2.axhline(process_mean, color='blue', linestyle='--', linewidth=2, label='Process mean')

# Control limits (3σ)
UCL = process_mean + 3 * process_std / np.sqrt(subgroup_size)
LCL = process_mean - 3 * process_std / np.sqrt(subgroup_size)
ax2.axhline(UCL, color='red', linestyle='--', linewidth=1.5, label='Control limits')
ax2.axhline(LCL, color='red', linestyle='--', linewidth=1.5)

ax2.set_xlabel('Subgroup Number', fontsize=11)
ax2.set_ylabel('Subgroup Mean (mm)', fontsize=11)
ax2.set_title('X-bar Control Chart', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Process Capability Visualization
ax3 = plt.subplot(2, 2, 3)

# Show specification window and process spread
spec_width = USL - LSL
process_width = 6 * process_std

ax3.barh(['Specification\nWindow', 'Process\nSpread (6σ)'], 
         [spec_width, process_width], 
         color=['green', 'blue'], alpha=0.6, edgecolor='black')

ax3.set_xlabel('Width (mm)', fontsize=11)
ax3.set_title(f'Process Capability: Cp={Cp:.2f}, Cpk={Cpk:.2f}', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

for i, (label, value) in enumerate([('Spec', spec_width), ('Process', process_width)]):
    ax3.text(value/2, i, f'{value:.3f} mm', va='center', ha='center', 
             fontweight='bold', fontsize=10)

# 4. Summary Table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

summary_text = f"""
PROCESS SUMMARY
{'='*45}

Target:         {target:.2f} mm
Tolerance:      ±{tolerance:.2f} mm
LSL:            {LSL:.2f} mm
USL:            {USL:.2f} mm

{'='*45}

PROCESS PARAMETERS
{'-'*45}
Mean (μ):       {process_mean:.4f} mm
Std Dev (σ):    {process_std:.4f} mm
Shift:          {process_mean - target:+.4f} mm

{'='*45}

CAPABILITY INDICES
{'-'*45}
Cp  =           {Cp:.3f}
Cpk =           {Cpk:.3f}

{'='*45}

DEFECT RATE
{'-'*45}
Expected:       {defect_rate_total*100:.3f}%
                ({parts_per_million:.0f} ppm)

Actual sample:  {defects/n_samples*100:.2f}%
                ({defects}/{n_samples} parts)

{'='*45}
"""

status_color = 'lightgreen' if Cpk >= 1.33 else 'lightyellow' if Cpk >= 1.0 else 'lightcoral'

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.5))

plt.suptitle('Statistical Process Control (SPC) Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()

print("\n✅ Dashboard created!")
plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Process Capability Indices:
   - Cp: Potential capability (if centered)
   - Cpk: Actual capability (accounting for centering)
   - Target: Cpk ≥ 1.33 (4σ), Ideal: Cpk ≥ 2.0 (6σ)

2. Two paths to improvement:
   - Re-center process (adjust mean)
   - Reduce variation (lower σ)

3. SPC Tools:
   - Control charts (X-bar, R charts)
   - Histogram with spec limits
   - Process capability studies

4. Six Sigma Levels:
   - 3σ: 66,807 defects per million
   - 4σ: 6,210 defects per million
   - 5σ: 233 defects per million
   - 6σ: 3.4 defects per million

5. Always check:
   - Is process centered?
   - Is variation acceptable?
   - Are we meeting spec limits?

Next: Check 07_i18n/ for multilingual output examples!
""")
