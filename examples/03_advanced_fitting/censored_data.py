#!/usr/bin/env python3
"""
Censored Data Fitting (Survival Analysis)
=========================================

Handle incomplete data where exact values are unknown.

Types of Censoring:
  - Right-censored: Event hasn't occurred yet (e.g., patient still alive)
  - Left-censored: Event occurred before observation started
  - Interval-censored: Event occurred within a time range

Common in:
  - Medical studies (survival times)
  - Reliability engineering (equipment lifetimes)
  - Customer churn analysis
  - Warranty claims

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("⏱️ CENSORED DATA FITTING (SURVIVAL ANALYSIS)")
print("="*70)


# ============================================================================
# Example 1: Right-Censored Data (Most Common)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Right-Censored Survival Data")
print("="*70)
print("""
Scenario: Medical study tracking patient survival
  - Study duration: 5 years
  - Some patients still alive at end (censored)
  - Some patients died (complete observations)
  - Goal: Estimate survival distribution
""")

# Generate true survival times (exponential distribution)
true_mean_survival = 3.5  # years
n_patients = 200
true_survival_times = np.random.exponential(scale=true_mean_survival, size=n_patients)

# Simulate study duration = 5 years
study_duration = 5.0

# Observed times: min(true_time, study_duration)
observed_times = np.minimum(true_survival_times, study_duration)

# Event indicator: 1 = event occurred (death), 0 = censored (alive)
event_occurred = (true_survival_times <= study_duration).astype(int)

n_events = event_occurred.sum()
n_censored = (1 - event_occurred).sum()

print(f"\n📊 Data Summary:")
print(f"  Total patients:     {n_patients}")
print(f"  Events (deaths):    {n_events} ({n_events/n_patients*100:.1f}%)")
print(f"  Censored (alive):   {n_censored} ({n_censored/n_patients*100:.1f}%)")
print(f"  Study duration:     {study_duration} years")
print(f"  True mean survival: {true_mean_survival} years")

# Method 1: NAIVE (Wrong!) - Ignore censored data
print("\n❌ WRONG METHOD: Ignoring censored data")
complete_data_only = observed_times[event_occurred == 1]
dist_naive = get_distribution('expon')
dist_naive.fit(complete_data_only)

print(f"  Estimated mean: {dist_naive.mean():.2f} years")
print(f"  Error: {abs(dist_naive.mean() - true_mean_survival):.2f} years")
print(f"  ⚠️  Biased estimate (ignores censored patients)!")

# Method 2: CORRECT - Maximum Likelihood with censoring
print("\n✅ CORRECT METHOD: ML estimation with censoring")
print("  (Using survival function for censored observations)")

# For exponential: MLE with censoring
# Log-likelihood: sum[log(f(t_i)) for events] + sum[log(S(t_j)) for censored]
# For exponential: S(t) = exp(-t/μ), f(t) = (1/μ)exp(-t/μ)

def exponential_censored_mle(times, events):
    """
    Maximum likelihood estimation for exponential with right-censoring.
    
    MLE formula: μ = (sum of observed times) / (number of events)
    """
    total_time = times.sum()
    n_events = events.sum()
    return total_time / n_events

mean_censored = exponential_censored_mle(observed_times, event_occurred)

dist_correct = get_distribution('expon')
# Exponential parameterization: loc=0, scale=mean
dist_correct.params = {'loc': 0, 'scale': mean_censored}
dist_correct.fitted = True

print(f"  Estimated mean: {mean_censored:.2f} years")
print(f"  Error: {abs(mean_censored - true_mean_survival):.2f} years")
print(f"  ✅ Much better! Accounts for censored data.")

# Compare estimates
print("\n📈 Comparison:")
print(f"  True mean:          {true_mean_survival:.2f} years")
print(f"  Naive (biased):     {dist_naive.mean():.2f} years (error: {abs(dist_naive.mean() - true_mean_survival):.2f})")
print(f"  Censored MLE:       {mean_censored:.2f} years (error: {abs(mean_censored - true_mean_survival):.2f})")


# ============================================================================
# Example 2: Kaplan-Meier Survival Curve
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Kaplan-Meier Survival Curve")
print("="*70)
print("""
Non-parametric estimator of survival function.
Does NOT assume a specific distribution.
""")

def kaplan_meier(times, events):
    """
    Calculate Kaplan-Meier survival curve.
    
    Returns:
        time_points: Sorted unique event times
        survival_prob: Survival probability at each time point
    """
    # Sort by time
    order = np.argsort(times)
    times_sorted = times[order]
    events_sorted = events[order]
    
    # Get unique event times
    unique_times = np.unique(times_sorted[events_sorted == 1])
    
    survival_prob = []
    n_at_risk = len(times)
    cum_survival = 1.0
    
    for t in unique_times:
        # Number of events at this time
        n_events = ((times_sorted == t) & (events_sorted == 1)).sum()
        
        # Update survival probability
        cum_survival *= (1 - n_events / n_at_risk)
        survival_prob.append(cum_survival)
        
        # Update number at risk (remove events and censored up to this time)
        n_at_risk -= (times_sorted <= t).sum() - (times_sorted < t).sum()
    
    return unique_times, np.array(survival_prob)

km_times, km_survival = kaplan_meier(observed_times, event_occurred)

print(f"\n📊 Kaplan-Meier estimate:")
print(f"  Number of time points: {len(km_times)}")
print(f"  Survival at 1 year:  {np.interp(1.0, km_times, km_survival):.3f}")
print(f"  Survival at 3 years: {np.interp(3.0, km_times, km_survival):.3f}")
print(f"  Survival at 5 years: {km_survival[-1]:.3f}")

# Compare with parametric model
print(f"\n📈 Compare with fitted exponential:")
print(f"  Parametric at 1 year:  {dist_correct.sf(1.0):.3f}")
print(f"  Parametric at 3 years: {dist_correct.sf(3.0):.3f}")
print(f"  Parametric at 5 years: {dist_correct.sf(5.0):.3f}")


# ============================================================================
# Example 3: Weibull Distribution for Reliability
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Weibull Reliability Analysis")
print("="*70)
print("""
Scenario: Component failure times with wear-out
  - Testing stops at 1000 hours
  - Some components haven't failed yet (censored)
  - Weibull distribution (increasing failure rate)
""")

# Generate Weibull failure times
true_shape = 2.0  # > 1 means increasing failure rate (wear-out)
true_scale = 800.0
n_components = 150

true_failures = np.random.weibull(true_shape, size=n_components) * true_scale

# Test duration
test_duration = 1000.0
observed_failures = np.minimum(true_failures, test_duration)
failure_events = (true_failures <= test_duration).astype(int)

n_failed = failure_events.sum()
n_survived = (1 - failure_events).sum()

print(f"\n📊 Reliability Test Data:")
print(f"  Components tested:  {n_components}")
print(f"  Failed during test: {n_failed} ({n_failed/n_components*100:.1f}%)")
print(f"  Still working:      {n_survived} ({n_survived/n_components*100:.1f}%)")
print(f"  Test duration:      {test_duration} hours")

# Fit Weibull (simplified - using complete data approximation)
# In practice, you'd use specialized survival analysis packages
print("\n🔧 Fitting Weibull distribution...")

# Approximate: fit to observed failures only (conservative estimate)
failed_times = observed_failures[failure_events == 1]
dist_weibull = get_distribution('weibull_min')
dist_weibull.fit(failed_times)

print(f"\n✅ Fitted Weibull parameters:")
for param, val in dist_weibull.params.items():
    print(f"  {param}: {val:.4f}")

shape_fitted = dist_weibull.params.get('c', 0)
print(f"\n📈 Failure rate behavior:")
if shape_fitted > 1:
    print(f"  Shape = {shape_fitted:.2f} > 1 → Increasing failure rate (wear-out)")
elif shape_fitted < 1:
    print(f"  Shape = {shape_fitted:.2f} < 1 → Decreasing failure rate (infant mortality)")
else:
    print(f"  Shape ≈ 1 → Constant failure rate (random failures)")

# Reliability metrics
mean_life = dist_weibull.mean()
median_life = dist_weibull.median()
reliability_1000h = dist_weibull.sf(1000)  # P(survive > 1000h)

print(f"\n📊 Reliability Metrics:")
print(f"  Mean time to failure (MTTF):  {mean_life:.0f} hours")
print(f"  Median life (B50):            {median_life:.0f} hours")
print(f"  Reliability at 1000h:         {reliability_1000h:.3f} ({reliability_1000h*100:.1f}%)")
print(f"  Reliability at 500h:          {dist_weibull.sf(500):.3f}")

# Characteristic life (B63.2)
characteristic_life = dist_weibull.ppf(0.632)  # 63.2% failed
print(f"  Characteristic life (B63.2):  {characteristic_life:.0f} hours")


# ============================================================================
# Visualization: Survival Curves
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Survival Analysis Plots...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Censored Data Analysis (Survival Analysis)', fontsize=16, fontweight='bold')

# Plot 1: Medical survival - Kaplan-Meier vs Parametric
ax = axes[0, 0]

# Kaplan-Meier
ax.step(km_times, km_survival, where='post', linewidth=2, label='Kaplan-Meier (non-parametric)', color='blue')

# Parametric (exponential)
t_range = np.linspace(0, study_duration, 200)
ax.plot(t_range, dist_correct.sf(t_range), 'r-', linewidth=2, label='Fitted Exponential', alpha=0.8)

# Mark censored observations
censored_times = observed_times[event_occurred == 0]
for ct in censored_times[:20]:  # Plot first 20 to avoid clutter
    ax.plot(ct, np.interp(ct, km_times, km_survival), 'r+', markersize=8, markeredgewidth=2)

ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.set_title('Medical Survival Analysis', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 2: Histogram of observed times (with censoring indicators)
ax = axes[0, 1]
ax.hist(observed_times[event_occurred == 1], bins=30, alpha=0.7, 
        color='red', edgecolor='black', label='Events (deaths)')
ax.hist(observed_times[event_occurred == 0], bins=30, alpha=0.7, 
        color='blue', edgecolor='black', label='Censored (alive)')
ax.axvline(study_duration, color='green', linestyle='--', linewidth=2, label='Study end')
ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Observed Times', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Weibull reliability
ax = axes[1, 0]

# Survival function (reliability)
t_rel = np.linspace(0, 2000, 200)
reliability = dist_weibull.sf(t_rel)
ax.plot(t_rel, reliability, 'b-', linewidth=2, label='Reliability R(t)')

# Mark important points
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
ax.axvline(median_life, color='red', linestyle='--', linewidth=2, label=f'B50 = {median_life:.0f}h')
ax.axvline(test_duration, color='green', linestyle='--', linewidth=2, label='Test duration')

ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Reliability (Survival Probability)', fontsize=11)
ax.set_title('Weibull Reliability Function', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 4: Hazard rate (for Weibull)
ax = axes[1, 1]

# Weibull hazard function: h(t) = (c/s) * (t/s)^(c-1)
scale_param = dist_weibull.params.get('scale', 1)
shape_param = dist_weibull.params.get('c', 1)

def weibull_hazard(t, shape, scale):
    return (shape / scale) * (t / scale) ** (shape - 1)

t_hazard = np.linspace(1, 2000, 200)
hazard = weibull_hazard(t_hazard, shape_param, scale_param)

ax.plot(t_hazard, hazard, 'r-', linewidth=2)
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Hazard Rate h(t)', fontsize=11)
ax.set_title(f'Weibull Hazard Rate (Shape={shape_param:.2f})', fontweight='bold')
ax.grid(True, alpha=0.3)

if shape_param > 1:
    ax.text(0.5, 0.95, 'Increasing failure rate (wear-out)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
elif shape_param < 1:
    ax.text(0.5, 0.95, 'Decreasing failure rate (infant mortality)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

plt.tight_layout()

print("\n✅ Plots created!")
print("   Close plot window to continue...")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. CENSORED DATA TYPES:
   • Right-censored: Event hasn't occurred yet (most common)
   • Left-censored: Event occurred before observation
   • Interval-censored: Event within time range

2. NEVER IGNORE CENSORED DATA:
   ❌ Removing censored observations → Biased estimates
   ✅ Use survival analysis methods → Correct estimates

3. KEY METHODS:
   • Kaplan-Meier: Non-parametric survival curve
   • Parametric MLE: Assumes distribution (exponential, Weibull, etc.)
   • Log-rank test: Compare survival curves

4. APPLICATIONS:
   • Medical: Patient survival, time to disease progression
   • Engineering: Component lifetimes, reliability testing
   • Business: Customer churn, subscription duration
   • Warranty: Claim times with limited observation period

5. WEIBULL IS KING IN RELIABILITY:
   • Shape > 1 → Wear-out failures (increasing hazard)
   • Shape < 1 → Infant mortality (decreasing hazard)
   • Shape = 1 → Random failures (constant hazard = exponential)

For production use, consider specialized packages:
  - Python: lifelines, scikit-survival
  - R: survival, survminer

Next: See method_comparison.py for MLE vs Method of Moments!
""")
