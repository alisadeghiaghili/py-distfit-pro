#!/usr/bin/env python3
"""
Censored Data Fitting (Survival Analysis)
=========================================

Handle incomplete data where some observations are censored.

Common in:
  - Medical survival studies
  - Reliability engineering (time to failure)
  - Customer churn analysis
  - Warranty claims

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
from distfit_pro.core.censored import CensoredFitting
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("⏱️ CENSORED DATA FITTING (SURVIVAL ANALYSIS)")
print("="*70)


# ============================================================================
# What is Censored Data?
# ============================================================================

print("\n" + "="*70)
print("📚 Understanding Censored Data")
print("="*70)
print("""
Censoring occurs when you don't observe the exact event time:

Types:
  • Right-censored: Event hasn't occurred yet (study ended)
  • Left-censored: Event occurred before observation started
  • Interval-censored: Event occurred in unknown interval

Example: Clinical trial measuring time to disease progression
  - Patient A: Progressed at 120 days ✓ (complete observation)
  - Patient B: Left study at 180 days, no progression ✗ (right-censored)
  - Patient C: Progressed at 90 days ✓ (complete)
  
Patient B contributes information: "survived at least 180 days"
""")


# ============================================================================
# Example 1: Product Reliability (Right-Censored)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Product Failure Time Analysis")
print("="*70)
print("""
Scenario: Testing product lifetimes
  - Test duration: 1000 hours
  - Some products failed during test (complete data)
  - Some still working at end (right-censored)
  - Goal: Estimate failure time distribution
""")

# Generate synthetic failure data
true_scale = 800  # Mean time to failure
n_tested = 100
test_duration = 1000

# Generate actual failure times (some beyond test duration)
failure_times = np.random.exponential(scale=true_scale, size=n_tested)

# Separate complete vs censored observations
complete_mask = failure_times <= test_duration
censored_mask = ~complete_mask

observed_times = failure_times.copy()
observed_times[censored_mask] = test_duration  # Right-censored at test end

n_failures = complete_mask.sum()
n_censored = censored_mask.sum()

print(f"\n🔌 Test setup:")
print(f"  Products tested: {n_tested}")
print(f"  Test duration:   {test_duration} hours")
print(f"\n📊 Results:")
print(f"  Failed during test: {n_failures} ({n_failures/n_tested*100:.1f}%)")
print(f"  Survived (censored): {n_censored} ({n_censored/n_tested*100:.1f}%)")


# Fit WITHOUT accounting for censoring (WRONG!)
print("\n1️⃣ Naive approach (ignoring censoring):")
print("   ⚠️  Treating censored observations as actual failures")

dist_naive = get_distribution('expon')
dist_naive.fit(observed_times)  # Wrong: includes censored as if they failed

print(f"  Estimated MTTF (wrong): {dist_naive.mean():.1f} hours")
print(f"  → Biased LOW (treats survivors as failures)")


# Fit WITH censoring (CORRECT)
print("\n2️⃣ Correct approach (accounting for censoring):")
print("   ✅ Using survival analysis methods")

dist_correct = get_distribution('expon')
censor_indicator = complete_mask.astype(int)  # 1=failed, 0=censored

params_correct = CensoredFitting.fit_right_censored(
    observed_times, censor_indicator, dist_correct
)
dist_correct.params = params_correct
dist_correct.fitted = True

print(f"  Estimated MTTF (correct): {dist_correct.mean():.1f} hours")
print(f"  True MTTF: {true_scale} hours")
print(f"  → Much closer to truth!")

print(f"\n🎯 Comparison:")
print(f"  True MTTF:        {true_scale} hours")
print(f"  Naive estimate:   {dist_naive.mean():.1f} hours (error: {abs(dist_naive.mean()-true_scale):.1f})")
print(f"  Correct estimate: {dist_correct.mean():.1f} hours (error: {abs(dist_correct.mean()-true_scale):.1f})")


# ============================================================================
# Example 2: Customer Churn Analysis
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Customer Lifetime Analysis")
print("="*70)
print("""
Scenario: SaaS subscription service
  - Some customers already churned (complete data)
  - Many still active subscribers (right-censored)
  - Want to estimate typical customer lifetime
""")

# Customer data
n_customers = 200
analysis_date = 365  # 1 year into business

# True lifetime distribution (Weibull with shape=1.5)
true_lifetimes = np.random.weibull(a=1.5, size=n_customers) * 300

# Customers who signed up are observed up to analysis_date
churn_mask = true_lifetimes <= analysis_date
active_mask = ~churn_mask

observed_lifetimes = true_lifetimes.copy()
observed_lifetimes[active_mask] = analysis_date  # Still active

n_churned = churn_mask.sum()
n_active = active_mask.sum()

print(f"\n📈 Customer data (at day {analysis_date}):")
print(f"  Total customers: {n_customers}")
print(f"  Churned:  {n_churned} ({n_churned/n_customers*100:.1f}%)")
print(f"  Active:   {n_active} ({n_active/n_customers*100:.1f}%)")

# Naive approach (wrong)
print("\n1️⃣ Naive: Only analyze churned customers")
print("   ⚠️  Ignores active customers completely")

churned_lifetimes = true_lifetimes[churn_mask]
if len(churned_lifetimes) > 0:
    naive_mean = churned_lifetimes.mean()
    print(f"  Estimated lifetime: {naive_mean:.1f} days")
    print(f"  → Severely biased LOW!")

# Correct approach with censoring
print("\n2️⃣ Correct: Include active customers as censored")
print("   ✅ Uses all available information")

dist_weibull = get_distribution('weibull_min')
censor_ind = churn_mask.astype(int)

params_weibull = CensoredFitting.fit_right_censored(
    observed_lifetimes, censor_ind, dist_weibull
)
dist_weibull.params = params_weibull
dist_weibull.fitted = True

print(f"  Estimated median lifetime: {dist_weibull.median():.1f} days")
print(f"  Estimated mean lifetime:   {dist_weibull.mean():.1f} days")

# Business metrics
print(f"\n💼 Business metrics:")
print(f"  30-day retention:  {(1 - dist_weibull.cdf(30))*100:.1f}%")
print(f"  90-day retention:  {(1 - dist_weibull.cdf(90))*100:.1f}%")
print(f"  180-day retention: {(1 - dist_weibull.cdf(180))*100:.1f}%")


# ============================================================================
# Visualization: Kaplan-Meier Survival Curve
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Survival Curve Visualization...")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Product reliability
ax = axes[0]

# Empirical survival curve (Kaplan-Meier style)
times_sorted = np.sort(observed_times)
at_risk = np.arange(n_tested, 0, -1)
events = np.array([complete_mask[np.argsort(observed_times)][i] for i in range(n_tested)])

survival_prob = np.ones(n_tested)
for i in range(1, n_tested):
    if events[i-1]:
        survival_prob[i] = survival_prob[i-1] * (1 - 1/at_risk[i-1])
    else:
        survival_prob[i] = survival_prob[i-1]

ax.step(times_sorted, survival_prob, where='post', linewidth=2, 
        label='Kaplan-Meier (Empirical)', color='blue')

# Fitted survival curve
x_range = np.linspace(0, test_duration, 200)
y_survival = dist_correct.sf(x_range)  # Survival function = 1 - CDF
ax.plot(x_range, y_survival, 'r-', linewidth=2, 
        label='Fitted (with censoring)', alpha=0.8)

# Naive fit (for comparison)
y_naive = dist_naive.sf(x_range)
ax.plot(x_range, y_naive, 'g--', linewidth=2, 
        label='Naive (ignoring censoring)', alpha=0.6)

ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Survival Probability', fontsize=11)
ax.set_title('Product Reliability: Survival Analysis', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 2: Customer churn
ax = axes[1]

# Kaplan-Meier for customers
times_cust = np.sort(observed_lifetimes)
at_risk_cust = np.arange(n_customers, 0, -1)
events_cust = np.array([churn_mask[np.argsort(observed_lifetimes)][i] 
                        for i in range(n_customers)])

survival_cust = np.ones(n_customers)
for i in range(1, n_customers):
    if events_cust[i-1]:
        survival_cust[i] = survival_cust[i-1] * (1 - 1/at_risk_cust[i-1])
    else:
        survival_cust[i] = survival_cust[i-1]

ax.step(times_cust, survival_cust, where='post', linewidth=2, 
        label='Kaplan-Meier', color='blue')

# Fitted
x_cust = np.linspace(0, analysis_date, 200)
y_fitted = dist_weibull.sf(x_cust)
ax.plot(x_cust, y_fitted, 'r-', linewidth=2, label='Fitted Weibull')

ax.set_xlabel('Days since signup', fontsize=11)
ax.set_ylabel('Retention Rate', fontsize=11)
ax.set_title('Customer Retention Analysis', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()

print("\n✅ Visualizations created!")
plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Censored data is VERY common in real applications:
   • Product testing (not all fail during test)
   • Customer analysis (many still active)
   • Medical studies (patients leave study)

2. Ignoring censoring leads to BIASED estimates:
   • Treats "no event yet" as short event time
   • Systematically underestimates true values

3. Proper censored fitting:
   from distfit_pro.core.censored import CensoredFitting
   censor_indicator = [1, 1, 0, 1, 0, ...]  # 1=event, 0=censored
   params = CensoredFitting.fit_right_censored(times, censor, dist)

4. Survival curves (Kaplan-Meier) visualize censored data well

5. Always account for censoring in survival analysis!

Next: See method_comparison.py for MLE vs Method of Moments!
""")
