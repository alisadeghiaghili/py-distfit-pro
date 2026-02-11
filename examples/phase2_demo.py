"""
Phase 2 Features Demo
====================

Demonstrates all Phase 2 features:
- Goodness-of-Fit tests
- Bootstrap confidence intervals
- Advanced diagnostics
- Weighted data fitting
- Mixture models
"""

import numpy as np
from distfit_pro import (
    get_distribution,
    GoodnessOfFitTests,
    Bootstrap,
    Diagnostics,
    WeightedDistributionFitter,
    MixtureModel,
    format_gof_results,
    format_bootstrap_results,
    format_diagnostics
)

# Set seed for reproducibility
np.random.seed(42)

print("="*70)
print("PHASE 2 FEATURES DEMONSTRATION")
print("="*70)

# =========================================================================
# 1. GOODNESS-OF-FIT TESTS
# =========================================================================

print("\n" + "="*70)
print("1. GOODNESS-OF-FIT TESTS")
print("="*70)

# Generate data from normal distribution
data = np.random.normal(10, 2, 1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data, method='mle')

print(f"\nFitted: {dist}")

# Run all GOF tests
gof = GoodnessOfFitTests()
results = gof.run_all_tests(data, dist)

print(format_gof_results(results))

# Individual tests
print("\nIndividual test details:")
print(f"KS test: statistic={results['ks']['statistic']:.6f}, p-value={results['ks']['p_value']:.6f}")
print(f"AD test: statistic={results['ad']['statistic']:.6f}, p-value={results['ad']['p_value']:.6f}")

# =========================================================================
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# =========================================================================

print("\n" + "="*70)
print("2. BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)

# Parametric bootstrap
bs = Bootstrap(n_bootstrap=500, confidence_level=0.95, n_jobs=-1, random_state=42)
ci_parametric = bs.parametric(data, dist, method='percentile')

print("\nParametric Bootstrap:")
print(format_bootstrap_results(ci_parametric))

# Non-parametric bootstrap
ci_nonparametric = bs.nonparametric(data, dist, method='percentile')

print("\nNon-parametric Bootstrap:")
print(format_bootstrap_results(ci_nonparametric))

# =========================================================================
# 3. ADVANCED DIAGNOSTICS
# =========================================================================

print("\n" + "="*70)
print("3. ADVANCED DIAGNOSTICS")
print("="*70)

# Add some outliers
data_with_outliers = np.concatenate([data, [25, 30, -5]])

# Refit
dist_outliers = get_distribution('normal')
dist_outliers.fit(data_with_outliers, method='mle')

# Run diagnostics
diag = Diagnostics()
diag_results = diag.run_full_diagnostics(data_with_outliers, dist_outliers)

print(format_diagnostics(diag_results))

# Individual diagnostic components
print("\nResidual Statistics:")
residuals = diag_results['residuals']
print(f"  Mean: {residuals.mean:.6f}")
print(f"  Std: {residuals.std:.6f}")
print(f"  Skewness: {residuals.skewness:.6f}")

print("\nOutlier Detection (IQR):")
outliers = diag_results['outliers_iqr']
print(f"  Found {outliers.n_outliers} outliers")
if outliers.n_outliers > 0:
    print(f"  Values: {outliers.outlier_values}")

print("\nInfluential Points:")
influence = diag_results['influence']
print(f"  Found {influence.n_influential} influential points")

print("\nTail Behavior:")
tail = diag_results['tail_behavior']
print(f"  Left tail: {tail['left_tail']['verdict']}")
print(f"  Right tail: {tail['right_tail']['verdict']}")
print(f"  Overall: {tail['overall']}")

# =========================================================================
# 4. WEIGHTED DATA FITTING
# =========================================================================

print("\n" + "="*70)
print("4. WEIGHTED DATA FITTING")
print("="*70)

# Generate weighted data
data_weighted = np.random.normal(15, 3, 500)
weights = np.random.uniform(0.5, 1.5, 500)  # Varying importance

print(f"\nData: {len(data_weighted)} observations")
print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

# Standard fit (unweighted)
dist_unweighted = get_distribution('normal')
dist_unweighted.fit(data_weighted, method='mle')

print(f"\nUnweighted fit: {dist_unweighted}")

# Weighted fit
fitter = WeightedDistributionFitter()
params_weighted = fitter.fit(data_weighted, weights, dist_unweighted, method='mle')

print(f"Weighted params: loc={params_weighted['loc']:.4f}, scale={params_weighted['scale']:.4f}")

# Compare
print(f"\nComparison:")
print(f"  Unweighted: loc={dist_unweighted.params['loc']:.4f}, scale={dist_unweighted.params['scale']:.4f}")
print(f"  Weighted:   loc={params_weighted['loc']:.4f}, scale={params_weighted['scale']:.4f}")
print(f"  Difference: Δloc={abs(params_weighted['loc'] - dist_unweighted.params['loc']):.4f}")

# =========================================================================
# 5. MIXTURE MODELS
# =========================================================================

print("\n" + "="*70)
print("5. MIXTURE MODELS")
print("="*70)

# Generate bimodal data
data_mode1 = np.random.normal(5, 1, 400)
data_mode2 = np.random.normal(15, 2, 600)
data_bimodal = np.concatenate([data_mode1, data_mode2])

print(f"\nBimodal data: {len(data_bimodal)} observations")
print(f"True mixture: 40% N(5,1) + 60% N(15,2)")

# Fit mixture model
mixture = MixtureModel(n_components=2, distribution_name='normal', random_state=42)
mixture.fit(data_bimodal, max_iter=50)

print(mixture.summary())

# Compare with true values
print("\nComparison with true values:")
for i, comp in enumerate(mixture.components, 1):
    true_loc = 5 if i == 1 else 15
    true_scale = 1 if i == 1 else 2
    print(f"Component {i}:")
    print(f"  True: loc={true_loc}, scale={true_scale}")
    print(f"  Fitted: loc={comp.distribution.params['loc']:.2f}, scale={comp.distribution.params['scale']:.2f}")

# Generate samples from mixture
samples = mixture.rvs(size=100)
print(f"\nGenerated {len(samples)} samples from fitted mixture")
print(f"Sample statistics: mean={np.mean(samples):.2f}, std={np.std(samples):.2f}")

# =========================================================================
# SUMMARY
# =========================================================================

print("\n" + "="*70)
print("PHASE 2 FEATURES - SUMMARY")
print("="*70)

print("""
✅ Goodness-of-Fit Tests:
   - Kolmogorov-Smirnov (KS)
   - Anderson-Darling (AD)
   - Cramér-von Mises (CVM)
   - Chi-Square (χ²)
   - Likelihood Ratio Test

✅ Bootstrap Confidence Intervals:
   - Parametric bootstrap
   - Non-parametric bootstrap
   - Methods: percentile, basic, BCa
   - Parallel processing support

✅ Advanced Diagnostics:
   - Residual analysis (raw, standardized, quantile)
   - Outlier detection (IQR, Z-score, modified Z-score)
   - Influence analysis (Cook's distance, leverage)
   - Tail behavior assessment

✅ Weighted Data Fitting:
   - Weighted MLE
   - Weighted method of moments
   - All major distributions supported

✅ Mixture Models:
   - EM algorithm
   - Automatic component initialization
   - PDF, CDF, random sampling
   - Component weights and parameters

All features are production-ready with proper error handling!
""")

print("="*70)
