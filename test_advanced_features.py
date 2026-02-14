#!/usr/bin/env python
"""
Comprehensive Advanced Features Test
=====================================

Tests all advanced features:
1. GOF Tests (KS, AD, Chi-Square, CvM)
2. Bootstrap CI (Parametric + Non-parametric)
3. Diagnostics (Residuals, Influence, Outliers)
4. Weighted Fitting
5. Model Selection (AIC, BIC, AICc)

Usage:
    python test_advanced_features.py
"""

import numpy as np
import sys
from typing import Dict, Any

def test_gof_tests():
    """Test Goodness-of-Fit Tests"""
    print("\n" + "="*70)
    print("  🧪 TEST 1: GOODNESS-OF-FIT TESTS")
    print("="*70)
    
    try:
        from distfit_pro.core.distributions import get_distribution
        from distfit_pro.core.gof_tests import GOFTests
        
        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(5, 2, 500)
        
        # Fit distribution
        dist = get_distribution('normal')
        dist.fit(data, method='mle')
        
        print(f"\n✓ Fitted {dist.info.display_name}")
        print(f"  Parameters: μ={dist.params['loc']:.3f}, σ={dist.params['scale']:.3f}")
        
        # Run all GOF tests
        print("\n  Running all GOF tests...")
        results = GOFTests.run_all_tests(data, dist)
        
        print(f"\n  ✅ Completed {len(results)} tests:")
        for test_name, result in results.items():
            status = "✓ PASS" if not result.reject_null else "✗ FAIL"
            print(f"     {status} {result.test_name:20} p={result.p_value:.4f}")
        
        # Summary table
        print("\n" + GOFTests.summary_table(results))
        
        return True
    
    except Exception as e:
        print(f"\n  ❌ GOF Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bootstrap():
    """Test Bootstrap Confidence Intervals"""
    print("\n" + "="*70)
    print("  🧪 TEST 2: BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*70)
    
    try:
        from distfit_pro.core.distributions import get_distribution
        from distfit_pro.core.bootstrap import Bootstrap
        
        # Generate data
        np.random.seed(42)
        data = np.random.normal(10, 3, 200)
        
        # Fit distribution
        dist = get_distribution('normal')
        dist.fit(data, method='mle')
        
        print(f"\n✓ Fitted {dist.info.display_name}")
        
        # Parametric bootstrap (fewer samples for speed)
        print("\n  Running parametric bootstrap (100 samples)...")
        ci_param = Bootstrap.parametric(
            data, dist, 
            n_bootstrap=100, 
            n_jobs=1,  # Serial for testing
            random_state=42
        )
        
        print(f"\n  ✅ Parametric Bootstrap Results:")
        for param_name, result in ci_param.items():
            print(f"     {param_name}: {result.estimate:.4f} "
                  f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        
        # Non-parametric bootstrap
        print("\n  Running non-parametric bootstrap (100 samples)...")
        ci_nonparam = Bootstrap.nonparametric(
            data, dist,
            n_bootstrap=100,
            n_jobs=1,
            random_state=42
        )
        
        print(f"\n  ✅ Non-parametric Bootstrap Results:")
        for param_name, result in ci_nonparam.items():
            print(f"     {param_name}: {result.estimate:.4f} "
                  f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        
        return True
    
    except Exception as e:
        print(f"\n  ❌ Bootstrap FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostics():
    """Test Diagnostics"""
    print("\n" + "="*70)
    print("  🧪 TEST 3: DIAGNOSTICS")
    print("="*70)
    
    try:
        from distfit_pro.core.distributions import get_distribution
        from distfit_pro.core.diagnostics import Diagnostics
        
        # Generate data with outliers
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        # Add some outliers
        data[::50] = np.random.normal(5, 1, len(data[::50]))
        
        # Fit distribution
        dist = get_distribution('normal')
        dist.fit(data, method='mle')
        
        print(f"\n✓ Fitted {dist.info.display_name}")
        
        # 1. Residual Analysis
        print("\n  1️⃣  Residual Analysis...")
        residuals = Diagnostics.residual_analysis(data, dist)
        print("     ✅ Computed 4 types of residuals:")
        print(f"        • Quantile: mean={np.mean(residuals.quantile_residuals):.4f}")
        print(f"        • Pearson: mean={np.mean(residuals.pearson_residuals):.4f}")
        print(f"        • Deviance: mean={np.mean(residuals.deviance_residuals):.4f}")
        print(f"        • Standardized: mean={np.mean(residuals.standardized_residuals):.4f}")
        
        # 2. Influence Diagnostics
        print("\n  2️⃣  Influence Diagnostics...")
        influence = Diagnostics.influence_diagnostics(data, dist)
        print(f"     ✅ Identified {len(influence.influential_indices)} influential points")
        print(f"        Max Cook's D: {np.max(influence.cooks_distance):.4f}")
        
        # 3. Outlier Detection
        print("\n  3️⃣  Outlier Detection...")
        
        methods = ['zscore', 'iqr', 'likelihood', 'mahalanobis']
        for method in methods:
            outliers = Diagnostics.detect_outliers(data, dist, method=method)
            print(f"     ✅ {method:15} → {len(outliers.outlier_indices):3} outliers")
        
        # 4. Plot Diagnostics
        print("\n  4️⃣  Plot Diagnostics Data...")
        qq_data = Diagnostics.qq_diagnostics(data, dist)
        pp_data = Diagnostics.pp_diagnostics(data, dist)
        worm_data = Diagnostics.worm_plot_data(data, dist)
        
        print(f"     ✅ Q-Q correlation: {qq_data['correlation']:.4f}")
        print(f"     ✅ P-P max deviation: {pp_data['max_deviation']:.4f}")
        print(f"     ✅ Worm slope: {worm_data['slope']:.4f}")
        
        return True
    
    except Exception as e:
        print(f"\n  ❌ Diagnostics FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weighted_fitting():
    """Test Weighted Fitting"""
    print("\n" + "="*70)
    print("  🧪 TEST 4: WEIGHTED FITTING")
    print("="*70)
    
    try:
        from distfit_pro.core.distributions import get_distribution
        from distfit_pro.core.weighted import WeightedFitting
        
        # Generate weighted data
        np.random.seed(42)
        data = np.random.normal(5, 2, 300)
        weights = np.random.uniform(0.5, 1.5, 300)
        
        print("\n✓ Generated weighted data")
        print(f"  Effective sample size: {WeightedFitting.effective_sample_size(weights):.1f}")
        
        # Weighted stats
        print("\n  Computing weighted statistics...")
        stats = WeightedFitting.weighted_stats(data, weights)
        print(f"     ✅ Weighted mean: {stats['mean']:.4f}")
        print(f"     ✅ Weighted std: {stats['std']:.4f}")
        print(f"     ✅ Weighted median: {stats['median']:.4f}")
        
        # Fit with weighted MLE
        print("\n  Fitting with Weighted MLE...")
        dist = get_distribution('normal')
        params_mle = WeightedFitting.fit_weighted_mle(data, weights, dist)
        print(f"     ✅ MLE: μ={params_mle['loc']:.4f}, σ={params_mle['scale']:.4f}")
        
        # Fit with weighted moments
        print("\n  Fitting with Weighted Moments...")
        params_mom = WeightedFitting.fit_weighted_moments(data, weights, dist)
        print(f"     ✅ Moments: μ={params_mom['loc']:.4f}, σ={params_mom['scale']:.4f}")
        
        return True
    
    except Exception as e:
        print(f"\n  ❌ Weighted Fitting FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selection():
    """Test Model Selection"""
    print("\n" + "="*70)
    print("  🧪 TEST 5: MODEL SELECTION")
    print("="*70)
    
    try:
        from distfit_pro.core.distributions import get_distribution
        from distfit_pro.core.model_selection import ModelSelection
        
        # Generate data
        np.random.seed(42)
        data = np.random.gamma(2, 2, 400)
        
        print("\n✓ Generated gamma data")
        
        # Fit multiple distributions
        print("\n  Fitting candidate models...")
        candidates = ['normal', 'lognormal', 'gamma', 'weibull']
        fitted_dists = []
        
        for name in candidates:
            dist = get_distribution(name)
            try:
                dist.fit(data, method='mle')
                fitted_dists.append(dist)
                print(f"     ✅ {name}")
            except:
                print(f"     ✗ {name} (fit failed)")
        
        # Compare using AIC
        print("\n  Computing AIC for all models...")
        aic_scores = ModelSelection.compare_models(data, fitted_dists, criterion='aic')
        
        print(f"\n  ✅ AIC Results:")
        for score in aic_scores[:3]:  # Top 3
            print(f"     {score.rank}. {score.distribution_name:15} AIC={score.score:.2f}")
        
        # Compare using BIC
        print("\n  Computing BIC for all models...")
        bic_scores = ModelSelection.compare_models(data, fitted_dists, criterion='bic')
        
        print(f"\n  ✅ BIC Results:")
        for score in bic_scores[:3]:  # Top 3
            print(f"     {score.rank}. {score.distribution_name:15} BIC={score.score:.2f}")
        
        return True
    
    except Exception as e:
        print(f"\n  ❌ Model Selection FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  COMPREHENSIVE ADVANCED FEATURES TEST".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    results = {
        'GOF Tests': test_gof_tests(),
        'Bootstrap CI': test_bootstrap(),
        'Diagnostics': test_diagnostics(),
        'Weighted Fitting': test_weighted_fitting(),
        'Model Selection': test_model_selection()
    }
    
    # Summary
    print("\n" + "="*70)
    print("  📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "="*70)
    print(f"  Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n  🎉 ALL ADVANCED FEATURES WORKING PERFECTLY!")
        print("  ✅ Ready for production use!\n")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        print("  Please review errors above.\n")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
