#!/usr/bin/env python
"""
Package Setup Test
==================

Comprehensive test to verify that all imports work correctly
after fixing __init__.py and setup.py.

Usage:
    python test_package_setup.py
"""

import sys
from typing import Dict, List, Tuple


class PackageTest:
    """Test framework for package imports"""
    
    def __init__(self):
        self.results = []
        self.total = 0
        self.passed = 0
        self.failed = 0
    
    def test_import(self, module_path: str, names: List[str], category: str) -> bool:
        """Test importing specific names from a module"""
        success = True
        errors = []
        
        for name in names:
            self.total += 1
            try:
                if module_path:
                    exec(f"from {module_path} import {name}")
                else:
                    exec(f"import {name}")
                self.passed += 1
                self.results.append((category, name, True, None))
            except Exception as e:
                self.failed += 1
                success = False
                error_msg = f"{type(e).__name__}: {str(e)}"
                errors.append((name, error_msg))
                self.results.append((category, name, False, error_msg))
        
        return success, errors
    
    def print_summary(self):
        """Print beautiful summary"""
        print("\n" + "#" * 70)
        print("#" + " " * 68 + "#")
        print("#" + "  PACKAGE SETUP TEST SUMMARY".center(68) + "#")
        print("#" + " " * 68 + "#")
        print("#" * 70)
        
        # Group by category
        categories = {}
        for cat, name, success, error in self.results:
            if cat not in categories:
                categories[cat] = {'passed': [], 'failed': []}
            if success:
                categories[cat]['passed'].append(name)
            else:
                categories[cat]['failed'].append((name, error))
        
        # Print each category
        for cat in sorted(categories.keys()):
            print(f"\n{'=' * 70}")
            print(f"  📦 {cat}")
            print("=" * 70)
            
            passed = categories[cat]['passed']
            failed = categories[cat]['failed']
            
            if passed:
                print(f"  ✅ Passed ({len(passed)}):")
                for name in passed[:5]:  # Show first 5
                    print(f"     ✓ {name}")
                if len(passed) > 5:
                    print(f"     ... and {len(passed) - 5} more")
            
            if failed:
                print(f"\n  ❌ Failed ({len(failed)}):")
                for name, error in failed:
                    print(f"     ✗ {name}")
                    print(f"       Error: {error}")
        
        # Overall summary
        print("\n" + "=" * 70)
        print("  📊 OVERALL RESULTS")
        print("=" * 70)
        print(f"  Total Tests:  {self.total}")
        print(f"  ✅ Passed:     {self.passed} ({self.passed/self.total*100:.1f}%)")
        print(f"  ❌ Failed:     {self.failed} ({self.failed/self.total*100:.1f}%)")
        print("=" * 70)
        
        if self.failed == 0:
            print("\n  🎉 ALL TESTS PASSED! Package setup is correct!")
            print("  ✅ Ready to install with: pip install -e .\n")
            return 0
        else:
            print(f"\n  ⚠️  {self.failed} test(s) failed")
            print("  Please fix the issues above.\n")
            return 1


def main():
    """Run all package tests"""
    tester = PackageTest()
    
    print("\n" + "=" * 70)
    print("  🧪 TESTING PACKAGE SETUP")
    print("=" * 70)
    print("  Checking all imports from __init__.py...\n")
    
    # Test 1: Core Distributions
    print("[1/9] Testing Core Distributions...")
    distributions = [
        'NormalDistribution',
        'ExponentialDistribution',
        'UniformDistribution',
        'GammaDistribution',
        'BetaDistribution',
        'WeibullDistribution',
        'LognormalDistribution',
        'PoissonDistribution',
        'BinomialDistribution',
        'get_distribution',
        'list_distributions',
    ]
    tester.test_import('distfit_pro', distributions, 'Core Distributions')
    
    # Test 2: Base Classes
    print("[2/9] Testing Base Classes...")
    base_classes = [
        'BaseDistribution',
        'ContinuousDistribution',
        'DiscreteDistribution',
        'DistributionInfo',
        'FittingMethod',
    ]
    tester.test_import('distfit_pro', base_classes, 'Base Classes')
    
    # Test 3: GOF Tests
    print("[3/9] Testing Goodness-of-Fit Tests...")
    gof_classes = [
        'GOFTest',
        'GOFResult',
        'KolmogorovSmirnovTest',
        'AndersonDarlingTest',
        'ChiSquareTest',
        'CramerVonMisesTest',
    ]
    tester.test_import('distfit_pro', gof_classes, 'GOF Tests')
    
    # Test 4: Bootstrap
    print("[4/9] Testing Bootstrap Methods...")
    bootstrap_items = [
        'bootstrap_ci',
        'parametric_bootstrap',
        'nonparametric_bootstrap',
        'BootstrapResult',
        'bootstrap_hypothesis_test',
        'bootstrap_goodness_of_fit',
    ]
    tester.test_import('distfit_pro', bootstrap_items, 'Bootstrap Methods')
    
    # Test 5: High-Level API
    print("[5/9] Testing High-Level API...")
    api_items = [
        'DistributionFitter',
        'FitResults',
        'FitResult',
        'fit',
        'find_best_distribution',
        'compare_distributions',
        'ComparisonResult',
    ]
    tester.test_import('distfit_pro', api_items, 'High-Level API')
    
    # Test 6: Visualization
    print("[6/9] Testing Visualization...")
    viz_items = ['DistributionPlotter']
    tester.test_import('distfit_pro', viz_items, 'Visualization')
    
    # Test 7: Config
    print("[7/9] Testing Configuration...")
    config_items = [
        'config',
        'set_language',
        'set_verbosity',
        'get_language',
        'get_verbosity',
    ]
    tester.test_import('distfit_pro', config_items, 'Configuration')
    
    # Test 8: Advanced Features (may not be available)
    print("[8/9] Testing Advanced Features...")
    advanced_items = [
        'GOFTests',
        'Bootstrap',
        'Diagnostics',
        'WeightedFitting',
        'ModelSelection',
    ]
    success, errors = tester.test_import('distfit_pro', advanced_items, 'Advanced Features')
    if not success:
        print("  ⚠️  Note: Some advanced features may not be exported")
    
    # Test 9: Direct module imports
    print("[9/9] Testing Direct Module Imports...")
    try:
        from distfit_pro.core import distributions
        from distfit_pro.core import base
        from distfit_pro.core import gof_tests
        from distfit_pro.core import bootstrap as core_bootstrap
        from distfit_pro.core import diagnostics
        from distfit_pro.core import weighted
        from distfit_pro.core import model_selection
        
        print("  ✅ All core modules importable")
        tester.passed += 7
        tester.total += 7
        for mod in ['distributions', 'base', 'gof_tests', 'bootstrap', 'diagnostics', 'weighted', 'model_selection']:
            tester.results.append(('Direct Imports', f'core.{mod}', True, None))
    except Exception as e:
        print(f"  ❌ Core module import failed: {e}")
        tester.failed += 1
        tester.total += 1
        tester.results.append(('Direct Imports', 'core modules', False, str(e)))
    
    # Print summary
    exit_code = tester.print_summary()
    
    # Additional info
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("  🚀 NEXT STEPS")
        print("=" * 70)
        print("  1. Install package:")
        print("     cd D:\\py-distfit-pro\\py-distfit-pro-main")
        print("     pip install -e .")
        print("")
        print("  2. Run full tests:")
        print("     python test_all_distributions.py")
        print("     python test_advanced_features.py")
        print("")
        print("  3. Try a quick example:")
        print("     python -c \"from distfit_pro import fit; import numpy as np; ")
        print("     data = np.random.normal(5, 2, 100); result = fit(data, 'normal'); ")
        print("     print(result.summary())\"")
        print("=" * 70 + "\n")
    
    return exit_code


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
