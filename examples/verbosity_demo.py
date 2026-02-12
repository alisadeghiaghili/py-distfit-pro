#!/usr/bin/env python
"""
Verbosity System Demo
=====================

Demonstrates the self-explanatory verbosity system in distfit-pro.

Features:
- Multiple verbosity levels
- Multilingual support (en/fa/de)
- Context managers for temporary verbosity
- Detailed statistical explanations

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro.core.config import config
from distfit_pro.core.distributions import get_distribution
from distfit_pro.utils.verbose import logger


def demo_verbosity_levels():
    """Demonstrate different verbosity levels."""
    print("\n" + "="*80)
    print("DEMO 1: Verbosity Levels")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=1000)
    
    # =========================================================================
    # SILENT MODE (0) - No output except errors
    # =========================================================================
    print("\n[1] SILENT MODE - No output:")
    config.set_verbosity('silent')
    dist = get_distribution('normal')
    dist.fit(data, method='mle')
    print("(Nothing printed - fitting happened silently)")
    
    # =========================================================================
    # NORMAL MODE (1) - Basic progress messages (default)
    # =========================================================================
    print("\n[2] NORMAL MODE - Basic progress:")
    config.set_verbosity('normal')
    dist = get_distribution('normal')
    dist.fit(data, method='mle')
    
    # =========================================================================
    # VERBOSE MODE (2) - Detailed explanations
    # =========================================================================
    print("\n[3] VERBOSE MODE - Detailed explanations:")
    config.set_verbosity('verbose')
    dist = get_distribution('normal')
    dist.fit(data, method='mle')
    
    # =========================================================================
    # DEBUG MODE (3) - Everything including internals
    # =========================================================================
    print("\n[4] DEBUG MODE - All internal details:")
    config.set_verbosity('debug')
    dist = get_distribution('normal')
    dist.fit(data, method='mle')
    logger.debug("Internal parameters: {params}", params=dist.params)
    logger.debug("Scipy dist object: {scipy}", scipy=dist._scipy_dist)


def demo_multilingual_support():
    """Demonstrate multilingual verbose output."""
    print("\n" + "="*80)
    print("DEMO 2: Multilingual Support")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.weibull(a=2, size=500) * 10
    
    # Set verbose mode
    config.set_verbosity('verbose')
    
    # =========================================================================
    # English (default)
    # =========================================================================
    print("\n[1] English:")
    config.set_language('en')
    dist = get_distribution('weibull')
    dist.fit(data, method='mle')
    
    # =========================================================================
    # Persian (فارسی)
    # =========================================================================
    print("\n[2] Persian (فارسی):")
    config.set_language('fa')
    dist = get_distribution('weibull')
    dist.fit(data, method='mle')
    
    # =========================================================================
    # German (Deutsch)
    # =========================================================================
    print("\n[3] German (Deutsch):")
    config.set_language('de')
    dist = get_distribution('weibull')
    dist.fit(data, method='mle')
    
    # Reset to English
    config.set_language('en')


def demo_context_managers():
    """Demonstrate temporary verbosity with context managers."""
    print("\n" + "="*80)
    print("DEMO 3: Context Managers for Temporary Verbosity")
    print("="*80)
    
    # Set default to normal
    config.set_verbosity('normal')
    
    # Generate sample data
    np.random.seed(42)
    data1 = np.random.gamma(shape=2, scale=3, size=500)
    data2 = np.random.exponential(scale=5, size=500)
    data3 = np.random.lognormal(mean=1, sigma=0.5, size=500)
    
    print("\nDefault mode (NORMAL):")
    dist1 = get_distribution('gamma')
    dist1.fit(data1)
    
    # Temporarily switch to verbose
    print("\nTemporarily switch to VERBOSE:")
    with config.verbose_mode():
        dist2 = get_distribution('exponential')
        dist2.fit(data2)
    
    # Back to normal
    print("\nBack to NORMAL:")
    dist3 = get_distribution('lognormal')
    dist3.fit(data3)
    
    # Temporarily silent
    print("\nTemporarily SILENT:")
    with config.silent_mode():
        dist4 = get_distribution('gamma')
        dist4.fit(data1)
        print("(This line prints, but fitting was silent)")


def demo_fitting_explanations():
    """Demonstrate detailed fitting process explanations."""
    print("\n" + "="*80)
    print("DEMO 4: Self-Explanatory Fitting Process")
    print("="*80)
    
    # Set verbose mode
    config.set_verbosity('verbose')
    
    # Generate sample data with known characteristics
    np.random.seed(42)
    
    # Right-skewed data
    print("\n[1] Fitting Right-Skewed Data (Weibull):")
    data_skewed = np.random.weibull(a=1.5, size=1000) * 50
    dist_weibull = get_distribution('weibull')
    dist_weibull.fit(data_skewed, method='mle')
    
    # Symmetric data
    print("\n[2] Fitting Symmetric Data (Normal):")
    data_normal = np.random.normal(loc=100, scale=15, size=1000)
    dist_normal = get_distribution('normal')
    dist_normal.fit(data_normal, method='mle')
    
    # Heavy-tailed data
    print("\n[3] Fitting Heavy-Tailed Data (Student-t):")
    data_heavy = np.random.standard_t(df=3, size=1000) * 10 + 50
    dist_t = get_distribution('studentt')
    dist_t.fit(data_heavy, method='mle')


def demo_comparison_mle_vs_moments():
    """Compare MLE vs Method of Moments with verbose output."""
    print("\n" + "="*80)
    print("DEMO 5: Comparing Fitting Methods (MLE vs Moments)")
    print("="*80)
    
    # Set verbose mode
    config.set_verbosity('verbose')
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.gamma(shape=3, scale=2, size=800)
    
    print("\n[1] Maximum Likelihood Estimation (MLE):")
    dist_mle = get_distribution('gamma')
    dist_mle.fit(data, method='mle')
    
    print("\n[2] Method of Moments (MoM):")
    dist_mom = get_distribution('gamma')
    dist_mom.fit(data, method='mom')
    
    # Compare results
    print("\n" + "-"*70)
    print("COMPARISON:")
    print("-"*70)
    print(f"\nMLE Parameters:")
    for param, value in dist_mle.params.items():
        print(f"  {param}: {value:.6f}")
    
    print(f"\nMoM Parameters:")
    for param, value in dist_mom.params.items():
        print(f"  {param}: {value:.6f}")
    
    print(f"\nLog-Likelihoods:")
    print(f"  MLE: {dist_mle.log_likelihood():.2f}")
    print(f"  MoM: {dist_mom.log_likelihood():.2f}")
    print(f"  Difference: {dist_mle.log_likelihood() - dist_mom.log_likelihood():.2f}")


def demo_per_call_verbosity_override():
    """Demonstrate overriding verbosity per call."""
    print("\n" + "="*80)
    print("DEMO 6: Per-Call Verbosity Override")
    print("="*80)
    
    # Set global to normal
    config.set_verbosity('normal')
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.lognormal(mean=2, sigma=0.8, size=600)
    
    print("\n[1] Global setting (NORMAL) - basic output:")
    dist1 = get_distribution('lognormal')
    dist1.fit(data)
    
    print("\n[2] Override to VERBOSE for this call only:")
    dist2 = get_distribution('lognormal')
    dist2.fit(data, verbose=True)  # Explicit verbose=True
    
    print("\n[3] Override to SILENT for this call only:")
    dist3 = get_distribution('lognormal')
    dist3.fit(data, verbose=False)  # Explicit verbose=False
    print("(Fitting was silent despite global NORMAL setting)")


def main():
    """
    Run all verbosity demos.
    """
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "DISTFIT-PRO VERBOSITY SYSTEM DEMO" + " "*25 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Run all demos
    demo_verbosity_levels()
    demo_multilingual_support()
    demo_context_managers()
    demo_fitting_explanations()
    demo_comparison_mle_vs_moments()
    demo_per_call_verbosity_override()
    
    # Reset to defaults
    config.reset()
    
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*30 + "DEMO COMPLETE!" + " "*34 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    print("\nKey Takeaways:")
    print("  • SILENT (0): No output (for production)")
    print("  • NORMAL (1): Basic progress (default)")
    print("  • VERBOSE (2): Detailed explanations (for learning)")
    print("  • DEBUG (3): Everything (for troubleshooting)")
    print("\n  • Use config.set_verbosity() for global setting")
    print("  • Use verbose=True/False in fit() to override per call")
    print("  • Use context managers for temporary changes")
    print("  • All messages auto-translate to selected language\n")


if __name__ == '__main__':
    main()
