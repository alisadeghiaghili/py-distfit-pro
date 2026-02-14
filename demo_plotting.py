#!/usr/bin/env python
"""
Plotting Demo - Visual Verification
====================================

Generates sample plots in all 3 languages for visual inspection.

Usage:
    python demo_plotting.py
    
Output:
    plots/comparison_en.png
    plots/comparison_fa.png
    plots/comparison_de.png
    plots/diagnostic_en.png
    plots/diagnostic_fa.png
    plots/diagnostic_de.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution
from distfit_pro.core.config import set_language
from distfit_pro.visualization.plots import DistributionPlotter

# Configure matplotlib for better output
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

def main():
    print("\n" + "="*70)
    print("  PLOTTING DEMO - Generating Visual Examples")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}/\n")
    
    # Generate sample data (slightly skewed for interesting plots)
    print("📊 Generating sample data...")
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=10, scale=2, size=400),
        np.random.normal(loc=15, scale=1, size=100)  # Add some skewness
    ])
    print(f"   Sample size: {len(data)}")
    print(f"   Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    print(f"   Min: {data.min():.2f}, Max: {data.max():.2f}\n")
    
    # Fit multiple distributions
    print("🔧 Fitting distributions...")
    distributions = ['normal', 'lognormal', 'gamma']
    fitted_models = []
    
    for dist_name in distributions:
        dist = get_distribution(dist_name)
        dist.fit(data, verbose=False)
        fitted_models.append(dist)
        print(f"   ✓ {dist.info.display_name} fitted")
    
    best_model = fitted_models[0]  # Normal as best
    print(f"\n   Best model: {best_model.info.display_name}\n")
    
    # Language configurations
    languages = {
        'en': {'name': 'English', 'flag': '🇬🇧'},
        'fa': {'name': 'Persian (فارسی)', 'flag': '🇮🇷'},
        'de': {'name': 'German (Deutsch)', 'flag': '🇩🇪'}
    }
    
    # Generate plots for each language
    for lang_code, lang_info in languages.items():
        print(f"{lang_info['flag']} Generating {lang_info['name']} plots...")
        
        # Set language
        set_language(lang_code)
        
        # Create plotter
        plotter = DistributionPlotter(data, fitted_models, best_model=best_model)
        
        # 1. Comparison Plot
        try:
            fig_comp = plotter.plot_comparison(figsize=(14, 10), show_top_n=3)
            comp_filename = f"{output_dir}/comparison_{lang_code}.png"
            fig_comp.savefig(comp_filename, dpi=150, bbox_inches='tight')
            plt.close(fig_comp)
            print(f"   ✓ Saved: {comp_filename}")
        except Exception as e:
            print(f"   ✗ Comparison plot failed: {e}")
        
        # 2. Diagnostic Plot
        try:
            fig_diag = plotter.plot_diagnostics(figsize=(14, 10))
            diag_filename = f"{output_dir}/diagnostic_{lang_code}.png"
            fig_diag.savefig(diag_filename, dpi=150, bbox_inches='tight')
            plt.close(fig_diag)
            print(f"   ✓ Saved: {diag_filename}")
        except Exception as e:
            print(f"   ✗ Diagnostic plot failed: {e}")
        
        print()
    
    # Summary
    print("="*70)
    print("  ✅ DEMO COMPLETED!")
    print("="*70)
    print(f"\n📁 All plots saved in: {output_dir}/\n")
    print("📊 Generated plots:")
    print("   • comparison_en.png  - English comparison plots")
    print("   • comparison_fa.png  - Persian comparison plots (فارسی)")
    print("   • comparison_de.png  - German comparison plots (Deutsch)")
    print("   • diagnostic_en.png  - English diagnostic plots")
    print("   • diagnostic_fa.png  - Persian diagnostic plots (فارسی)")
    print("   • diagnostic_de.png  - German diagnostic plots (Deutsch)")
    print("\n🎨 Open the files to see the multilingual plots!\n")
    
    # Translation samples
    print("="*70)
    print("  📝 TRANSLATION SAMPLES")
    print("="*70 + "\n")
    
    from distfit_pro.locales import t
    
    sample_keys = [
        ('pdf_comparison', 'PDF Comparison title'),
        ('qq_plot', 'Q-Q Plot title'),
        ('residuals', 'Residuals label'),
        ('tail_behavior', 'Tail behavior title')
    ]
    
    for lang_code, lang_info in languages.items():
        set_language(lang_code)
        print(f"{lang_info['flag']} {lang_info['name']}:")
        for key, description in sample_keys:
            print(f"   {key:20} → {t(key)}")
        print()
    
    print("="*70)
    print("  🎉 ALL DONE! Enjoy the beautiful multilingual plots!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
