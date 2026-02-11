"""
Internationalization (i18n) System
===================================

Provides translations for package outputs in multiple languages.

Supported Languages:
- English (en)
- Persian/Farsi (fa)
- German (de)
"""

from typing import Dict
from ..core.config import get_language


# ═══════════════════════════════════════════════════════════════
# ENGLISH TRANSLATIONS
# ═══════════════════════════════════════════════════════════════

EN_TRANSLATIONS = {
    # Distribution info headers
    'estimated_parameters': 'ESTIMATED PARAMETERS',
    'location_statistics': 'LOCATION STATISTICS',
    'spread_statistics': 'SPREAD STATISTICS',
    'shape_statistics': 'SHAPE STATISTICS',
    'key_quantiles': 'KEY QUANTILES',
    'practical_applications': 'PRACTICAL APPLICATIONS',
    'characteristics': 'CHARACTERISTICS',
    'warning': 'Warning',
    
    # Statistics
    'mean': 'Mean (μ)',
    'median': 'Median',
    'mode': 'Mode',
    'variance': 'Variance (σ²)',
    'std_deviation': 'Std Deviation (σ)',
    'skewness': 'Skewness (asymmetry)',
    'kurtosis': 'Kurtosis (tail weight)',
    'percentile': '{p}th percentile',
    
    # Skewness interpretations
    'right_skewed': '→ Right-skewed (positive)',
    'left_skewed': '→ Left-skewed (negative)',
    'symmetric': '→ Approximately symmetric',
    
    # Kurtosis interpretations
    'heavy_tails': '→ Heavy tails (leptokurtic)',
    'light_tails': '→ Light tails (platykurtic)',
    'normal_tails': '→ Normal-like tails',
    
    # Messages
    'not_fitted': 'has not been fitted yet.',
    'for_explanation': 'For conceptual explanation, use .explain()',
    'for_statistics': 'For complete statistics: dist.summary()',
    'undefined': 'Undefined',
    'na': 'N/A',
    
    # Fitter and Results
    'fit_results': 'Distribution Fitting Results',
    'data_summary': 'Data Summary',
    'sample_size': 'Sample size',
    'ci_95': '95% CI',
    'std_dev': 'Std. deviation',
    'outliers': 'Outliers',
    'model_ranking': 'Model Ranking',
    'rank': 'Rank',
    'distribution': 'Distribution',
    'delta': 'Δ',
    'status': 'Status',
    'best_model': 'Best Model',
    'diagnostic_notes': 'Diagnostic Notes',
    'recommendations': 'Recommendations',
    'suggested_distributions': 'Suggested Distributions',
    'fitting': 'Fitting',
    'distributions': 'distributions',
    'estimation_method': 'Estimation method',
    'selection_criterion': 'Selection criterion',
    'num_cores': 'Number of cores',
    'all': 'all',
    'comparing_models': 'Comparing models',
    'fit_complete': 'Fitting complete',
    
    # Plotting
    'pdf_comparison': 'PDF Comparison',
    'cdf_comparison': 'CDF Comparison',
    'qq_plot': 'Q-Q Plot',
    'pp_plot': 'P-P Plot',
    'residual_plot': 'Residual Plot',
    'residual_distribution': 'Residual Distribution',
    'tail_behavior': 'Tail Behavior (Survival Function)',
    'influence_plot': 'Influence Plot',
    'diagnostic_plots': 'Diagnostic Plots',
    'comparison_plots': 'Distribution Fitting - Comparison Plots',
    'value': 'Value',
    'density': 'Density',
    'cumulative_probability': 'Cumulative Probability',
    'theoretical_quantiles': 'Theoretical Quantiles',
    'empirical_quantiles': 'Empirical Quantiles',
    'theoretical_probabilities': 'Theoretical Probabilities',
    'empirical_probabilities': 'Empirical Probabilities',
    'perfect_fit': 'Perfect fit',
    'data': 'Data',
    'fitted': 'Fitted',
    'empirical': 'Empirical',
    'empirical_cdf': 'Empirical CDF',
    'residuals': 'Residuals',
    'standardized_residuals': 'Standardized Residuals',
    'zero_line': 'Zero line',
    'influence': 'Influence',
    'high_influence': 'High influence (>{threshold:.3f})',
    'data_points': 'Data points',
    'interactive_dashboard': 'Interactive Distribution Fitting Dashboard - Best: {model}',
}


# ═══════════════════════════════════════════════════════════════
# PERSIAN/FARSI TRANSLATIONS
# ═══════════════════════════════════════════════════════════════

FA_TRANSLATIONS = {
    # Distribution info headers
    'estimated_parameters': 'پارامترهای برآورد شده',
    'location_statistics': 'آمارهای مکانی',
    'spread_statistics': 'آمارهای پراکندگی',
    'shape_statistics': 'آمارهای شکل',
    'key_quantiles': 'کوانتایل‌های کلیدی',
    'practical_applications': 'کاربردهای عملی',
    'characteristics': 'ویژگی‌های این توزیع',
    'warning': 'هشدار',
    
    # Statistics
    'mean': 'میانگین (μ)',
    'median': 'میانه',
    'mode': 'مد (نما)',
    'variance': 'واریانس (σ²)',
    'std_deviation': 'انحراف معیار (σ)',
    'skewness': 'چولگی (عدم تقارن)',
    'kurtosis': 'کشیدگی (سنگینی دنباله)',
    'percentile': 'صدک {p}',
    
    # Skewness interpretations
    'right_skewed': '→ راست‌چوله (چولگی مثبت)',
    'left_skewed': '→ چپ‌چوله (چولگی منفی)',
    'symmetric': '→ تقریباً متقارن',
    
    # Kurtosis interpretations
    'heavy_tails': '→ دنباله‌های سنگین',
    'light_tails': '→ دنباله‌های سبک',
    'normal_tails': '→ دنباله‌های شبیه نرمال',
    
    # Messages
    'not_fitted': 'هنوز فیت نشده است.',
    'for_explanation': 'برای توضیح مفهومی، از explain. استفاده کنید',
    'for_statistics': 'برای آمارهای کامل: dist.summary()',
    'undefined': 'نامعلوم',
    'na': 'ندارد',
    
    # Fitter and Results
    'fit_results': 'نتایج فیت توزیع‌های آماری',
    'data_summary': 'خلاصه داده',
    'sample_size': 'تعداد نمونه',
    'ci_95': 'فاصله اطمینان ۹۵٪',
    'std_dev': 'انحراف معیار',
    'outliers': 'نقاط پرت',
    'model_ranking': 'رتبه‌بندی مدل‌ها',
    'rank': 'رتبه',
    'distribution': 'توزیع',
    'delta': 'Δ',
    'status': 'وضعیت',
    'best_model': 'مدل برتر',
    'diagnostic_notes': 'یادداشت‌های تشخیصی',
    'recommendations': 'پیشنهادات',
    'suggested_distributions': 'توزیع‌های پیشنهادی',
    'fitting': 'در حال فیت',
    'distributions': 'توزیع',
    'estimation_method': 'روش تخمین',
    'selection_criterion': 'معیار انتخاب',
    'num_cores': 'تعداد کور',
    'all': 'همه',
    'comparing_models': 'در حال مقایسه مدل‌ها',
    'fit_complete': 'فیت کامل شد',
    
    # Plotting
    'pdf_comparison': 'مقایسه PDF',
    'cdf_comparison': 'مقایسه CDF',
    'qq_plot': 'نمودار Q-Q',
    'pp_plot': 'نمودار P-P',
    'residual_plot': 'نمودار باقیمانده',
    'residual_distribution': 'توزیع باقیمانده',
    'tail_behavior': 'رفتار دنباله (تابع بقا)',
    'influence_plot': 'نمودار نفوذ',
    'diagnostic_plots': 'نمودارهای تشخیصی',
    'comparison_plots': 'فیت توزیع - نمودارهای مقایسه‌ای',
    'value': 'مقدار',
    'density': 'چگالی',
    'cumulative_probability': 'احتمال تجمعی',
    'theoretical_quantiles': 'کوانتایل‌های نظری',
    'empirical_quantiles': 'کوانتایل‌های تجربی',
    'theoretical_probabilities': 'احتمالات نظری',
    'empirical_probabilities': 'احتمالات تجربی',
    'perfect_fit': 'فیت کامل',
    'data': 'داده',
    'fitted': 'فیت شده',
    'empirical': 'تجربی',
    'empirical_cdf': 'CDF تجربی',
    'residuals': 'باقیمانده‌ها',
    'standardized_residuals': 'باقیمانده‌های استاندارد',
    'zero_line': 'خط صفر',
    'influence': 'نفوذ',
    'high_influence': 'نفوذ بالا (>{threshold:.3f})',
    'data_points': 'نقاط داده',
    'interactive_dashboard': 'داشبورد تعاملی فیت توزیع - بهترین: {model}',
}


# ═══════════════════════════════════════════════════════════════
# GERMAN TRANSLATIONS
# ═══════════════════════════════════════════════════════════════

DE_TRANSLATIONS = {
    # Distribution info headers
    'estimated_parameters': 'GESCHÄTZTE PARAMETER',
    'location_statistics': 'LAGESTATISTIKEN',
    'spread_statistics': 'STREUUNGSSTATISTIKEN',
    'shape_statistics': 'FORMSTATISTIKEN',
    'key_quantiles': 'WICHTIGE QUANTILE',
    'practical_applications': 'PRAKTISCHE ANWENDUNGEN',
    'characteristics': 'EIGENSCHAFTEN',
    'warning': 'Warnung',
    
    # Statistics
    'mean': 'Mittelwert (μ)',
    'median': 'Median',
    'mode': 'Modus',
    'variance': 'Varianz (σ²)',
    'std_deviation': 'Standardabweichung (σ)',
    'skewness': 'Schiefe (Asymmetrie)',
    'kurtosis': 'Kurtosis (Schwere der Ausläufer)',
    'percentile': '{p}. Perzentil',
    
    # Skewness interpretations
    'right_skewed': '→ Rechtschief (positiv)',
    'left_skewed': '→ Linkschief (negativ)',
    'symmetric': '→ Annähernd symmetrisch',
    
    # Kurtosis interpretations
    'heavy_tails': '→ Schwere Ausläufer (leptokurtisch)',
    'light_tails': '→ Leichte Ausläufer (platykurtisch)',
    'normal_tails': '→ Normalverteilungsähnliche Ausläufer',
    
    # Messages
    'not_fitted': 'wurde noch nicht angepasst.',
    'for_explanation': 'Für konzeptionelle Erklärung verwenden Sie .explain()',
    'for_statistics': 'Für vollständige Statistiken: dist.summary()',
    'undefined': 'Undefiniert',
    'na': 'N/V',
    
    # Fitter and Results
    'fit_results': 'Ergebnisse der Verteilungsanpassung',
    'data_summary': 'Datenzusammenfassung',
    'sample_size': 'Stichprobengröße',
    'ci_95': '95% KI',
    'std_dev': 'Standardabweichung',
    'outliers': 'Ausreißer',
    'model_ranking': 'Modellrangliste',
    'rank': 'Rang',
    'distribution': 'Verteilung',
    'delta': 'Δ',
    'status': 'Status',
    'best_model': 'Bestes Modell',
    'diagnostic_notes': 'Diagnosehinweise',
    'recommendations': 'Empfehlungen',
    'suggested_distributions': 'Vorgeschlagene Verteilungen',
    'fitting': 'Anpassung',
    'distributions': 'Verteilungen',
    'estimation_method': 'Schätzmethode',
    'selection_criterion': 'Auswahlkriterium',
    'num_cores': 'Anzahl Kerne',
    'all': 'alle',
    'comparing_models': 'Modellvergleich',
    'fit_complete': 'Anpassung abgeschlossen',
    
    # Plotting
    'pdf_comparison': 'PDF-Vergleich',
    'cdf_comparison': 'CDF-Vergleich',
    'qq_plot': 'Q-Q-Diagramm',
    'pp_plot': 'P-P-Diagramm',
    'residual_plot': 'Residuendiagramm',
    'residual_distribution': 'Residuenverteilung',
    'tail_behavior': 'Tail-Verhalten (Überlebensfunktion)',
    'influence_plot': 'Einflussdiagramm',
    'diagnostic_plots': 'Diagnosediagramme',
    'comparison_plots': 'Verteilungsanpassung - Vergleichsdiagramme',
    'value': 'Wert',
    'density': 'Dichte',
    'cumulative_probability': 'Kumulative Wahrscheinlichkeit',
    'theoretical_quantiles': 'Theoretische Quantile',
    'empirical_quantiles': 'Empirische Quantile',
    'theoretical_probabilities': 'Theoretische Wahrscheinlichkeiten',
    'empirical_probabilities': 'Empirische Wahrscheinlichkeiten',
    'perfect_fit': 'Perfekte Anpassung',
    'data': 'Daten',
    'fitted': 'Angepasst',
    'empirical': 'Empirisch',
    'empirical_cdf': 'Empirische CDF',
    'residuals': 'Residuen',
    'standardized_residuals': 'Standardisierte Residuen',
    'zero_line': 'Nulllinie',
    'influence': 'Einfluss',
    'high_influence': 'Hoher Einfluss (>{threshold:.3f})',
    'data_points': 'Datenpunkte',
    'interactive_dashboard': 'Interaktives Verteilungsanpassungs-Dashboard - Beste: {model}',
}


# ═══════════════════════════════════════════════════════════════
# TRANSLATION MANAGER
# ═══════════════════════════════════════════════════════════════

TRANSLATIONS = {
    'en': EN_TRANSLATIONS,
    'fa': FA_TRANSLATIONS,
    'de': DE_TRANSLATIONS,
}


def t(key: str, **kwargs) -> str:
    """
    Translate a key to the current language
    
    Parameters:
    -----------
    key : str
        Translation key
    **kwargs : dict
        Format parameters for the translation string
        
    Returns:
    --------
    translation : str
        Translated string
        
    Example:
    --------
    >>> from distfit_pro.locales import t
    >>> from distfit_pro import set_language
    >>> 
    >>> set_language('en')
    >>> print(t('mean'))  # 'Mean (μ)'
    >>> 
    >>> set_language('fa')
    >>> print(t('mean'))  # 'میانگین (μ)'
    >>> 
    >>> set_language('de')
    >>> print(t('mean'))  # 'Mittelwert (μ)'
    """
    lang = get_language()
    translations = TRANSLATIONS.get(lang, EN_TRANSLATIONS)
    text = translations.get(key, key)
    
    # Format with kwargs if provided
    if kwargs:
        return text.format(**kwargs)
    return text


__all__ = ['t', 'TRANSLATIONS', 'EN_TRANSLATIONS', 'FA_TRANSLATIONS', 'DE_TRANSLATIONS']
