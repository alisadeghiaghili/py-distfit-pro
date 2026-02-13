"""
Internationalization (i18n) System
===================================

Provides translations for package outputs in multiple languages.

Supported Languages:
- English (en)
- Persian/Farsi (fa)
- German (de)
"""

from typing import Dict, List
try:
    from ..core.config import get_language
except:
    # Fallback if config not available
    def get_language():
        return 'en'


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
    
    # Section headers for verbose mode
    'section_fitting': 'Fitting',
    'section_fitted_parameters': 'Fitted Parameters',
    'section_distribution_statistics': 'Distribution Statistics',
    
    # Verbose mode - Data characteristics
    'data_characteristics': 'Data Characteristics',
    'sample_size': 'Sample size',
    'observations': 'observations',
    'mean': 'Mean',
    'std_dev': 'Standard Deviation',
    'skewness': 'Skewness',
    'data_approximately_symmetric': 'Data is approximately symmetric',
    'data_right_skewed': 'Data is right-skewed (long tail on the right)',
    'data_left_skewed': 'Data is left-skewed (long tail on the left)',
    
    # Verbose mode - Fitting process
    'fitting_process': 'Fitting Process',
    'distribution': 'Distribution',
    'method': 'Method',
    'about_method': 'About this method',
    'mle_explanation': 'Maximum Likelihood Estimation finds parameters that maximize the probability of observing your data.',
    'moments_explanation': 'Method of Moments matches theoretical moments (mean, variance) with sample moments.',
    'method_explanation': 'This method fits the distribution to your data.',
    
    # Verbose mode - Parameter explanations
    'meaning': 'Meaning',
    'impact': 'Practical Impact',
    
    # Parameter meanings
    'param_loc_meaning': 'Location parameter (center of distribution)',
    'param_scale_meaning': 'Scale parameter (spread of distribution)',
    'param_shape_meaning': 'Shape parameter (affects distribution shape)',
    'param_rate_meaning': 'Rate parameter (inverse of scale)',
    'param_df_meaning': 'Degrees of freedom',
    
    # Practical impacts
    'impact_low_variability': 'Low variability - data tightly clustered',
    'impact_moderate_variability': 'Moderate variability',
    'impact_high_variability': 'High variability - data widely spread',
    'impact_see_docs': 'See documentation for interpretation',
    
    # Statistics explanations
    'stat_expected_value': 'Expected value',
    'stat_typical_deviation': 'Typical deviation from mean',
    'stat_50_percent_below': '50% of data below this value',
    'stat_most_common_value': 'Most common value',
    'stat_approximately_symmetric': 'Approximately symmetric',
    'stat_normal_tail_behavior': 'Normal tail behavior',
    
    # Fitting success messages
    'fitting_completed_successfully': 'fitting completed successfully',
    
    # Statistics
    'median': 'Median',
    'mode': 'Mode',
    'variance': 'Variance (σ²)',
    'std_deviation': 'Std Deviation (σ)',
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
}


# ═══════════════════════════════════════════════════════════════
# PERSIAN/FARSI TRANSLATIONS  
# ═══════════════════════════════════════════════════════════════

FA_TRANSLATIONS = {
    # Distribution info headers
    'estimated_parameters': 'پارامترهای برآورد شده',
    'location_statistics': 'آماره‌های مکانی',
    'spread_statistics': 'آماره‌های پراکندگی',
    'shape_statistics': 'آماره‌های شکل',
    'key_quantiles': 'چارک‌های کلیدی',
    'practical_applications': 'کاربردهای عملی',
    'characteristics': 'ویژگی‌های این توزیع',
    'warning': 'هشدار',
    
    # Section headers
    'section_fitting': 'فیت کردن',
    'section_fitted_parameters': 'پارامترهای فیت شده',
    'section_distribution_statistics': 'آماره‌های توزیع',
    
    # Verbose mode - Data characteristics
    'data_characteristics': 'ویژگی‌های داده',
    'sample_size': 'تعداد نمونه',
    'observations': 'مشاهده',
    'mean': 'میانگین',
    'std_dev': 'انحراف معیار',
    'skewness': 'چولگی',
    'data_approximately_symmetric': 'داده تقریباً متقارن است',
    'data_right_skewed': 'داده راست‌چوله است (دنباله بلند سمت راست)',
    'data_left_skewed': 'داده چپ‌چوله است (دنباله بلند سمت چپ)',
    
    # Verbose mode - Fitting process
    'fitting_process': 'فرآیند فیت کردن',
    'distribution': 'توزیع',
    'method': 'روش',
    'about_method': 'درباره این روش',
    'mle_explanation': 'برآورد حداکثر درستنمایی پارامترهایی را پیدا می‌کند که احتمال مشاهده داده شما را حداکثر می‌کنند.',
    'moments_explanation': 'روش گشتاورها، گشتاورهای نظری (میانگین، واریانس) را با گشتاورهای نمونه برابر قرار می‌دهد.',
    'method_explanation': 'این روش توزیع را با داده شما فیت می‌کند.',
    
    # Verbose mode - Parameter explanations
    'meaning': 'معنی',
    'impact': 'تأثیر عملی',
    
    # Parameter meanings
    'param_loc_meaning': 'پارامتر مکان (مرکز توزیع)',
    'param_scale_meaning': 'پارامتر مقیاس (پراکندگی توزیع)',
    'param_shape_meaning': 'پارامتر شکل (تأثیر روی فرم توزیع)',
    'param_rate_meaning': 'پارامتر نرخ (معکوس مقیاس)',
    'param_df_meaning': 'درجه آزادی',
    
    # Practical impacts
    'impact_low_variability': 'پراکندگی کم - داده‌ها به هم نزدیک',
    'impact_moderate_variability': 'پراکندگی متوسط',
    'impact_high_variability': 'پراکندگی زیاد - داده‌ها پراکنده',
    'impact_see_docs': 'برای تفسیر به مستندات مراجعه کنید',
    
    # Statistics explanations
    'stat_expected_value': 'مقدار مورد انتظار',
    'stat_typical_deviation': 'انحراف معمول از میانگین',
    'stat_50_percent_below': '50% داده‌ها زیر این مقدار',
    'stat_most_common_value': 'مقدار رایج‌تر',
    'stat_approximately_symmetric': 'تقریباً متقارن',
    'stat_normal_tail_behavior': 'رفتار دنباله نرمال',
    
    # Fitting success messages
    'fitting_completed_successfully': 'فیت با موفقیت کامل شد',
    
    # Statistics
    'median': 'میانه',
    'mode': 'مد (نما)',
    'variance': 'واریانس (σ²)',
    'std_deviation': 'انحراف معیار (σ)',
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
    'for_statistics': 'برای آماره‌های کامل: dist.summary()',
    'undefined': 'نامعلوم',
    'na': 'ندارد',
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
    
    # Section headers for verbose mode
    'section_fitting': 'Anpassung',
    'section_fitted_parameters': 'Angepasste Parameter',
    'section_distribution_statistics': 'Verteilungsstatistiken',
    
    # Verbose mode - Data characteristics
    'data_characteristics': 'Datenmerkmale',
    'sample_size': 'Stichprobengröße',
    'observations': 'Beobachtungen',
    'mean': 'Mittelwert',
    'std_dev': 'Standardabweichung',
    'skewness': 'Schiefe',
    'data_approximately_symmetric': 'Daten sind annähernd symmetrisch',
    'data_right_skewed': 'Daten sind rechtsschief (langer Schwanz rechts)',
    'data_left_skewed': 'Daten sind linksschief (langer Schwanz links)',
    
    # Verbose mode - Fitting process
    'fitting_process': 'Anpassungsprozess',
    'distribution': 'Verteilung',
    'method': 'Methode',
    'about_method': 'Über diese Methode',
    'mle_explanation': 'Maximum-Likelihood-Schätzung findet Parameter, die die Wahrscheinlichkeit maximieren, Ihre Daten zu beobachten.',
    'moments_explanation': 'Momentenmethode passt theoretische Momente (Mittelwert, Varianz) an Stichprobenmomente an.',
    'method_explanation': 'Diese Methode passt die Verteilung an Ihre Daten an.',
    
    # Verbose mode - Parameter explanations
    'meaning': 'Bedeutung',
    'impact': 'Praktische Auswirkung',
    
    # Parameter meanings
    'param_loc_meaning': 'Lageparameter (Zentrum der Verteilung)',
    'param_scale_meaning': 'Skalierungsparameter (Streuung der Verteilung)',
    'param_shape_meaning': 'Formparameter (beeinflusst Verteilungsform)',
    'param_rate_meaning': 'Ratenparameter (Kehrwert der Skalierung)',
    'param_df_meaning': 'Freiheitsgrade',
    
    # Practical impacts
    'impact_low_variability': 'Geringe Variabilität - Daten eng gruppiert',
    'impact_moderate_variability': 'Mittlere Variabilität',
    'impact_high_variability': 'Hohe Variabilität - Daten weit gestreut',
    'impact_see_docs': 'Siehe Dokumentation für Interpretation',
    
    # Statistics explanations
    'stat_expected_value': 'Erwartungswert',
    'stat_typical_deviation': 'Typische Abweichung vom Mittelwert',
    'stat_50_percent_below': '50% der Daten unter diesem Wert',
    'stat_most_common_value': 'Häufigster Wert',
    'stat_approximately_symmetric': 'Annähernd symmetrisch',
    'stat_normal_tail_behavior': 'Normales Schwanzverhalten',
    
    # Fitting success messages
    'fitting_completed_successfully': 'Anpassung erfolgreich abgeschlossen',
    
    # Statistics
    'median': 'Median',
    'mode': 'Modus',
    'variance': 'Varianz (σ²)',
    'std_deviation': 'Standardabweichung (σ)',
    'kurtosis': 'Kurtosis (Schwanzgewicht)',
    'percentile': '{p}. Perzentil',
    
    # Skewness interpretations
    'right_skewed': '→ Rechtsschief (positiv)',
    'left_skewed': '→ Linksschief (negativ)',
    'symmetric': '→ Annähernd symmetrisch',
    
    # Kurtosis interpretations
    'heavy_tails': '→ Schwere Schwänze (leptokurtisch)',
    'light_tails': '→ Leichte Schwänze (platykurtisch)',
    'normal_tails': '→ Normalähnliche Schwänze',
    
    # Messages
    'not_fitted': 'wurde noch nicht angepasst.',
    'for_explanation': 'Für konzeptionelle Erklärung, verwenden Sie .explain()',
    'for_statistics': 'Für vollständige Statistiken: dist.summary()',
    'undefined': 'Undefiniert',
    'na': 'N/V',
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
    Translate a key to the current language.
    
    Parameters
    ----------
    key : str
        Translation key
    **kwargs : dict
        Format parameters for the translation string
        
    Returns
    -------
    translation : str
        Translated string
        
    Examples
    --------
    >>> from distfit_pro.locales import t
    >>> t('mean')
    'Mean'  # (if language is 'en')
    'میانگین'  # (if language is 'fa')
    'Mittelwert'  # (if language is 'de')
    """
    lang = get_language()
    translations = TRANSLATIONS.get(lang, EN_TRANSLATIONS)
    text = translations.get(key, key)
    
    # Format with kwargs if provided
    if kwargs:
        return text.format(**kwargs)
    return text


def t_list(keys: List[str]) -> List[str]:
    """
    Translate a list of keys.
    
    Parameters
    -----------
    keys : list of str
        List of translation keys
        
    Returns
    --------
    translations : list of str
        List of translated strings
        
    Example
    -------
    >>> t_list(['mean', 'median', 'mode'])
    ['Mean', 'Median', 'Mode']  # English
    ['میانگین', 'میانه', 'مد']  # Persian
    ['Mittelwert', 'Median', 'Modus']  # German
    """
    return [t(key) for key in keys]


__all__ = ['t', 't_list', 'TRANSLATIONS', 'EN_TRANSLATIONS', 'FA_TRANSLATIONS', 'DE_TRANSLATIONS']
