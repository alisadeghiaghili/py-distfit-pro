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
