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
    'goodness_of_fit': 'GOODNESS OF FIT',
    'warning': 'Warning',
    
    # Goodness of fit metrics
    'log_likelihood': 'Log-Likelihood',
    'aic': 'AIC',
    'bic': 'BIC',
    
    # Distribution descriptions (25 total)
    'normal_description': 'Symmetric bell-shaped continuous distribution. Fundamental in statistics due to Central Limit Theorem.',
    'exponential_description': 'Memoryless continuous distribution modeling waiting times.',
    'uniform_description': 'Constant probability over a finite interval.',
    'gamma_description': 'Flexible distribution for positive continuous data. Sum of exponentials.',
    'beta_description': 'Flexible distribution for data bounded between 0 and 1.',
    'weibull_description': 'Widely used in reliability analysis and lifetime modeling.',
    'lognormal_description': 'Distribution of variable whose logarithm is normally distributed.',
    'logistic_description': 'Similar to normal but with heavier tails.',
    'gumbel_description': 'Type I extreme value distribution.',
    'pareto_description': 'Power law distribution.',
    'cauchy_description': 'Heavy-tailed distribution with undefined mean and variance.',
    'studentt_description': 'Heavy-tailed distribution used in statistical inference.',
    'chisquare_description': 'Distribution of sum of squared standard normal variables.',
    'f_description': 'Ratio of two chi-square distributions.',
    'laplace_description': 'Double exponential distribution.',
    'rayleigh_description': 'Models magnitude of 2D vector with normally distributed components.',
    'wald_description': 'Inverse Gaussian distribution.',
    'triangular_description': 'Simple distribution defined by minimum, maximum, and mode.',
    'burr_description': 'Flexible distribution for modeling heavy-tailed data.',
    'genextreme_description': 'Family of distributions for modeling extreme values.',
    'poisson_description': 'Models count of events in fixed interval with constant rate.',
    'binomial_description': 'Number of successes in n independent Bernoulli trials.',
    'negative_binomial_description': 'Number of failures before r-th success.',
    'geometric_description': 'Number of trials until first success.',
    'hypergeometric_description': 'Sampling without replacement from finite population.',
    
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
    'goodness_of_fit': 'نیکویی برازش',
    'warning': 'هشدار',
    
    # Goodness of fit metrics
    'log_likelihood': 'لگاریتم درستنمایی',
    'aic': 'معیار اطلاعات آکائیکه (AIC)',
    'bic': 'معیار اطلاعات بیزی (BIC)',
    
    # Distribution descriptions (25 total)
    'normal_description': 'توزیع پیوسته متقارن زنگوله‌ای. پایه و اساس آمار به دلیل قضیه حد مرکزی.',
    'exponential_description': 'توزیع پیوسته بدون حافظه برای مدل‌سازی زمان انتظار.',
    'uniform_description': 'احتمال ثابت در یک بازه محدود.',
    'gamma_description': 'توزیع انعطاف‌پذیر برای داده‌های مثبت پیوسته. مجموع نماییات.',
    'beta_description': 'توزیع انعطاف‌پذیر برای داده‌های محدود بین 0 و 1.',
    'weibull_description': 'پرکاربرد در تحلیل قابلیت اطمینان و مدل‌سازی طول عمر.',
    'lognormal_description': 'توزیع متغیری که لگاریتم آن نرمال است.',
    'logistic_description': 'شبیه نرمال اما با دنباله‌های سنگین‌تر.',
    'gumbel_description': 'توزیع مقادیر حدی نوع اول.',
    'pareto_description': 'توزیع قانون توان.',
    'cauchy_description': 'توزیع دنباله‌سنگین با میانگین و واریانس نامعلوم.',
    'studentt_description': 'توزیع دنباله‌سنگین در استنباط آماری.',
    'chisquare_description': 'توزیع مجموع مربعات متغیرهای نرمال استاندارد.',
    'f_description': 'نسبت دو توزیع کای‌دو.',
    'laplace_description': 'توزیع نمایی دوگانه.',
    'rayleigh_description': 'مدل‌سازی بزرگی بردار دوبعدی با مؤلفه‌های نرمال.',
    'wald_description': 'توزیع گوسی معکوس.',
    'triangular_description': 'توزیع ساده با حداقل، حداکثر و مد.',
    'burr_description': 'توزیع انعطاف‌پذیر برای داده‌های دنباله‌سنگین.',
    'genextreme_description': 'خانواده توزیع‌ها برای مدل‌سازی مقادیر حدی.',
    'poisson_description': 'مدل شمارش رویدادها در بازه ثابت با نرخ ثابت.',
    'binomial_description': 'تعداد موفقیت در n آزمایش برنولی مستقل.',
    'negative_binomial_description': 'تعداد شکست قبل از rامین موفقیت.',
    'geometric_description': 'تعداد آزمایش تا اولین موفقیت.',
    'hypergeometric_description': 'نمونه‌گیری بدون جایگزینی از جمعیت محدود.',
    
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
    'goodness_of_fit': 'ANPASSUNGSGÜTE',
    'warning': 'Warnung',
    
    # Goodness of fit metrics (LOG-LIKELIHOOD FIXED!)
    'log_likelihood': 'Logarithmische Likelihood',
    'aic': 'AIC',
    'bic': 'BIC',
    
    # Distribution descriptions (25 total)
    'normal_description': 'Symmetrische glockenförmige stetige Verteilung. Grundlegend in der Statistik aufgrund des Zentralen Grenzwertsatzes.',
    'exponential_description': 'Gedächtnislose stetige Verteilung zur Modellierung von Wartezeiten.',
    'uniform_description': 'Konstante Wahrscheinlichkeit über ein endliches Intervall.',
    'gamma_description': 'Flexible Verteilung für positive stetige Daten. Summe von Exponentialverteilungen.',
    'beta_description': 'Flexible Verteilung für Daten zwischen 0 und 1.',
    'weibull_description': 'Weit verbreitet in der Zuverlässigkeitsanalyse und Lebensdauermodellierung.',
    'lognormal_description': 'Verteilung einer Variable, deren Logarithmus normalverteilt ist.',
    'logistic_description': 'Ähnlich wie Normal, aber mit schwereren Schwänzen.',
    'gumbel_description': 'Extremwertverteilung Typ I.',
    'pareto_description': 'Potenzgesetzverteilung.',
    'cauchy_description': 'Verteilung mit schweren Schwänzen und undefiniertem Mittelwert und Varianz.',
    'studentt_description': 'Verteilung mit schweren Schwänzen für statistische Inferenz.',
    'chisquare_description': 'Verteilung der Summe quadrierter standardnormalverteilter Variablen.',
    'f_description': 'Verhältnis zweier Chi-Quadrat-Verteilungen.',
    'laplace_description': 'Doppelte Exponentialverteilung.',
    'rayleigh_description': 'Modelliert Betrag eines 2D-Vektors mit normalverteilten Komponenten.',
    'wald_description': 'Inverse Gaußverteilung.',
    'triangular_description': 'Einfache Verteilung definiert durch Minimum, Maximum und Modus.',
    'burr_description': 'Flexible Verteilung zur Modellierung von Daten mit schweren Schwänzen.',
    'genextreme_description': 'Familie von Verteilungen zur Modellierung von Extremwerten.',
    'poisson_description': 'Modelliert Anzahl von Ereignissen in festem Intervall mit konstanter Rate.',
    'binomial_description': 'Anzahl der Erfolge in n unabhängigen Bernoulli-Versuchen.',
    'negative_binomial_description': 'Anzahl der Misserfolge vor dem r-ten Erfolg.',
    'geometric_description': 'Anzahl der Versuche bis zum ersten Erfolg.',
    'hypergeometric_description': 'Stichprobenziehung ohne Zurücklegen aus endlicher Grundgesamtheit.',
    
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
