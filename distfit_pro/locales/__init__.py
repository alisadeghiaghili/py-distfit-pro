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
    
    # Distribution characteristics and use cases
    'use_measurement_errors': 'Measurement errors',
    'use_height_weight': 'Height and weight',
    'use_test_scores': 'Test scores',
    'use_signal_noise': 'Signal noise',
    'use_income': 'Income',
    'use_stock_prices': 'Stock prices',
    'use_failure_time': 'Failure time',
    'use_reliability': 'Reliability analysis',
    'use_wind_speed': 'Wind speed',
    'use_waiting_time': 'Waiting time',
    'use_rainfall': 'Rainfall',
    'use_bayesian_prior': 'Bayesian prior',
    'use_time_between_events': 'Time between events',
    'use_component_lifetime': 'Component lifetime',
    'use_probabilities': 'Probabilities',
    'use_success_rate': 'Success rate',
    'use_random_number_gen': 'Random number generation',
    'use_uninformative_prior': 'Uninformative prior',
    'use_pert': 'PERT / Project estimation',
    'use_expert_estimation': 'Expert estimation',
    'use_logistic_regression': 'Logistic regression',
    'use_growth_models': 'Growth models',
    'use_flood': 'Flood analysis',
    'use_earthquake': 'Earthquake',
    'use_extreme_values': 'Extreme values (maxima)',
    'use_extreme_positive': 'Positive extreme values',
    'use_insurance': 'Insurance',
    'use_wealth': 'Wealth distribution',
    'use_80_20_rule': '80-20 rule',
    'use_physics': 'Physics',
    'use_resonance': 'Resonance',
    'use_hypothesis_testing': 'Hypothesis testing',
    'use_small_sample': 'Small sample analysis',
    'use_gof_test': 'Goodness-of-fit test',
    'use_variance_test': 'Variance testing',
    'use_anova': 'ANOVA',
    'use_variance_ratio': 'Variance ratio',
    'use_radar_signal': 'Radar signal',
    'use_differences': 'Differences',
    'use_lasso_regression': 'Lasso regression',
    'use_variance_prior': 'Prior for variance',
    'use_survival_analysis': 'Survival analysis',
    'use_event_counts': 'Event counts',
    'use_call_counts': 'Call counts',
    'use_success_failure': 'Success/failure experiments',
    'use_overdispersed_counts': 'Overdispersed counts',
    'use_time_to_success': 'Time until first success',
    'use_sampling_without_replacement': 'Sampling without replacement',
    
    'char_symmetric': 'Symmetric',
    'char_68_in_1sd': '68% within μ±σ',
    'char_95_in_2sd': '95% within μ±2σ',
    'char_right_skewed': 'Right-skewed',
    'char_positive': 'Only positive values',
    'char_k_less_1': 'k<1: decreasing failure rate',
    'char_k_eq_1': 'k=1: exponential',
    'char_k_gr_1': 'k>1: wear-out',
    'char_alpha_1_exponential': 'α=1: exponential',
    'char_alpha_large_normal': 'Large α → normal',
    'char_memoryless': 'Memoryless property',
    'char_constant_hazard': 'Constant hazard rate',
    'char_flexible': 'Flexible shape',
    'char_bounded_0_1': 'Bounded to [0,1]',
    'char_equal_probability': 'Equal probability',
    'char_max_entropy': 'Maximum entropy',
    'char_simple': 'Simple',
    'char_intuitive': 'Intuitive',
    'char_heavier_tails_than_normal': 'Heavier tails than normal',
    'char_positive_skew': 'Positive skew',
    'char_extreme_values': 'For extreme values',
    'char_very_heavy_tails': 'Very heavy tails',
    'char_power_law': 'Power law',
    'char_mean_undefined': 'Mean undefined',
    'char_extremely_heavy_tails': 'Extremely heavy tails',
    'char_special_case_gamma': 'Special case of Gamma',
    'char_skewed': 'Skewed',
    'char_heavy_tails_dist': 'Heavy tails',
    'char_mean_var_equal_lambda': 'Mean = Variance = λ',
    'char_n_independent_trials': 'n independent trials',
    'char_var_greater_mean': 'Variance > Mean',
    'char_memoryless_discrete': 'Memoryless (discrete)',
    'char_finite': 'Finite support',
    
    'warn_not_for_skewed': 'Not suitable for skewed data',
    'warn_positive_only': 'Only for positive values',
    'warn_no_mean_variance': 'Has no mean or variance',
    
    # Fitter and Results
    'fit_results': 'Distribution Fitting Results',
    'data_summary': 'Data Summary',
    'ci_95': '95% CI',
    'outliers': 'Outliers',
    'model_ranking': 'Model Ranking',
    'rank': 'Rank',
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
    
    # (Rest of translations - keeping all the existing ones)
    **{k: v for k, v in [
        ('right_skewed', '→ راست‌چوله (چولگی مثبت)'),
        ('left_skewed', '→ چپ‌چوله (چولگی منفی)'),
        ('symmetric', '→ تقریباً متقارن'),
        ('heavy_tails', '→ دنباله‌های سنگین'),
        ('light_tails', '→ دنباله‌های سبک'),
        ('normal_tails', '→ دنباله‌های شبیه نرمال'),
        ('not_fitted', 'هنوز فیت نشده است.'),
        ('for_explanation', 'برای توضیح مفهومی، از explain. استفاده کنید'),
        ('for_statistics', 'برای آمارهای کامل: dist.summary()'),
        ('undefined', 'نامعلوم'),
        ('na', 'ندارد'),
    ]},
    
    # All other existing FA translations
    'fit_results': 'نتایج فیت توزیع‌های آماری',
    'data_summary': 'خلاصه داده',
    'ci_95': 'فاصله اطمینان ۹۵٪',
    'fitting': 'در حال فیت',
    'distributions': 'توزیع',
    'fit_complete': 'فیت کامل شد',
}


# ═══════════════════════════════════════════════════════════════
# GERMAN TRANSLATIONS
# ═══════════════════════════════════════════════════════════════

DE_TRANSLATIONS = {
    **EN_TRANSLATIONS  # Fallback to English for now
    # TODO: Add German translations
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
    Translate a list of keys
    
    Parameters:
    -----------
    keys : list of str
        List of translation keys
        
    Returns:
    --------
    translations : list of str
        List of translated strings
        
    Example:
    --------
    >>> t_list(['use_income', 'use_stock_prices'])
    ['Income', 'Stock prices']  # (in English)
    ['درآمد', 'قیمت سهام']  # (in Persian)
    """
    return [t(key) for key in keys]


__all__ = ['t', 't_list', 'TRANSLATIONS', 'EN_TRANSLATIONS', 'FA_TRANSLATIONS', 'DE_TRANSLATIONS']
