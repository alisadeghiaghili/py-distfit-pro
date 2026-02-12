"""
Verbose Logging Utilities
========================

Structured verbose output with multilingual support.

Author: Ali Sadeghi Aghili
"""

import sys
from typing import Optional, Any, Dict, List
from datetime import datetime
import numpy as np
from ..core.config import config, VerbosityLevel
from ..locales import t


class VerboseLogger:
    """
    Logger for verbose and self-explanatory output.
    
    Automatically translates all messages to the configured language
    and respects the global verbosity level.
    
    Examples
    --------
    >>> from distfit_pro.utils import VerboseLogger
    >>> logger = VerboseLogger('MyModule')
    >>> 
    >>> logger.info("Starting process...")
    >>> logger.verbose("Detailed step explanation")
    >>> logger.debug("Internal variable: x = {x}", x=42)
    """
    
    def __init__(self, name: str = "distfit-pro"):
        self.name = name
    
    def _should_print(self, level: VerbosityLevel) -> bool:
        """Check if message should be printed based on verbosity."""
        return config.verbosity >= level
    
    def _format_message(self, level_str: str, message: str) -> str:
        """Format message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {level_str}: {message}"
    
    def info(self, message: str, **kwargs):
        """
        Print informational message (NORMAL level).
        
        Parameters
        ----------
        message : str
            Message to print (will be translated if starts with 'msg_')
        **kwargs
            Format arguments for message
        """
        if not self._should_print(VerbosityLevel.NORMAL):
            return
        
        # Translate if message key
        if message.startswith('msg_'):
            message = t(message, **kwargs)
        else:
            message = message.format(**kwargs) if kwargs else message
        
        print(f"ℹ️  {message}")
    
    def verbose(self, message: str, **kwargs):
        """
        Print verbose message with detailed explanation (VERBOSE level).
        
        Parameters
        ----------
        message : str
            Detailed message
        **kwargs
            Format arguments
        """
        if not self._should_print(VerbosityLevel.VERBOSE):
            return
        
        if message.startswith('msg_'):
            message = t(message, **kwargs)
        else:
            message = message.format(**kwargs) if kwargs else message
        
        print(f"📝 {message}")
    
    def debug(self, message: str, **kwargs):
        """
        Print debug message (DEBUG level).
        
        Parameters
        ----------
        message : str
            Debug message
        **kwargs
            Format arguments
        """
        if not self._should_print(VerbosityLevel.DEBUG):
            return
        
        message = message.format(**kwargs) if kwargs else message
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"🐛 [{timestamp}] DEBUG: {message}")
    
    def success(self, message: str, **kwargs):
        """
        Print success message (NORMAL level).
        
        Parameters
        ----------
        message : str
            Success message
        **kwargs
            Format arguments
        """
        if not self._should_print(VerbosityLevel.NORMAL):
            return
        
        if message.startswith('msg_'):
            message = t(message, **kwargs)
        else:
            message = message.format(**kwargs) if kwargs else message
        
        print(f"✅ {message}")
    
    def warning(self, message: str, **kwargs):
        """
        Print warning message (always shown unless SILENT).
        
        Parameters
        ----------
        message : str
            Warning message
        **kwargs
            Format arguments
        """
        if config.is_silent():
            return
        
        if message.startswith('msg_'):
            message = t(message, **kwargs)
        else:
            message = message.format(**kwargs) if kwargs else message
        
        print(f"⚠️  {message}")
    
    def error(self, message: str, **kwargs):
        """
        Print error message (always shown).
        
        Parameters
        ----------
        message : str
            Error message
        **kwargs
            Format arguments
        """
        if message.startswith('msg_'):
            message = t(message, **kwargs)
        else:
            message = message.format(**kwargs) if kwargs else message
        
        print(f"❌ {message}", file=sys.stderr)
    
    def section(self, title: str):
        """
        Print section header (NORMAL level).
        
        Parameters
        ----------
        title : str
            Section title
        """
        if not self._should_print(VerbosityLevel.NORMAL):
            return
        
        if title.startswith('msg_'):
            title = t(title)
        
        print(f"\n{'='*70}")
        print(f"📊 {title}")
        print(f"{'='*70}")
    
    def subsection(self, title: str):
        """
        Print subsection header (VERBOSE level).
        
        Parameters
        ----------
        title : str
            Subsection title
        """
        if not self._should_print(VerbosityLevel.VERBOSE):
            return
        
        if title.startswith('msg_'):
            title = t(title)
        
        print(f"\n{'-'*70}")
        print(f"📌 {title}")
        print(f"{'-'*70}")
    
    def explain_statistic(self, name: str, value: float, interpretation: Optional[str] = None):
        """
        Explain a statistical value (VERBOSE level).
        
        Parameters
        ----------
        name : str
            Statistic name
        value : float
            Statistic value
        interpretation : str, optional
            Human-readable interpretation
        """
        if not self._should_print(VerbosityLevel.VERBOSE):
            return
        
        if name.startswith('msg_'):
            name = t(name)
        
        print(f"   📊 {name}: {value:.6f}")
        if interpretation:
            if interpretation.startswith('msg_'):
                interpretation = t(interpretation)
            print(f"      → {interpretation}")
    
    def explain_parameter(self, name: str, value: float, meaning: str, practical_impact: Optional[str] = None):
        """
        Explain a distribution parameter in detail (VERBOSE level).
        
        Parameters
        ----------
        name : str
            Parameter name
        value : float
            Parameter value
        meaning : str
            What this parameter means
        practical_impact : str, optional
            Practical interpretation
        """
        if not self._should_print(VerbosityLevel.VERBOSE):
            return
        
        print(f"\n   🔧 {name} = {value:.6f}")
        print(f"      📖 {t('meaning')}: {meaning}")
        if practical_impact:
            print(f"      💡 {t('impact')}: {practical_impact}")
    
    def explain_data_characteristics(self, data: np.ndarray):
        """
        Explain data characteristics in plain language (VERBOSE level).
        
        Parameters
        ----------
        data : ndarray
            Input data
        """
        if not self._should_print(VerbosityLevel.VERBOSE):
            return
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        skew = self._calculate_skewness(data)
        
        self.subsection(t('data_characteristics'))
        
        print(f"   📏 {t('sample_size')}: {n} {t('observations')}")
        print(f"   📊 {t('mean')}: {mean:.4f}")
        print(f"   📊 {t('std_dev')}: {std:.4f}")
        
        # Explain skewness
        if abs(skew) < 0.5:
            skew_interpretation = t('data_approximately_symmetric')
        elif skew > 0:
            skew_interpretation = t('data_right_skewed')
        else:
            skew_interpretation = t('data_left_skewed')
        
        print(f"   📊 {t('skewness')}: {skew:.4f}")
        print(f"      → {skew_interpretation}")
    
    def explain_fitting_process(self, distribution_name: str, method: str, data_size: int):
        """
        Explain the fitting process (VERBOSE level).
        
        Parameters
        ----------
        distribution_name : str
            Name of distribution being fitted
        method : str
            Fitting method (MLE, moments, etc.)
        data_size : int
            Size of dataset
        """
        if not self._should_print(VerbosityLevel.VERBOSE):
            return
        
        self.subsection(t('fitting_process'))
        
        print(f"   🎯 {t('distribution')}: {distribution_name}")
        print(f"   🔧 {t('method')}: {method.upper()}")
        print(f"   📏 {t('sample_size')}: {data_size}")
        
        # Explain method
        if method == 'mle':
            explanation = t('mle_explanation')
        elif method == 'moments':
            explanation = t('moments_explanation')
        else:
            explanation = t('method_explanation')
        
        print(f"\n   📖 {t('about_method')}:")
        print(f"      {explanation}")
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        from scipy import stats
        return stats.skew(data)
    
    def progress_bar(self, current: int, total: int, description: str = ""):
        """
        Simple progress indicator (NORMAL level).
        
        Parameters
        ----------
        current : int
            Current progress
        total : int
            Total items
        description : str, optional
            Progress description
        """
        if not self._should_print(VerbosityLevel.NORMAL):
            return
        
        percentage = int(100 * current / total)
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r  [{bar}] {percentage}% - {description}", end='', flush=True)
        
        if current == total:
            print()  # New line when complete


# Global logger instance
logger = VerboseLogger()
