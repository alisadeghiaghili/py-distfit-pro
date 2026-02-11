"""
Global Configuration System
============================

Manages package-wide settings including language preferences.

Example:
--------
>>> from distfit_pro import set_language, get_language
>>> 
>>> # Set output language
>>> set_language('en')  # English
>>> set_language('fa')  # Persian/Farsi
>>> set_language('de')  # German
>>> 
>>> # Get current language
>>> print(get_language())  # 'en'
"""

from typing import Literal

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'fa', 'de']
LanguageCode = Literal['en', 'fa', 'de']


class Config:
    """
    Global configuration singleton
    
    Attributes:
    -----------
    language : str
        Current output language ('en', 'fa', or 'de')
    """
    
    def __init__(self):
        self._language: LanguageCode = 'en'  # Default: English
    
    @property
    def language(self) -> LanguageCode:
        """Get current language"""
        return self._language
    
    @language.setter
    def language(self, value: str):
        """Set current language"""
        value = value.lower()
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{value}'. "
                f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        self._language = value


# Global config instance
_config = Config()


def get_language() -> LanguageCode:
    """
    Get current output language
    
    Returns:
    --------
    language : str
        Current language code ('en', 'fa', or 'de')
        
    Example:
    --------
    >>> from distfit_pro import get_language
    >>> print(get_language())
    'en'
    """
    return _config.language


def set_language(language: LanguageCode):
    """
    Set output language for all package outputs
    
    Parameters:
    -----------
    language : str
        Language code: 'en' (English), 'fa' (Persian), or 'de' (German)
        
    Example:
    --------
    >>> from distfit_pro import set_language
    >>> 
    >>> # Use English
    >>> set_language('en')
    >>> 
    >>> # Use Persian/Farsi
    >>> set_language('fa')
    >>> 
    >>> # Use German
    >>> set_language('de')
    """
    _config.language = language


def get_config() -> Config:
    """
    Get global configuration instance
    
    Returns:
    --------
    config : Config
        Global configuration object
    """
    return _config
