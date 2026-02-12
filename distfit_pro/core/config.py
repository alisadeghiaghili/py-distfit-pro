"""
Configuration Module
===================

Global configuration for distfit-pro including:
- Language settings
- Verbosity levels
- Default parameters

Author: Ali Sadeghi Aghili
"""

import os
from typing import Optional, Literal
from enum import IntEnum
from contextlib import contextmanager


class VerbosityLevel(IntEnum):
    """
    Verbosity levels for package output.
    
    SILENT (0): No output except errors
    NORMAL (1): Basic progress messages (default)
    VERBOSE (2): Detailed explanations and statistics
    DEBUG (3): Everything including internal details
    """
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class Config:
    """
    Global configuration singleton for distfit-pro.
    
    Controls:
    - Language for all messages (en, fa, de)
    - Verbosity level for explanations
    - Default fitting parameters
    - Plot settings
    
    Examples
    --------
    >>> from distfit_pro.core.config import config
    >>> 
    >>> # Set language
    >>> config.set_language('fa')
    >>> 
    >>> # Set verbosity
    >>> config.set_verbosity('verbose')
    >>> 
    >>> # Temporary verbosity change
    >>> with config.verbose_mode():
    ...     # Code here runs with VERBOSE level
    ...     pass
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Language settings
        self._language = os.getenv('DISTFIT_LANG', 'en')
        
        # Verbosity settings
        self._verbosity = VerbosityLevel.NORMAL
        
        # Default fitting parameters
        self.default_fitting_method = 'mle'
        self.default_gof_test = 'ks'
        self.default_bootstrap_samples = 1000
        self.default_confidence_level = 0.95
        
        # Plot settings
        self.default_figure_size = (14, 10)
        self.default_dpi = 100
        self.show_plots_by_default = True
        
        self._initialized = True
    
    @property
    def language(self) -> str:
        """Current language setting."""
        return self._language
    
    @property
    def verbosity(self) -> VerbosityLevel:
        """Current verbosity level."""
        return self._verbosity
    
    def set_language(self, lang: Literal['en', 'fa', 'de']):
        """
        Set the language for all messages and explanations.
        
        Parameters
        ----------
        lang : {'en', 'fa', 'de'}
            Language code
        
        Examples
        --------
        >>> config.set_language('fa')  # Persian
        >>> config.set_language('en')  # English
        >>> config.set_language('de')  # German
        """
        if lang not in ['en', 'fa', 'de']:
            raise ValueError(f"Unsupported language: {lang}. Use 'en', 'fa', or 'de'")
        self._language = lang
    
    def set_verbosity(
        self,
        level: Literal['silent', 'normal', 'verbose', 'debug', 0, 1, 2, 3]
    ):
        """
        Set verbosity level for package output.
        
        Parameters
        ----------
        level : str or int
            Verbosity level:
            - 'silent' or 0: No output except errors
            - 'normal' or 1: Basic progress (default)
            - 'verbose' or 2: Detailed explanations
            - 'debug' or 3: Everything including internals
        
        Examples
        --------
        >>> config.set_verbosity('verbose')
        >>> config.set_verbosity(2)  # Same as 'verbose'
        >>> config.set_verbosity('silent')  # Quiet mode
        """
        if isinstance(level, str):
            level_map = {
                'silent': VerbosityLevel.SILENT,
                'normal': VerbosityLevel.NORMAL,
                'verbose': VerbosityLevel.VERBOSE,
                'debug': VerbosityLevel.DEBUG
            }
            if level not in level_map:
                raise ValueError(
                    f"Unknown verbosity level: {level}. "
                    f"Use 'silent', 'normal', 'verbose', or 'debug'"
                )
            self._verbosity = level_map[level]
        elif isinstance(level, int):
            if level not in [0, 1, 2, 3]:
                raise ValueError(f"Verbosity level must be 0-3, got {level}")
            self._verbosity = VerbosityLevel(level)
        else:
            raise TypeError(f"Invalid type for verbosity: {type(level)}")
    
    def is_verbose(self) -> bool:
        """Check if verbosity is VERBOSE or higher."""
        return self._verbosity >= VerbosityLevel.VERBOSE
    
    def is_debug(self) -> bool:
        """Check if verbosity is DEBUG."""
        return self._verbosity >= VerbosityLevel.DEBUG
    
    def is_silent(self) -> bool:
        """Check if verbosity is SILENT."""
        return self._verbosity == VerbosityLevel.SILENT
    
    @contextmanager
    def temp_verbosity(self, level: Literal['silent', 'normal', 'verbose', 'debug']):
        """
        Temporarily change verbosity level.
        
        Parameters
        ----------
        level : str
            Temporary verbosity level
        
        Examples
        --------
        >>> with config.temp_verbosity('verbose'):
        ...     # Code here runs with verbose output
        ...     fitter.fit(data)
        """
        old_level = self._verbosity
        try:
            self.set_verbosity(level)
            yield
        finally:
            self._verbosity = old_level
    
    @contextmanager
    def verbose_mode(self):
        """Context manager for VERBOSE mode."""
        with self.temp_verbosity('verbose'):
            yield
    
    @contextmanager
    def silent_mode(self):
        """Context manager for SILENT mode."""
        with self.temp_verbosity('silent'):
            yield
    
    @contextmanager
    def debug_mode(self):
        """Context manager for DEBUG mode."""
        with self.temp_verbosity('debug'):
            yield
    
    def reset(self):
        """Reset to default settings."""
        self._language = 'en'
        self._verbosity = VerbosityLevel.NORMAL
        self.default_fitting_method = 'mle'
        self.default_gof_test = 'ks'
        self.default_bootstrap_samples = 1000
        self.default_confidence_level = 0.95


# Global singleton instance
config = Config()


# Convenience functions
def get_language() -> str:
    """Get current language setting."""
    return config.language


def set_language(lang: str):
    """Set language."""
    config.set_language(lang)


def get_verbosity() -> VerbosityLevel:
    """Get current verbosity level."""
    return config.verbosity


def set_verbosity(level):
    """Set verbosity level."""
    config.set_verbosity(level)


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return config.is_verbose()


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return config.is_debug()


def is_silent() -> bool:
    """Check if silent mode is enabled."""
    return config.is_silent()
