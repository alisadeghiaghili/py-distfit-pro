"""
Internationalization Tests
==========================

Test multilingual support.
"""

import pytest
import numpy as np
from distfit_pro.core.config import config
from distfit_pro.locales import t
from distfit_pro import get_distribution


class TestI18n:
    """Test translation system"""
    
    def test_english(self):
        """Test English translations"""
        config.set_language('en')
        assert t('mean') == 'Mean'
        assert t('std_dev') == 'Standard Deviation'
    
    def test_persian(self):
        """Test Persian translations"""
        config.set_language('fa')
        assert t('mean') == 'میانگین'
        assert t('std_dev') == 'انحراف معیار'
    
    def test_german(self):
        """Test German translations"""
        config.set_language('de')
        assert t('mean') == 'Mittelwert'
        assert t('std_dev') == 'Standardabweichung'
    
    def test_verbose_output_languages(self):
        """Test verbose output in different languages"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        for lang in ['en', 'fa', 'de']:
            config.set_language(lang)
            config.set_verbosity('silent')  # Don't print during test
            
            dist = get_distribution('normal')
            dist.fit(data)
            
            assert dist.fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
