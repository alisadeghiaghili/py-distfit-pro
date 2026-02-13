"""
Basic MVP Tests
===============

Simple tests for core functionality.
"""

import pytest
import numpy as np
from distfit_pro import (
    get_distribution,
    NormalDistribution,
    ExponentialDistribution,
)


class TestBasicFitting:
    """Test basic distribution fitting"""
    
    def test_normal_fit(self):
        """Test normal distribution fitting"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=500)
        
        dist = get_distribution('normal')
        dist.fit(data, method='mle')
        
        assert dist.fitted
        params = dist.params
        assert 'loc' in params
        assert 'scale' in params
        assert abs(params['loc'] - 10) < 0.5
        assert abs(params['scale'] - 2) < 0.5
    
    def test_exponential_fit(self):
        """Test exponential distribution fitting"""
        np.random.seed(42)
        data = np.random.exponential(2, size=500)
        
        dist = ExponentialDistribution()
        dist.fit(data, method='mle')
        
        assert dist.fitted
        params = dist.params
        assert 'scale' in params
        assert abs(params['scale'] - 2) < 0.5
    
    def test_get_distribution(self):
        """Test get_distribution function"""
        dist = get_distribution('normal')
        assert dist is not None
        assert dist.info.name == 'normal'
    
    def test_pdf_cdf(self):
        """Test PDF and CDF calculations"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=500)
        
        dist = get_distribution('normal')
        dist.fit(data)
        
        # Test PDF
        pdf_vals = dist.pdf(np.array([0, 1, -1]))
        assert all(pdf_vals > 0)
        
        # Test CDF
        cdf_vals = dist.cdf(np.array([0, 1, -1]))
        assert all((cdf_vals >= 0) & (cdf_vals <= 1))
    
    def test_sampling(self):
        """Test random sampling"""
        dist = get_distribution('normal')
        dist.params = {'loc': 0, 'scale': 1}
        
        samples = dist.rvs(size=100, random_state=42)
        assert len(samples) == 100
        assert abs(np.mean(samples)) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
