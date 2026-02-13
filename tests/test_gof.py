"""
GOF Tests
=========

Test goodness-of-fit tests.
"""

import pytest
import numpy as np
from distfit_pro import (
    get_distribution,
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    ChiSquareTest,
    CramerVonMisesTest,
)


class TestGOFTests:
    """Test all GOF tests"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal data"""
        np.random.seed(42)
        return np.random.normal(0, 1, size=500)
    
    @pytest.fixture
    def fitted_dist(self, normal_data):
        """Fitted distribution"""
        dist = get_distribution('normal')
        dist.fit(normal_data)
        return dist
    
    def test_ks_test(self, normal_data, fitted_dist):
        """Test Kolmogorov-Smirnov test"""
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(normal_data, fitted_dist)
        
        assert result is not None
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        assert 0 <= result.p_value <= 1
    
    def test_ad_test(self, normal_data, fitted_dist):
        """Test Anderson-Darling test"""
        ad_test = AndersonDarlingTest()
        result = ad_test.test(normal_data, fitted_dist)
        
        assert result is not None
        assert hasattr(result, 'statistic')
    
    def test_chi_square_test(self, normal_data, fitted_dist):
        """Test Chi-Square test"""
        chi_test = ChiSquareTest()
        result = chi_test.test(normal_data, fitted_dist)
        
        assert result is not None
        assert hasattr(result, 'statistic')
    
    def test_cvm_test(self, normal_data, fitted_dist):
        """Test Cramer-von Mises test"""
        cvm_test = CramerVonMisesTest()
        result = cvm_test.test(normal_data, fitted_dist)
        
        assert result is not None
        assert hasattr(result, 'statistic')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
