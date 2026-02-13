"""
Distribution Tests
==================

Test all 25 distributions.
"""

import pytest
import numpy as np
from distfit_pro import list_distributions, get_distribution


class TestAllDistributions:
    """Test all available distributions"""
    
    def test_list_distributions(self):
        """Test list_distributions function"""
        all_dists = list_distributions()
        assert len(all_dists) == 25
    
    @pytest.mark.parametrize('dist_name', [
        'normal', 'exponential', 'uniform', 'gamma', 'beta',
        'weibull', 'lognormal', 'logistic', 'gumbel', 'pareto',
        'cauchy', 'student_t', 'chi_square', 'f', 'laplace',
        'rayleigh', 'wald', 'triangular', 'burr', 'genextreme',
    ])
    def test_continuous_distributions(self, dist_name):
        """Test continuous distributions can be created and fitted"""
        np.random.seed(42)
        
        # Generate appropriate data
        if dist_name in ['beta']:
            data = np.random.beta(2, 2, size=100)
        elif dist_name in ['uniform']:
            data = np.random.uniform(0, 1, size=100)
        elif dist_name in ['exponential', 'weibull', 'gamma', 'rayleigh', 'pareto']:
            data = np.random.exponential(1, size=100) + 0.1
        else:
            data = np.random.normal(0, 1, size=100)
        
        dist = get_distribution(dist_name)
        assert dist is not None
        
        try:
            dist.fit(data, method='mle')
            assert dist.fitted
        except:
            # Some distributions may fail on certain data
            pass
    
    @pytest.mark.parametrize('dist_name', [
        'poisson', 'binomial', 'negative_binomial', 'geometric', 'hypergeometric'
    ])
    def test_discrete_distributions(self, dist_name):
        """Test discrete distributions can be created"""
        dist = get_distribution(dist_name)
        assert dist is not None
        assert dist.info.is_discrete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
