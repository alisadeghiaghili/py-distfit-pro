"""
Comprehensive Tests for Basic Distributions (Part 1/10)
=======================================================

Tests for 5 fundamental distributions:
1. Normal Distribution
2. Lognormal Distribution
3. Exponential Distribution
4. Uniform Distribution
5. Weibull Distribution

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    NormalDistribution,
    LognormalDistribution,
    ExponentialDistribution,
    UniformDistribution,
    WeibullDistribution,
    get_distribution
)


# ============================================================================
# TEST: NORMAL DISTRIBUTION
# ============================================================================

class TestNormalDistribution:
    """Comprehensive tests for Normal Distribution"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test Normal distribution initialization"""
        dist = NormalDistribution()
        assert dist.info.name == "normal"
        assert not dist.fitted
        assert dist.params is None
    
    def test_fit_mle(self, normal_data):
        """Test MLE fitting"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params
        assert 'scale' in dist.params
        assert 9.5 < dist.params['loc'] < 10.5
        assert 1.8 < dist.params['scale'] < 2.2
    
    def test_fit_moments(self, normal_data):
        """Test Method of Moments fitting"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='moments')
        
        assert dist.fitted
        assert abs(dist.params['loc'] - np.mean(normal_data)) < 0.01
        assert abs(dist.params['scale'] - np.std(normal_data, ddof=1)) < 0.01
    
    def test_pdf(self, normal_data):
        """Test PDF calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        x = np.array([8, 10, 12])
        pdf_values = dist.pdf(x)
        
        assert len(pdf_values) == len(x)
        assert np.all(pdf_values > 0)
        assert pdf_values[1] > pdf_values[0]  # PDF higher at mean
        assert pdf_values[1] > pdf_values[2]
    
    def test_cdf(self, normal_data):
        """Test CDF calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        x = np.array([8, 10, 12])
        cdf_values = dist.cdf(x)
        
        assert len(cdf_values) == len(x)
        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)
        assert np.all(np.diff(cdf_values) > 0)  # Monotonically increasing
    
    def test_ppf(self, normal_data):
        """Test PPF (inverse CDF)"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        q = np.array([0.25, 0.5, 0.75])
        quantiles = dist.ppf(q)
        
        assert len(quantiles) == len(q)
        assert np.all(np.diff(quantiles) > 0)
        assert abs(quantiles[1] - dist.params['loc']) < 0.1  # Median ≈ mean
    
    def test_statistics(self, normal_data):
        """Test statistical measures"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        mean = dist.mean()
        var = dist.var()
        std = dist.std()
        median = dist.median()
        mode = dist.mode()
        
        assert 9.5 < mean < 10.5
        assert 3.5 < var < 4.5
        assert abs(std - np.sqrt(var)) < 0.01
        assert abs(median - mean) < 0.1  # Symmetric
        assert abs(mode - mean) < 0.1
    
    def test_shape_statistics(self, normal_data):
        """Test skewness and kurtosis"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        skew = dist.skewness()
        kurt = dist.kurtosis()
        
        assert abs(skew) < 0.2  # Nearly symmetric
        assert abs(kurt) < 0.5  # Nearly mesokurtic
    
    def test_rvs(self, normal_data):
        """Test random sample generation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        
        assert len(samples) == 100
        assert 8 < np.mean(samples) < 12
        assert 1 < np.std(samples) < 3
    
    def test_summary(self, normal_data):
        """Test summary output"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        summary = dist.summary()
        
        assert "Normal" in summary
        assert "μ (mean)" in summary or "loc" in summary
        assert "σ (std)" in summary or "scale" in summary
    
    def test_explain(self, normal_data):
        """Test explanation output"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        explanation = dist.explain()
        
        assert "Normal" in explanation
        assert len(explanation) > 100


# ============================================================================
# TEST: LOGNORMAL DISTRIBUTION
# ============================================================================

class TestLognormalDistribution:
    """Comprehensive tests for Lognormal Distribution"""
    
    @pytest.fixture
    def lognormal_data(self):
        """Generate lognormal distributed data"""
        np.random.seed(42)
        return np.random.lognormal(mean=2, sigma=0.5, size=1000)
    
    def test_initialization(self):
        """Test Lognormal distribution initialization"""
        dist = LognormalDistribution()
        assert dist.info.name == "lognormal"
        assert not dist.fitted
    
    def test_fit_mle(self, lognormal_data):
        """Test MLE fitting"""
        dist = LognormalDistribution()
        dist.fit(lognormal_data, method='mle')
        
        assert dist.fitted
        assert 's' in dist.params
        assert 'scale' in dist.params
        assert dist.params['s'] > 0
        assert dist.params['scale'] > 0
    
    def test_positive_only(self):
        """Test that lognormal handles positive-only data"""
        dist = LognormalDistribution()
        data_with_negatives = np.array([-1, 0, 1, 2, 3, 4, 5])
        dist.fit(data_with_negatives, method='mle')
        
        # Should filter out non-positive values
        assert dist.fitted
    
    def test_pdf(self, lognormal_data):
        """Test PDF calculation"""
        dist = LognormalDistribution()
        dist.fit(lognormal_data, method='mle')
        
        x = np.array([1, 5, 10])
        pdf_values = dist.pdf(x)
        
        assert len(pdf_values) == len(x)
        assert np.all(pdf_values >= 0)
    
    def test_statistics(self, lognormal_data):
        """Test statistical measures"""
        dist = LognormalDistribution()
        dist.fit(lognormal_data, method='mle')
        
        mean = dist.mean()
        var = dist.var()
        median = dist.median()
        
        assert mean > 0
        assert var > 0
        assert median > 0
        assert median < mean  # Right-skewed
    
    def test_skewness(self, lognormal_data):
        """Test positive skewness"""
        dist = LognormalDistribution()
        dist.fit(lognormal_data, method='mle')
        
        skew = dist.skewness()
        assert skew > 0  # Right-skewed


# ============================================================================
# TEST: EXPONENTIAL DISTRIBUTION
# ============================================================================

class TestExponentialDistribution:
    """Comprehensive tests for Exponential Distribution"""
    
    @pytest.fixture
    def exponential_data(self):
        """Generate exponential distributed data"""
        np.random.seed(42)
        return np.random.exponential(scale=2, size=1000)
    
    def test_initialization(self):
        """Test Exponential distribution initialization"""
        dist = ExponentialDistribution()
        assert dist.info.name == "exponential"
        assert dist._mode_at_zero
    
    def test_fit_mle(self, exponential_data):
        """Test MLE fitting"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        assert dist.fitted
        assert 'scale' in dist.params
        assert 1.8 < dist.params['scale'] < 2.2
    
    def test_memoryless_property(self, exponential_data):
        """Test memoryless property"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        # P(X > s+t | X > s) = P(X > t)
        s, t = 1.0, 2.0
        prob1 = dist.sf(np.array([s + t]))[0] / dist.sf(np.array([s]))[0]
        prob2 = dist.sf(np.array([t]))[0]
        
        assert abs(prob1 - prob2) < 0.01
    
    def test_mode_at_zero(self, exponential_data):
        """Test mode is at zero"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        mode = dist.mode()
        assert mode == 0.0
    
    def test_mean_equals_scale(self, exponential_data):
        """Test mean equals scale parameter"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        mean = dist.mean()
        scale = dist.params['scale']
        
        assert abs(mean - scale) < 0.01
    
    def test_constant_hazard_rate(self, exponential_data):
        """Test constant hazard rate"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        # Hazard rate should be constant = 1/scale
        t_values = [0.5, 1.0, 2.0, 5.0]
        hazards = [dist.hazard_rate(t) for t in t_values]
        
        expected_hazard = 1.0 / dist.params['scale']
        for h in hazards:
            assert abs(h - expected_hazard) < 0.01


# ============================================================================
# TEST: UNIFORM DISTRIBUTION
# ============================================================================

class TestUniformDistribution:
    """Comprehensive tests for Uniform Distribution"""
    
    @pytest.fixture
    def uniform_data(self):
        """Generate uniform distributed data"""
        np.random.seed(42)
        return np.random.uniform(low=5, high=15, size=1000)
    
    def test_initialization(self):
        """Test Uniform distribution initialization"""
        dist = UniformDistribution()
        assert dist.info.name == "uniform"
    
    def test_fit_mle(self, uniform_data):
        """Test MLE fitting"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params
        assert 'scale' in dist.params
        
        # loc should be near min
        assert abs(dist.params['loc'] - np.min(uniform_data)) < 0.1
        # scale should be near range
        expected_scale = np.max(uniform_data) - np.min(uniform_data)
        assert abs(dist.params['scale'] - expected_scale) < 0.1
    
    def test_constant_pdf(self, uniform_data):
        """Test constant PDF within support"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        a = dist.params['loc']
        b = a + dist.params['scale']
        
        # PDF should be constant within [a, b]
        x = np.linspace(a + 0.1, b - 0.1, 10)
        pdf_values = dist.pdf(x)
        
        assert np.allclose(pdf_values, pdf_values[0], rtol=0.01)
    
    def test_mean_midpoint(self, uniform_data):
        """Test mean is at midpoint"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        a = dist.params['loc']
        b = a + dist.params['scale']
        expected_mean = (a + b) / 2
        
        assert abs(dist.mean() - expected_mean) < 0.01
    
    def test_variance_formula(self, uniform_data):
        """Test variance formula"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        width = dist.params['scale']
        expected_var = (width ** 2) / 12
        
        assert abs(dist.var() - expected_var) < 0.01
    
    def test_zero_skewness(self, uniform_data):
        """Test zero skewness (symmetric)"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        skew = dist.skewness()
        assert abs(skew) < 0.01


# ============================================================================
# TEST: WEIBULL DISTRIBUTION
# ============================================================================

class TestWeibullDistribution:
    """Comprehensive tests for Weibull Distribution"""
    
    @pytest.fixture
    def weibull_data(self):
        """Generate Weibull distributed data"""
        np.random.seed(42)
        return np.random.weibull(a=2, size=1000) * 5  # shape=2, scale=5
    
    def test_initialization(self):
        """Test Weibull distribution initialization"""
        dist = WeibullDistribution()
        assert dist.info.name == "weibull"
    
    def test_fit_mle(self, weibull_data):
        """Test MLE fitting"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        assert dist.fitted
        assert 'c' in dist.params  # shape
        assert 'scale' in dist.params
        assert dist.params['c'] > 0
        assert dist.params['scale'] > 0
    
    def test_shape_parameter_effects(self):
        """Test shape parameter effects on distribution"""
        np.random.seed(42)
        
        # k < 1: decreasing hazard (infant mortality)
        data1 = np.random.weibull(a=0.5, size=500) * 5
        dist1 = WeibullDistribution()
        dist1.fit(data1, method='mle')
        assert dist1.params['c'] < 1.2
        
        # k ≈ 1: constant hazard (exponential-like)
        data2 = np.random.weibull(a=1.0, size=500) * 5
        dist2 = WeibullDistribution()
        dist2.fit(data2, method='mle')
        assert 0.8 < dist2.params['c'] < 1.3
        
        # k > 1: increasing hazard (wear-out)
        data3 = np.random.weibull(a=2.5, size=500) * 5
        dist3 = WeibullDistribution()
        dist3.fit(data3, method='mle')
        assert dist3.params['c'] > 1.8
    
    def test_reliability_analysis(self, weibull_data):
        """Test reliability functions"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        t = 5.0
        reliability = dist.reliability(t)
        hazard = dist.hazard_rate(t)
        
        assert 0 <= reliability <= 1
        assert hazard > 0
    
    def test_positive_only(self):
        """Test Weibull handles positive-only data"""
        dist = WeibullDistribution()
        data_with_negatives = np.array([-1, 0, 1, 2, 3, 4, 5])
        dist.fit(data_with_negatives, method='mle')
        
        assert dist.fitted
        assert dist.params['scale'] > 0
    
    def test_pdf_shape(self, weibull_data):
        """Test PDF shape characteristics"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        x = np.linspace(0.1, 10, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
        assert pdf_values[0] < pdf_values[10]  # Initially increasing
    
    def test_cdf_bounds(self, weibull_data):
        """Test CDF is bounded [0, 1]"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        x = np.linspace(0, 100, 50)
        cdf_values = dist.cdf(x)
        
        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)
        assert cdf_values[-1] > 0.99  # Approaches 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestBasicDistributionsIntegration:
    """Integration tests for basic distributions"""
    
    def test_get_distribution_factory(self):
        """Test get_distribution factory function"""
        for name in ['normal', 'lognormal', 'exponential', 'uniform', 'weibull']:
            dist = get_distribution(name)
            assert dist.info.name == name
    
    def test_fit_and_sample_roundtrip(self):
        """Test fit -> sample -> fit consistency"""
        np.random.seed(42)
        
        # Original data
        original_data = np.random.normal(loc=10, scale=2, size=1000)
        
        # Fit
        dist1 = NormalDistribution()
        dist1.fit(original_data, method='mle')
        params1 = dist1.params.copy()
        
        # Generate samples
        samples = dist1.rvs(size=1000, random_state=42)
        
        # Fit again
        dist2 = NormalDistribution()
        dist2.fit(samples, method='mle')
        params2 = dist2.params
        
        # Parameters should be similar
        assert abs(params1['loc'] - params2['loc']) < 0.5
        assert abs(params1['scale'] - params2['scale']) < 0.3
    
    def test_cdf_ppf_inverse(self):
        """Test CDF and PPF are inverses"""
        np.random.seed(42)
        data = np.random.normal(loc=10, scale=2, size=500)
        
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        x_original = np.array([8, 10, 12])
        probabilities = dist.cdf(x_original)
        x_reconstructed = dist.ppf(probabilities)
        
        assert np.allclose(x_original, x_reconstructed, rtol=0.01)
    
    def test_all_methods_consistency(self):
        """Test all fitting methods give reasonable results"""
        np.random.seed(42)
        data = np.random.normal(loc=10, scale=2, size=500)
        
        dist_mle = NormalDistribution()
        dist_mle.fit(data, method='mle')
        
        dist_mom = NormalDistribution()
        dist_mom.fit(data, method='moments')
        
        # MLE and MoM should give similar results for Normal
        assert abs(dist_mle.params['loc'] - dist_mom.params['loc']) < 0.1
        assert abs(dist_mle.params['scale'] - dist_mom.params['scale']) < 0.1
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty data
        with pytest.raises(Exception):
            dist = NormalDistribution()
            dist.fit(np.array([]))
        
        # Single value
        dist = UniformDistribution()
        dist.fit(np.array([5.0]))
        assert dist.fitted
        
        # Data with NaN
        dist = NormalDistribution()
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        dist.fit(data_with_nan)
        assert dist.fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
