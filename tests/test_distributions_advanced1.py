"""
Comprehensive Tests for Advanced Distributions Part 1 (Part 2/10)
=================================================================

Tests for 5 advanced distributions:
1. Gamma Distribution
2. Beta Distribution
3. Triangular Distribution
4. Logistic Distribution
5. Gumbel Distribution

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    GammaDistribution,
    BetaDistribution,
    TriangularDistribution,
    LogisticDistribution,
    GumbelDistribution,
    get_distribution
)


# ============================================================================
# TEST: GAMMA DISTRIBUTION
# ============================================================================

class TestGammaDistribution:
    """Comprehensive tests for Gamma Distribution"""
    
    @pytest.fixture
    def gamma_data(self):
        """Generate gamma distributed data"""
        np.random.seed(42)
        return np.random.gamma(shape=2, scale=3, size=1000)
    
    def test_initialization(self):
        """Test Gamma distribution initialization"""
        dist = GammaDistribution()
        assert dist.info.name == "gamma"
        assert not dist.fitted
    
    def test_fit_mle(self, gamma_data):
        """Test MLE fitting"""
        dist = GammaDistribution()
        dist.fit(gamma_data, method='mle')
        
        assert dist.fitted
        assert 'a' in dist.params  # shape (alpha)
        assert 'scale' in dist.params  # scale (theta)
        assert 1.5 < dist.params['a'] < 2.5
        assert 2.5 < dist.params['scale'] < 3.5
    
    def test_fit_moments(self, gamma_data):
        """Test Method of Moments fitting"""
        dist = GammaDistribution()
        dist.fit(gamma_data, method='moments')
        
        assert dist.fitted
        m = np.mean(gamma_data)
        v = np.var(gamma_data, ddof=1)
        
        # alpha = m^2 / v, theta = v / m
        expected_alpha = m * m / v
        expected_theta = v / m
        
        assert abs(dist.params['a'] - expected_alpha) < 0.1
        assert abs(dist.params['scale'] - expected_theta) < 0.1
    
    def test_exponential_special_case(self):
        """Test that Gamma(1, theta) = Exponential(theta)"""
        np.random.seed(42)
        # Gamma with shape=1 is exponential
        data = np.random.gamma(shape=1, scale=2, size=1000)
        
        dist = GammaDistribution()
        dist.fit(data, method='mle')
        
        assert 0.8 < dist.params['a'] < 1.3  # Shape near 1
    
    def test_normal_approximation_large_alpha(self):
        """Test that large alpha approaches normal"""
        np.random.seed(42)
        # Large alpha -> normal-like
        data = np.random.gamma(shape=30, scale=1, size=1000)
        
        dist = GammaDistribution()
        dist.fit(data, method='mle')
        
        # Skewness should be small
        skew = dist.skewness()
        assert abs(skew) < 0.5  # Less skewed with large alpha
    
    def test_positive_only(self):
        """Test Gamma handles positive-only data"""
        dist = GammaDistribution()
        data_with_negatives = np.array([-1, 0, 1, 2, 3, 4, 5])
        dist.fit(data_with_negatives, method='mle')
        
        assert dist.fitted
    
    def test_statistics(self, gamma_data):
        """Test statistical measures"""
        dist = GammaDistribution()
        dist.fit(gamma_data, method='mle')
        
        mean = dist.mean()
        var = dist.var()
        
        # E[X] = alpha * theta
        expected_mean = dist.params['a'] * dist.params['scale']
        # Var[X] = alpha * theta^2
        expected_var = dist.params['a'] * (dist.params['scale'] ** 2)
        
        assert abs(mean - expected_mean) < 0.01
        assert abs(var - expected_var) < 0.01
    
    def test_pdf_shape(self, gamma_data):
        """Test PDF shape characteristics"""
        dist = GammaDistribution()
        dist.fit(gamma_data, method='mle')
        
        x = np.linspace(0.1, 20, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
        # Find mode
        mode_idx = np.argmax(pdf_values)
        # PDF should decay after mode
        assert pdf_values[mode_idx] > pdf_values[-1]
    
    def test_skewness_formula(self, gamma_data):
        """Test skewness formula"""
        dist = GammaDistribution()
        dist.fit(gamma_data, method='mle')
        
        skew = dist.skewness()
        # Skewness = 2 / sqrt(alpha)
        expected_skew = 2.0 / np.sqrt(dist.params['a'])
        
        assert abs(skew - expected_skew) < 0.2


# ============================================================================
# TEST: BETA DISTRIBUTION
# ============================================================================

class TestBetaDistribution:
    """Comprehensive tests for Beta Distribution"""
    
    @pytest.fixture
    def beta_data(self):
        """Generate beta distributed data"""
        np.random.seed(42)
        return np.random.beta(a=2, b=5, size=1000)
    
    def test_initialization(self):
        """Test Beta distribution initialization"""
        dist = BetaDistribution()
        assert dist.info.name == "beta"
        assert not dist.fitted
    
    def test_fit_mle(self, beta_data):
        """Test MLE fitting"""
        dist = BetaDistribution()
        dist.fit(beta_data, method='mle')
        
        assert dist.fitted
        assert 'a' in dist.params  # alpha
        assert 'b' in dist.params  # beta
        assert dist.params['a'] > 0
        assert dist.params['b'] > 0
    
    def test_fit_moments(self, beta_data):
        """Test Method of Moments fitting"""
        dist = BetaDistribution()
        dist.fit(beta_data, method='moments')
        
        assert dist.fitted
        m = np.mean(beta_data)
        v = np.var(beta_data, ddof=1)
        
        # Method of moments formulas
        common = m * (1 - m) / v - 1
        expected_a = m * common
        expected_b = (1 - m) * common
        
        assert abs(dist.params['a'] - expected_a) < 0.5
        assert abs(dist.params['b'] - expected_b) < 0.5
    
    def test_bounded_01(self, beta_data):
        """Test Beta is bounded in [0, 1]"""
        dist = BetaDistribution()
        dist.fit(beta_data, method='mle')
        
        # All samples should be in [0, 1]
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)
    
    def test_uniform_special_case(self):
        """Test that Beta(1, 1) = Uniform(0, 1)"""
        np.random.seed(42)
        data = np.random.beta(a=1, b=1, size=1000)
        
        dist = BetaDistribution()
        dist.fit(data, method='mle')
        
        # Parameters should be close to 1
        assert 0.8 < dist.params['a'] < 1.3
        assert 0.8 < dist.params['b'] < 1.3
    
    def test_symmetric_case(self):
        """Test symmetric Beta(a, a)"""
        np.random.seed(42)
        data = np.random.beta(a=5, b=5, size=1000)
        
        dist = BetaDistribution()
        dist.fit(data, method='mle')
        
        # Should be symmetric around 0.5
        mean = dist.mean()
        median = dist.median()
        
        assert abs(mean - 0.5) < 0.05
        assert abs(median - 0.5) < 0.05
    
    def test_skewness_direction(self):
        """Test skewness direction"""
        np.random.seed(42)
        
        # a > b: left-skewed
        data1 = np.random.beta(a=5, b=2, size=500)
        dist1 = BetaDistribution()
        dist1.fit(data1, method='mle')
        assert dist1.skewness() < 0
        
        # a < b: right-skewed
        data2 = np.random.beta(a=2, b=5, size=500)
        dist2 = BetaDistribution()
        dist2.fit(data2, method='mle')
        assert dist2.skewness() > 0
    
    def test_mean_formula(self, beta_data):
        """Test mean formula"""
        dist = BetaDistribution()
        dist.fit(beta_data, method='mle')
        
        mean = dist.mean()
        # E[X] = a / (a + b)
        expected_mean = dist.params['a'] / (dist.params['a'] + dist.params['b'])
        
        assert abs(mean - expected_mean) < 0.01
    
    def test_bayesian_prior(self):
        """Test Beta as Bayesian prior"""
        # Beta(1, 1) is uniform prior
        dist = BetaDistribution()
        uniform_data = np.random.uniform(0, 1, size=500)
        dist.fit(uniform_data, method='mle')
        
        assert dist.fitted


# ============================================================================
# TEST: TRIANGULAR DISTRIBUTION
# ============================================================================

class TestTriangularDistribution:
    """Comprehensive tests for Triangular Distribution"""
    
    @pytest.fixture
    def triangular_data(self):
        """Generate triangular distributed data"""
        np.random.seed(42)
        # mode at 0.3, from 0 to 1
        return np.random.triangular(left=0, mode=0.3, right=1, size=1000)
    
    def test_initialization(self):
        """Test Triangular distribution initialization"""
        dist = TriangularDistribution()
        assert dist.info.name == "triangular"
    
    def test_fit_mle(self, triangular_data):
        """Test MLE fitting"""
        dist = TriangularDistribution()
        dist.fit(triangular_data, method='mle')
        
        assert dist.fitted
        assert 'c' in dist.params  # mode position
        assert 'loc' in dist.params  # min
        assert 'scale' in dist.params  # width
        
        assert 0 <= dist.params['c'] <= 1
    
    def test_symmetric_case(self):
        """Test symmetric triangular (mode at center)"""
        np.random.seed(42)
        data = np.random.triangular(left=0, mode=0.5, right=1, size=1000)
        
        dist = TriangularDistribution()
        dist.fit(data, method='mle')
        
        # Mode position should be near 0.5
        c = dist.params['c']
        assert 0.4 < c < 0.6
    
    def test_mean_formula(self, triangular_data):
        """Test mean formula"""
        dist = TriangularDistribution()
        dist.fit(triangular_data, method='mle')
        
        a = dist.params['loc']
        b = a + dist.params['scale']
        mode = a + dist.params['c'] * dist.params['scale']
        
        # E[X] = (a + b + mode) / 3
        expected_mean = (a + b + mode) / 3
        actual_mean = dist.mean()
        
        assert abs(actual_mean - expected_mean) < 0.1
    
    def test_bounded_support(self, triangular_data):
        """Test triangular is bounded"""
        dist = TriangularDistribution()
        dist.fit(triangular_data, method='mle')
        
        a = dist.params['loc']
        b = a + dist.params['scale']
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= a - 0.01)  # Small tolerance
        assert np.all(samples <= b + 0.01)
    
    def test_pdf_shape(self, triangular_data):
        """Test PDF has triangular shape"""
        dist = TriangularDistribution()
        dist.fit(triangular_data, method='mle')
        
        a = dist.params['loc']
        b = a + dist.params['scale']
        mode = a + dist.params['c'] * dist.params['scale']
        
        # PDF at mode should be highest
        pdf_at_mode = dist.pdf(np.array([mode]))[0]
        pdf_at_a = dist.pdf(np.array([a + 0.01]))[0]
        pdf_at_b = dist.pdf(np.array([b - 0.01]))[0]
        
        assert pdf_at_mode > pdf_at_a
        assert pdf_at_mode > pdf_at_b
    
    def test_expert_estimation_use_case(self):
        """Test triangular for expert estimation (PERT)"""
        # Simulate expert estimates: min=10, most likely=15, max=25
        np.random.seed(42)
        data = np.random.triangular(left=10, mode=15, right=25, size=500)
        
        dist = TriangularDistribution()
        dist.fit(data, method='mle')
        
        assert dist.fitted
        assert 9 < dist.params['loc'] < 11
        assert 14 < dist.params['scale'] < 16


# ============================================================================
# TEST: LOGISTIC DISTRIBUTION
# ============================================================================

class TestLogisticDistribution:
    """Comprehensive tests for Logistic Distribution"""
    
    @pytest.fixture
    def logistic_data(self):
        """Generate logistic distributed data"""
        np.random.seed(42)
        return np.random.logistic(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test Logistic distribution initialization"""
        dist = LogisticDistribution()
        assert dist.info.name == "logistic"
    
    def test_fit_mle(self, logistic_data):
        """Test MLE fitting"""
        dist = LogisticDistribution()
        dist.fit(logistic_data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params  # mu (location)
        assert 'scale' in dist.params  # s (scale)
        assert 9 < dist.params['loc'] < 11
        assert 1.5 < dist.params['scale'] < 2.5
    
    def test_heavier_tails_than_normal(self):
        """Test logistic has heavier tails than normal"""
        np.random.seed(42)
        
        # Generate data
        data = np.random.logistic(loc=0, scale=1, size=1000)
        
        dist = LogisticDistribution()
        dist.fit(data, method='mle')
        
        # Kurtosis should be positive (heavier tails)
        kurt = dist.kurtosis()
        assert kurt > 0.5  # Logistic has kurtosis = 1.2
    
    def test_symmetric(self, logistic_data):
        """Test logistic is symmetric"""
        dist = LogisticDistribution()
        dist.fit(logistic_data, method='mle')
        
        skew = dist.skewness()
        assert abs(skew) < 0.2
        
        mean = dist.mean()
        median = dist.median()
        mode = dist.mode()
        
        # Mean = median = mode for symmetric
        assert abs(mean - median) < 0.1
        assert abs(mean - mode) < 0.1
    
    def test_cdf_sigmoid_shape(self, logistic_data):
        """Test CDF has sigmoid shape"""
        dist = LogisticDistribution()
        dist.fit(logistic_data, method='mle')
        
        x = np.linspace(-10, 30, 100)
        cdf_values = dist.cdf(x)
        
        # Should be sigmoid
        assert cdf_values[0] < 0.1
        assert cdf_values[-1] > 0.9
        assert 0.4 < cdf_values[50] < 0.6  # Near 0.5 at center
    
    def test_variance_formula(self, logistic_data):
        """Test variance formula"""
        dist = LogisticDistribution()
        dist.fit(logistic_data, method='mle')
        
        var = dist.var()
        # Var = (pi * s)^2 / 3
        expected_var = (np.pi * dist.params['scale']) ** 2 / 3
        
        assert abs(var - expected_var) < 0.1
    
    def test_logistic_regression_connection(self):
        """Test connection to logistic regression"""
        # The logistic distribution's CDF is the logistic function
        dist = LogisticDistribution()
        dist.params = {'loc': 0, 'scale': 1}
        dist.fitted = True
        
        # CDF(0) should be 0.5
        assert abs(dist.cdf(np.array([0]))[0] - 0.5) < 0.01


# ============================================================================
# TEST: GUMBEL DISTRIBUTION
# ============================================================================

class TestGumbelDistribution:
    """Comprehensive tests for Gumbel Distribution"""
    
    @pytest.fixture
    def gumbel_data(self):
        """Generate Gumbel distributed data"""
        np.random.seed(42)
        return np.random.gumbel(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test Gumbel distribution initialization"""
        dist = GumbelDistribution()
        assert dist.info.name == "gumbel"
    
    def test_fit_mle(self, gumbel_data):
        """Test MLE fitting"""
        dist = GumbelDistribution()
        dist.fit(gumbel_data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params  # mu (location)
        assert 'scale' in dist.params  # beta (scale)
        assert 9 < dist.params['loc'] < 11
        assert 1.5 < dist.params['scale'] < 2.5
    
    def test_fit_moments(self, gumbel_data):
        """Test Method of Moments fitting"""
        dist = GumbelDistribution()
        dist.fit(gumbel_data, method='moments')
        
        assert dist.fitted
        m = np.mean(gumbel_data)
        s = np.std(gumbel_data, ddof=1)
        
        # beta = s * sqrt(6) / pi
        # mu = m - gamma * beta (where gamma is Euler-Mascheroni constant)
        expected_beta = s * np.sqrt(6) / np.pi
        expected_mu = m - 0.5772 * expected_beta
        
        assert abs(dist.params['scale'] - expected_beta) < 0.2
        assert abs(dist.params['loc'] - expected_mu) < 0.5
    
    def test_extreme_value_property(self):
        """Test Gumbel for extreme value analysis"""
        np.random.seed(42)
        # Simulate annual maximum floods
        data = np.random.gumbel(loc=100, scale=20, size=50)
        
        dist = GumbelDistribution()
        dist.fit(data, method='mle')
        
        # 100-year flood (1% exceedance probability)
        flood_100yr = dist.ppf(0.99)
        assert flood_100yr > 100
    
    def test_positive_skewness(self, gumbel_data):
        """Test Gumbel is right-skewed"""
        dist = GumbelDistribution()
        dist.fit(gumbel_data, method='mle')
        
        skew = dist.skewness()
        # Gumbel has fixed skewness ≈ 1.14
        assert 0.8 < skew < 1.5
    
    def test_mode_less_than_median(self, gumbel_data):
        """Test mode < median < mean for Gumbel"""
        dist = GumbelDistribution()
        dist.fit(gumbel_data, method='mle')
        
        mode = dist.mode()
        median = dist.median()
        mean = dist.mean()
        
        # Right-skewed: mode < median < mean
        assert mode < median
        assert median < mean
    
    def test_pdf_shape(self, gumbel_data):
        """Test PDF shape"""
        dist = GumbelDistribution()
        dist.fit(gumbel_data, method='mle')
        
        x = np.linspace(0, 20, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
        # Should have a single peak
        max_idx = np.argmax(pdf_values)
        assert 10 < max_idx < 90  # Peak not at boundaries
    
    def test_return_period(self, gumbel_data):
        """Test return period calculations"""
        dist = GumbelDistribution()
        dist.fit(gumbel_data, method='mle')
        
        # 10-year return period (90% exceedance)
        value_10yr = dist.ppf(0.9)
        # 100-year return period (99% exceedance)
        value_100yr = dist.ppf(0.99)
        
        assert value_100yr > value_10yr


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAdvancedDistributions1Integration:
    """Integration tests for advanced distributions part 1"""
    
    def test_all_distributions_fit(self):
        """Test all distributions can be fitted"""
        np.random.seed(42)
        
        test_cases = [
            ('gamma', np.random.gamma(2, 3, 500)),
            ('beta', np.random.beta(2, 5, 500)),
            ('triangular', np.random.triangular(0, 0.3, 1, 500)),
            ('logistic', np.random.logistic(10, 2, 500)),
            ('gumbel', np.random.gumbel(10, 2, 500))
        ]
        
        for name, data in test_cases:
            dist = get_distribution(name)
            dist.fit(data, method='mle')
            assert dist.fitted, f"{name} failed to fit"
    
    def test_pdf_integrates_to_one(self):
        """Test PDF integrates to approximately 1"""
        np.random.seed(42)
        data = np.random.gamma(2, 3, 500)
        
        dist = GammaDistribution()
        dist.fit(data, method='mle')
        
        # Numerical integration
        x = np.linspace(0.01, 50, 1000)
        pdf_values = dist.pdf(x)
        integral = np.trapz(pdf_values, x)
        
        assert 0.95 < integral < 1.05
    
    def test_cdf_monotonic(self):
        """Test CDF is monotonically increasing"""
        np.random.seed(42)
        data = np.random.beta(2, 5, 500)
        
        dist = BetaDistribution()
        dist.fit(data, method='mle')
        
        x = np.linspace(0.01, 0.99, 100)
        cdf_values = dist.cdf(x)
        
        # Check monotonicity
        assert np.all(np.diff(cdf_values) >= 0)
    
    def test_ppf_cdf_inverse(self):
        """Test PPF and CDF are inverses"""
        np.random.seed(42)
        data = np.random.logistic(10, 2, 500)
        
        dist = LogisticDistribution()
        dist.fit(data, method='mle')
        
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        x = dist.ppf(q)
        q_reconstructed = dist.cdf(x)
        
        assert np.allclose(q, q_reconstructed, rtol=0.01)
    
    def test_summary_and_explain(self):
        """Test summary and explain methods"""
        np.random.seed(42)
        data = np.random.gamma(2, 3, 500)
        
        dist = GammaDistribution()
        dist.fit(data, method='mle')
        
        summary = dist.summary()
        explain = dist.explain()
        
        assert len(summary) > 200
        assert len(explain) > 100
        assert "Gamma" in summary
        assert "Gamma" in explain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
