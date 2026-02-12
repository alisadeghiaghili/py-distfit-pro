"""
Comprehensive Tests for Advanced Distributions Part 3 (Part 4/10)
=================================================================

Tests for final 5 continuous distributions:
1. F Distribution (ANOVA)
2. Rayleigh Distribution (Signal Processing)
3. Laplace Distribution (Double Exponential)
4. Inverse Gamma Distribution (Bayesian)
5. Log-Logistic Distribution (Survival Analysis)

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    FDistribution,
    RayleighDistribution,
    LaplaceDistribution,
    InverseGammaDistribution,
    LogLogisticDistribution,
    get_distribution
)


# ============================================================================
# TEST: F DISTRIBUTION
# ============================================================================

class TestFDistribution:
    """Comprehensive tests for F Distribution"""
    
    @pytest.fixture
    def f_data(self):
        """Generate F distributed data"""
        np.random.seed(42)
        return np.random.f(dfn=5, dfd=10, size=1000)
    
    def test_initialization(self):
        """Test F distribution initialization"""
        dist = FDistribution()
        assert dist.info.name == "f"
        assert not dist.fitted
    
    def test_fit_mle(self, f_data):
        """Test MLE fitting"""
        dist = FDistribution()
        dist.fit(f_data, method='mle')
        
        assert dist.fitted
        assert 'dfn' in dist.params  # d1 (numerator df)
        assert 'dfd' in dist.params  # d2 (denominator df)
        assert dist.params['dfn'] > 0
        assert dist.params['dfd'] > 0
    
    def test_positive_support(self, f_data):
        """Test F distribution support is [0, infinity)"""
        dist = FDistribution()
        dist.fit(f_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
    
    def test_right_skewed(self, f_data):
        """Test F distribution is right-skewed"""
        dist = FDistribution()
        dist.fit(f_data, method='mle')
        
        mean = dist.mean()
        median = dist.median()
        
        # Right-skewed: median < mean
        assert median < mean
    
    def test_anova_application(self):
        """Test F distribution for ANOVA"""
        dist = FDistribution()
        dist.params = {'dfn': 3, 'dfd': 20}  # 4 groups, 24 total observations
        dist.fitted = True
        
        # Critical value at alpha=0.05
        f_critical = dist.ppf(0.95)
        assert 3.0 < f_critical < 3.5  # Known range for F(3, 20)
    
    def test_variance_ratio(self):
        """Test F as ratio of chi-squared variables"""
        np.random.seed(42)
        
        # F = (Chi1/df1) / (Chi2/df2)
        df1, df2 = 5, 10
        chi1 = np.random.chisquare(df1, size=1000)
        chi2 = np.random.chisquare(df2, size=1000)
        f_ratio = (chi1 / df1) / (chi2 / df2)
        
        dist = FDistribution()
        dist.fit(f_ratio, method='mle')
        
        assert dist.fitted
    
    def test_mean_formula(self, f_data):
        """Test mean formula"""
        dist = FDistribution()
        dist.fit(f_data, method='mle')
        
        dfd = dist.params['dfd']
        
        if dfd > 2:
            mean = dist.mean()
            # E[F] = d2 / (d2 - 2) for d2 > 2
            expected_mean = dfd / (dfd - 2)
            assert abs(mean - expected_mean) / expected_mean < 0.3
    
    def test_pdf_shape(self, f_data):
        """Test PDF shape characteristics"""
        dist = FDistribution()
        dist.fit(f_data, method='mle')
        
        x = np.linspace(0.01, 5, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
        # Should have a mode for typical parameters
        assert np.max(pdf_values) > 0


# ============================================================================
# TEST: RAYLEIGH DISTRIBUTION
# ============================================================================

class TestRayleighDistribution:
    """Comprehensive tests for Rayleigh Distribution"""
    
    @pytest.fixture
    def rayleigh_data(self):
        """Generate Rayleigh distributed data"""
        np.random.seed(42)
        return np.random.rayleigh(scale=2, size=1000)
    
    def test_initialization(self):
        """Test Rayleigh distribution initialization"""
        dist = RayleighDistribution()
        assert dist.info.name == "rayleigh"
    
    def test_fit_mle(self, rayleigh_data):
        """Test MLE fitting"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        assert dist.fitted
        assert 'scale' in dist.params  # sigma (scale)
        assert 1.8 < dist.params['scale'] < 2.2
    
    def test_mle_formula(self, rayleigh_data):
        """Test MLE formula: sigma = sqrt(sum(x^2) / 2n)"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        expected_scale = np.sqrt(np.mean(rayleigh_data ** 2) / 2)
        assert abs(dist.params['scale'] - expected_scale) < 0.1
    
    def test_positive_support(self, rayleigh_data):
        """Test Rayleigh support is [0, infinity)"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
    
    def test_mean_formula(self, rayleigh_data):
        """Test mean formula"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        mean = dist.mean()
        # E[X] = sigma * sqrt(pi/2)
        expected_mean = dist.params['scale'] * np.sqrt(np.pi / 2)
        
        assert abs(mean - expected_mean) / expected_mean < 0.01
    
    def test_variance_formula(self, rayleigh_data):
        """Test variance formula"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        var = dist.var()
        # Var[X] = sigma^2 * (4 - pi) / 2
        expected_var = (dist.params['scale'] ** 2) * (4 - np.pi) / 2
        
        assert abs(var - expected_var) / expected_var < 0.02
    
    def test_mode_formula(self, rayleigh_data):
        """Test mode formula"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        mode = dist.mode()
        # Mode = sigma
        assert abs(mode - dist.params['scale']) < 0.1
    
    def test_signal_processing_application(self):
        """Test Rayleigh for signal amplitude"""
        np.random.seed(42)
        # Simulate signal amplitude from Gaussian noise
        noise_i = np.random.normal(0, 1, 500)
        noise_q = np.random.normal(0, 1, 500)
        amplitude = np.sqrt(noise_i**2 + noise_q**2)
        
        dist = RayleighDistribution()
        dist.fit(amplitude, method='mle')
        
        assert dist.fitted
        # Scale should be close to 1 (noise std)
        assert 0.8 < dist.params['scale'] < 1.2
    
    def test_wind_speed_application(self, rayleigh_data):
        """Test Rayleigh for wind speed modeling"""
        dist = RayleighDistribution()
        dist.fit(rayleigh_data, method='mle')
        
        # Calculate probability of wind speed > threshold
        threshold = 3.0
        prob_exceeds = dist.sf(np.array([threshold]))[0]
        
        assert 0 < prob_exceeds < 1


# ============================================================================
# TEST: LAPLACE DISTRIBUTION
# ============================================================================

class TestLaplaceDistribution:
    """Comprehensive tests for Laplace Distribution"""
    
    @pytest.fixture
    def laplace_data(self):
        """Generate Laplace distributed data"""
        np.random.seed(42)
        return np.random.laplace(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test Laplace distribution initialization"""
        dist = LaplaceDistribution()
        assert dist.info.name == "laplace"
    
    def test_fit_mle(self, laplace_data):
        """Test MLE fitting"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params  # mu (location)
        assert 'scale' in dist.params  # b (scale)
        assert 9 < dist.params['loc'] < 11
        assert 1.5 < dist.params['scale'] < 2.5
    
    def test_mle_uses_median(self, laplace_data):
        """Test MLE location is median"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        # MLE for location is median
        assert abs(dist.params['loc'] - np.median(laplace_data)) < 0.01
    
    def test_mle_uses_mad(self, laplace_data):
        """Test MLE scale is mean absolute deviation"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        # MLE for scale is mean absolute deviation
        expected_scale = np.mean(np.abs(laplace_data - np.median(laplace_data)))
        assert abs(dist.params['scale'] - expected_scale) < 0.1
    
    def test_symmetric(self, laplace_data):
        """Test Laplace is symmetric"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        skew = dist.skewness()
        assert abs(skew) < 0.2
        
        mean = dist.mean()
        median = dist.median()
        assert abs(mean - median) < 0.2
    
    def test_heavier_tails_than_normal(self, laplace_data):
        """Test Laplace has heavier tails than normal"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        kurt = dist.kurtosis()
        # Laplace has kurtosis = 3 (heavier than normal's 0)
        assert kurt > 2
    
    def test_variance_formula(self, laplace_data):
        """Test variance formula"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        var = dist.var()
        # Var[X] = 2 * b^2
        expected_var = 2 * (dist.params['scale'] ** 2)
        
        assert abs(var - expected_var) / expected_var < 0.02
    
    def test_pdf_shape(self, laplace_data):
        """Test PDF has double exponential shape"""
        dist = LaplaceDistribution()
        dist.fit(laplace_data, method='mle')
        
        mu = dist.params['loc']
        
        # PDF should peak at mu
        x = np.linspace(mu - 5, mu + 5, 100)
        pdf_values = dist.pdf(x)
        
        peak_idx = np.argmax(pdf_values)
        assert abs(x[peak_idx] - mu) < 0.5
    
    def test_lasso_regression_connection(self):
        """Test connection to Lasso regression"""
        # Laplace prior in Bayesian framework = Lasso penalty
        dist = LaplaceDistribution()
        dist.params = {'loc': 0, 'scale': 1}
        dist.fitted = True
        
        # Log-likelihood at 0 should be highest
        logpdf_0 = dist.logpdf(np.array([0]))[0]
        logpdf_1 = dist.logpdf(np.array([1]))[0]
        
        assert logpdf_0 > logpdf_1
    
    def test_robust_to_outliers(self):
        """Test Laplace is robust to outliers"""
        np.random.seed(42)
        # Data with outliers
        clean_data = np.random.laplace(0, 1, 950)
        outliers = np.random.uniform(-20, 20, 50)
        data = np.concatenate([clean_data, outliers])
        
        dist = LaplaceDistribution()
        dist.fit(data, method='mle')
        
        # Location should still be near 0
        assert abs(dist.params['loc']) < 1


# ============================================================================
# TEST: INVERSE GAMMA DISTRIBUTION
# ============================================================================

class TestInverseGammaDistribution:
    """Comprehensive tests for Inverse Gamma Distribution"""
    
    @pytest.fixture
    def invgamma_data(self):
        """Generate Inverse Gamma distributed data"""
        np.random.seed(42)
        return stats.invgamma.rvs(a=3, scale=2, size=1000)
    
    def test_initialization(self):
        """Test Inverse Gamma distribution initialization"""
        dist = InverseGammaDistribution()
        assert dist.info.name == "invgamma"
    
    def test_fit_mle(self, invgamma_data):
        """Test MLE fitting"""
        dist = InverseGammaDistribution()
        dist.fit(invgamma_data, method='mle')
        
        assert dist.fitted
        assert 'a' in dist.params  # alpha (shape)
        assert 'scale' in dist.params  # beta (scale)
        assert dist.params['a'] > 0
        assert dist.params['scale'] > 0
    
    def test_positive_support(self, invgamma_data):
        """Test Inverse Gamma support is (0, infinity)"""
        dist = InverseGammaDistribution()
        dist.fit(invgamma_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples > 0)
    
    def test_mean_formula(self, invgamma_data):
        """Test mean formula"""
        dist = InverseGammaDistribution()
        dist.fit(invgamma_data, method='mle')
        
        alpha = dist.params['a']
        beta = dist.params['scale']
        
        if alpha > 1:
            mean = dist.mean()
            # E[X] = beta / (alpha - 1) for alpha > 1
            expected_mean = beta / (alpha - 1)
            assert abs(mean - expected_mean) / expected_mean < 0.2
    
    def test_variance_formula(self, invgamma_data):
        """Test variance formula"""
        dist = InverseGammaDistribution()
        dist.fit(invgamma_data, method='mle')
        
        alpha = dist.params['a']
        beta = dist.params['scale']
        
        if alpha > 2:
            var = dist.var()
            # Var[X] = beta^2 / ((alpha-1)^2 * (alpha-2))
            expected_var = (beta ** 2) / ((alpha - 1) ** 2 * (alpha - 2))
            assert abs(var - expected_var) / expected_var < 0.3
    
    def test_bayesian_prior_application(self):
        """Test Inverse Gamma as conjugate prior for variance"""
        # Inverse Gamma is conjugate prior for normal variance
        dist = InverseGammaDistribution()
        dist.params = {'a': 2, 'scale': 1}  # Weakly informative prior
        dist.fitted = True
        
        # Prior should put mass on positive values
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples > 0)
        assert np.mean(samples) > 0
    
    def test_right_skewed(self, invgamma_data):
        """Test Inverse Gamma is right-skewed"""
        dist = InverseGammaDistribution()
        dist.fit(invgamma_data, method='mle')
        
        if dist.params['a'] > 3:
            skew = dist.skewness()
            assert skew > 0
    
    def test_heavy_tails(self, invgamma_data):
        """Test Inverse Gamma has heavy tails"""
        dist = InverseGammaDistribution()
        dist.fit(invgamma_data, method='mle')
        
        # Should have some large values
        samples = dist.rvs(size=1000, random_state=42)
        assert np.max(samples) > np.median(samples) * 3


# ============================================================================
# TEST: LOG-LOGISTIC DISTRIBUTION
# ============================================================================

class TestLogLogisticDistribution:
    """Comprehensive tests for Log-Logistic Distribution"""
    
    @pytest.fixture
    def loglogistic_data(self):
        """Generate Log-Logistic distributed data"""
        np.random.seed(42)
        return stats.fisk.rvs(c=2, scale=5, size=1000)
    
    def test_initialization(self):
        """Test Log-Logistic distribution initialization"""
        dist = LogLogisticDistribution()
        assert dist.info.name == "loglogistic"
    
    def test_fit_mle(self, loglogistic_data):
        """Test MLE fitting"""
        dist = LogLogisticDistribution()
        dist.fit(loglogistic_data, method='mle')
        
        assert dist.fitted
        assert 'c' in dist.params  # alpha (shape)
        assert 'scale' in dist.params  # beta (scale)
        assert dist.params['c'] > 0
        assert dist.params['scale'] > 0
    
    def test_positive_support(self, loglogistic_data):
        """Test Log-Logistic support is (0, infinity)"""
        dist = LogLogisticDistribution()
        dist.fit(loglogistic_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples > 0)
    
    def test_mean_formula(self, loglogistic_data):
        """Test mean formula"""
        dist = LogLogisticDistribution()
        dist.fit(loglogistic_data, method='mle')
        
        alpha = dist.params['c']
        beta = dist.params['scale']
        
        if alpha > 1:
            mean = dist.mean()
            # E[X] = beta * pi / (alpha * sin(pi/alpha)) for alpha > 1
            expected_mean = beta * np.pi / (alpha * np.sin(np.pi / alpha))
            assert abs(mean - expected_mean) / expected_mean < 0.2
    
    def test_median_equals_scale(self, loglogistic_data):
        """Test median equals scale parameter"""
        dist = LogLogisticDistribution()
        dist.fit(loglogistic_data, method='mle')
        
        median = dist.median()
        scale = dist.params['scale']
        
        assert abs(median - scale) / scale < 0.2
    
    def test_survival_analysis_application(self):
        """Test Log-Logistic for survival analysis"""
        np.random.seed(42)
        # Simulate survival times
        survival_times = stats.fisk.rvs(c=1.5, scale=10, size=100)
        
        dist = LogLogisticDistribution()
        dist.fit(survival_times, method='mle')
        
        # Calculate median survival time
        median_survival = dist.median()
        assert median_survival > 0
        
        # Calculate probability of surviving past time t
        t = 15
        survival_prob = dist.sf(np.array([t]))[0]
        assert 0 < survival_prob < 1
    
    def test_hazard_function_shape(self, loglogistic_data):
        """Test hazard function properties"""
        dist = LogLogisticDistribution()
        dist.fit(loglogistic_data, method='mle')
        
        # Hazard rate can be non-monotonic
        t_values = [1, 5, 10, 20]
        hazards = [dist.hazard_rate(t) for t in t_values]
        
        assert all(h > 0 for h in hazards)
    
    def test_pdf_shape(self, loglogistic_data):
        """Test PDF shape characteristics"""
        dist = LogLogisticDistribution()
        dist.fit(loglogistic_data, method='mle')
        
        x = np.linspace(0.1, 20, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
        assert np.max(pdf_values) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAdvancedDistributions3Integration:
    """Integration tests for advanced distributions part 3"""
    
    def test_all_distributions_fit(self):
        """Test all distributions can be fitted"""
        np.random.seed(42)
        
        test_cases = [
            ('f', np.random.f(dfn=5, dfd=10, size=500)),
            ('rayleigh', np.random.rayleigh(scale=2, size=500)),
            ('laplace', np.random.laplace(loc=10, scale=2, size=500)),
            ('invgamma', stats.invgamma.rvs(a=3, scale=2, size=500)),
            ('loglogistic', stats.fisk.rvs(c=2, scale=5, size=500))
        ]
        
        for name, data in test_cases:
            dist = get_distribution(name)
            dist.fit(data, method='mle')
            assert dist.fitted, f"{name} failed to fit"
    
    def test_positive_distributions(self):
        """Test all these distributions are positive-only"""
        np.random.seed(42)
        
        distributions = [
            (FDistribution(), np.random.f(5, 10, 300)),
            (RayleighDistribution(), np.random.rayleigh(2, 300)),
            (InverseGammaDistribution(), stats.invgamma.rvs(3, scale=2, size=300)),
            (LogLogisticDistribution(), stats.fisk.rvs(2, scale=5, size=300))
        ]
        
        for dist, data in distributions:
            dist.fit(data, method='mle')
            samples = dist.rvs(size=50, random_state=42)
            assert np.all(samples >= 0)
    
    def test_cdf_ppf_consistency(self):
        """Test CDF and PPF consistency"""
        np.random.seed(42)
        data = np.random.rayleigh(scale=2, size=500)
        
        dist = RayleighDistribution()
        dist.fit(data, method='mle')
        
        q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        x = dist.ppf(q)
        q_back = dist.cdf(x)
        
        assert np.allclose(q, q_back, rtol=0.01)
    
    def test_summary_output(self):
        """Test summary output for all distributions"""
        np.random.seed(42)
        
        distributions = [
            (FDistribution(), np.random.f(5, 10, 300)),
            (RayleighDistribution(), np.random.rayleigh(2, 300)),
            (LaplaceDistribution(), np.random.laplace(10, 2, 300))
        ]
        
        for dist, data in distributions:
            dist.fit(data, method='mle')
            summary = dist.summary()
            explain = dist.explain()
            
            assert len(summary) > 100
            assert len(explain) > 50
    
    def test_moments_computation(self):
        """Test moment computation"""
        np.random.seed(42)
        data = np.random.rayleigh(scale=2, size=500)
        
        dist = RayleighDistribution()
        dist.fit(data, method='mle')
        
        mean = dist.mean()
        var = dist.var()
        std = dist.std()
        
        assert mean > 0
        assert var > 0
        assert abs(std - np.sqrt(var)) < 0.01
    
    def test_all_continuous_distributions_complete(self):
        """Verify all 20 continuous distributions are tested"""
        continuous_dists = [
            'normal', 'lognormal', 'weibull', 'gamma', 'exponential',
            'beta', 'uniform', 'triangular', 'logistic', 'gumbel',
            'frechet', 'pareto', 'cauchy', 'studentt', 'chisquared',
            'f', 'rayleigh', 'laplace', 'invgamma', 'loglogistic'
        ]
        
        assert len(continuous_dists) == 20
        
        # All should be accessible
        for name in continuous_dists:
            dist = get_distribution(name)
            assert dist is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
