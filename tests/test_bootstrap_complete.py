"""
Comprehensive Tests for Bootstrap Methods and CI (Part 7/10)
=============================================================

Tests for Bootstrap framework:
1. Parametric Bootstrap
2. Non-parametric Bootstrap
3. BCa Bootstrap (Bias-Corrected accelerated)
4. Confidence Intervals
5. Standard Error Estimation

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    NormalDistribution,
    ExponentialDistribution,
    WeibullDistribution,
    get_distribution
)
from distfit_pro.bootstrap import (
    ParametricBootstrap,
    NonParametricBootstrap,
    BCaBootstrap,
    BootstrapResult,
    ConfidenceInterval
)


# ============================================================================
# TEST: PARAMETRIC BOOTSTRAP
# ============================================================================

class TestParametricBootstrap:
    """Comprehensive tests for Parametric Bootstrap"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=100)
    
    @pytest.fixture
    def exponential_data(self):
        """Generate exponential distributed data"""
        np.random.seed(42)
        return np.random.exponential(scale=2, size=100)
    
    def test_initialization(self):
        """Test parametric bootstrap initialization"""
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        assert bootstrap.n_bootstrap == 1000
    
    def test_generate_samples(self, normal_data):
        """Test parametric bootstrap sample generation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=100)
        samples = bootstrap.generate_samples(distribution=dist, size=len(normal_data))
        
        assert len(samples) == 100
        assert all(len(s) == len(normal_data) for s in samples)
    
    def test_parameter_estimation(self, normal_data):
        """Test bootstrap parameter estimation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=500)
        result = bootstrap.estimate(data=normal_data, distribution=dist, statistic='mean')
        
        assert isinstance(result, BootstrapResult)
        assert result.original_statistic is not None
        assert len(result.bootstrap_statistics) == 500
    
    def test_confidence_interval(self, normal_data):
        """Test confidence interval construction"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=normal_data, distribution=dist, statistic='mean')
        
        ci = result.confidence_interval(confidence_level=0.95)
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower < ci.upper
        assert ci.level == 0.95
    
    def test_standard_error(self, normal_data):
        """Test standard error estimation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=normal_data, distribution=dist, statistic='mean')
        
        se = result.standard_error()
        
        # SE should be positive
        assert se > 0
        # SE should be approximately sigma / sqrt(n)
        expected_se = 2 / np.sqrt(100)  # sigma=2, n=100
        assert 0.5 * expected_se < se < 2 * expected_se
    
    def test_bias_estimation(self, normal_data):
        """Test bias estimation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=normal_data, distribution=dist, statistic='mean')
        
        bias = result.bias()
        
        # Bias should be small for unbiased estimator
        assert abs(bias) < 0.5
    
    def test_percentile_ci(self, normal_data):
        """Test percentile confidence interval"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=normal_data, distribution=dist, statistic='mean')
        
        ci = result.confidence_interval(confidence_level=0.95, method='percentile')
        
        # CI should contain original statistic with high probability
        assert ci.lower < result.original_statistic < ci.upper
    
    def test_normal_ci(self, normal_data):
        """Test normal approximation confidence interval"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=normal_data, distribution=dist, statistic='mean')
        
        ci = result.confidence_interval(confidence_level=0.95, method='normal')
        
        # Should be symmetric around original statistic
        lower_dist = result.original_statistic - ci.lower
        upper_dist = ci.upper - result.original_statistic
        assert abs(lower_dist - upper_dist) < 0.5
    
    def test_different_statistics(self, normal_data):
        """Test bootstrap for different statistics"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        bootstrap = ParametricBootstrap(n_bootstrap=500)
        
        # Test mean
        result_mean = bootstrap.estimate(normal_data, dist, statistic='mean')
        assert result_mean.original_statistic is not None
        
        # Test variance
        result_var = bootstrap.estimate(normal_data, dist, statistic='variance')
        assert result_var.original_statistic is not None
        
        # Test median
        result_median = bootstrap.estimate(normal_data, dist, statistic='median')
        assert result_median.original_statistic is not None


# ============================================================================
# TEST: NON-PARAMETRIC BOOTSTRAP
# ============================================================================

class TestNonParametricBootstrap:
    """Comprehensive tests for Non-parametric Bootstrap"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=100)
    
    def test_initialization(self):
        """Test non-parametric bootstrap initialization"""
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        assert bootstrap.n_bootstrap == 1000
    
    def test_resampling(self, sample_data):
        """Test resampling with replacement"""
        bootstrap = NonParametricBootstrap(n_bootstrap=100)
        samples = bootstrap.resample(data=sample_data)
        
        assert len(samples) == 100
        assert all(len(s) == len(sample_data) for s in samples)
    
    def test_bootstrap_mean(self, sample_data):
        """Test bootstrap for mean estimation"""
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        assert isinstance(result, BootstrapResult)
        # Bootstrap mean should be close to sample mean
        assert abs(np.mean(result.bootstrap_statistics) - np.mean(sample_data)) < 0.5
    
    def test_bootstrap_median(self, sample_data):
        """Test bootstrap for median estimation"""
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='median')
        
        # Bootstrap median should be close to sample median
        assert abs(np.median(result.bootstrap_statistics) - np.median(sample_data)) < 0.5
    
    def test_confidence_interval(self, sample_data):
        """Test CI from non-parametric bootstrap"""
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        ci = result.confidence_interval(confidence_level=0.95)
        
        assert ci.lower < ci.upper
        # CI should contain sample mean
        assert ci.lower < np.mean(sample_data) < ci.upper
    
    def test_no_distribution_assumption(self, sample_data):
        """Test that non-parametric bootstrap makes no distribution assumption"""
        bootstrap = NonParametricBootstrap(n_bootstrap=500)
        
        # Works with any data
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        assert result.original_statistic is not None
        
        # Works with skewed data
        skewed_data = np.random.exponential(scale=2, size=100)
        result_skewed = bootstrap.estimate(data=skewed_data, statistic='mean')
        assert result_skewed.original_statistic is not None
    
    def test_custom_statistic(self, sample_data):
        """Test bootstrap with custom statistic function"""
        def custom_stat(data):
            return np.percentile(data, 75)  # 75th percentile
        
        bootstrap = NonParametricBootstrap(n_bootstrap=500)
        result = bootstrap.estimate(data=sample_data, statistic=custom_stat)
        
        assert result.original_statistic == np.percentile(sample_data, 75)
    
    def test_stratified_bootstrap(self):
        """Test stratified bootstrap"""
        np.random.seed(42)
        # Data with two groups
        group1 = np.random.normal(10, 2, 50)
        group2 = np.random.normal(15, 2, 50)
        data = np.concatenate([group1, group2])
        labels = np.array([0]*50 + [1]*50)
        
        bootstrap = NonParametricBootstrap(n_bootstrap=500)
        result = bootstrap.estimate_stratified(data=data, labels=labels, statistic='mean')
        
        assert result.original_statistic is not None
    
    def test_bootstrap_distribution_shape(self, sample_data):
        """Test that bootstrap distribution is approximately normal"""
        bootstrap = NonParametricBootstrap(n_bootstrap=2000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        # Test normality of bootstrap distribution
        _, p_value = stats.shapiro(result.bootstrap_statistics[:1000])
        # Should not strongly reject normality (though may not be perfect)
        assert p_value > 0.001


# ============================================================================
# TEST: BCa BOOTSTRAP
# ============================================================================

class TestBCaBootstrap:
    """Comprehensive tests for BCa (Bias-Corrected accelerated) Bootstrap"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=100)
    
    def test_initialization(self):
        """Test BCa bootstrap initialization"""
        bootstrap = BCaBootstrap(n_bootstrap=1000)
        assert bootstrap.n_bootstrap == 1000
    
    def test_bias_correction(self, sample_data):
        """Test bias correction calculation"""
        bootstrap = BCaBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        # BCa should calculate bias correction factor
        assert hasattr(result, 'bias_correction_factor')
    
    def test_acceleration_constant(self, sample_data):
        """Test acceleration constant calculation"""
        bootstrap = BCaBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        # BCa should calculate acceleration constant
        assert hasattr(result, 'acceleration_constant')
    
    def test_bca_confidence_interval(self, sample_data):
        """Test BCa confidence interval"""
        bootstrap = BCaBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        ci_bca = result.confidence_interval(confidence_level=0.95, method='bca')
        ci_percentile = result.confidence_interval(confidence_level=0.95, method='percentile')
        
        # BCa CI should be different from percentile CI (corrected)
        # May be close but should not be identical
        assert ci_bca.lower != ci_percentile.lower or ci_bca.upper != ci_percentile.upper
    
    def test_bca_vs_percentile(self, sample_data):
        """Test BCa vs percentile CI comparison"""
        bootstrap = BCaBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        ci_bca = result.confidence_interval(0.95, method='bca')
        ci_perc = result.confidence_interval(0.95, method='percentile')
        
        # Both should be valid CIs
        assert ci_bca.lower < ci_bca.upper
        assert ci_perc.lower < ci_perc.upper
    
    def test_skewed_data(self):
        """Test BCa with skewed data (where it shines)"""
        np.random.seed(42)
        # Skewed data (exponential)
        skewed_data = np.random.exponential(scale=2, size=100)
        
        bootstrap = BCaBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data=skewed_data, statistic='mean')
        
        ci = result.confidence_interval(0.95, method='bca')
        
        # Should produce valid CI
        assert ci.lower < ci.upper
    
    def test_jackknife_calculation(self, sample_data):
        """Test jackknife estimates for acceleration"""
        bootstrap = BCaBootstrap(n_bootstrap=500)
        result = bootstrap.estimate(data=sample_data, statistic='mean')
        
        # Jackknife should be used for acceleration constant
        assert hasattr(result, 'jackknife_statistics')
        assert len(result.jackknife_statistics) == len(sample_data)
    
    def test_coverage_probability(self):
        """Test that BCa achieves nominal coverage"""
        np.random.seed(42)
        
        # Simulate multiple experiments
        contains_true = 0
        n_experiments = 50
        true_mean = 10
        
        for _ in range(n_experiments):
            data = np.random.normal(true_mean, 2, size=100)
            
            bootstrap = BCaBootstrap(n_bootstrap=500)
            result = bootstrap.estimate(data=data, statistic='mean')
            ci = result.confidence_interval(0.95, method='bca')
            
            if ci.lower <= true_mean <= ci.upper:
                contains_true += 1
        
        coverage = contains_true / n_experiments
        # Should be close to 0.95 (allow some variance)
        assert 0.80 < coverage < 1.0


# ============================================================================
# TEST: CONFIDENCE INTERVALS
# ============================================================================

class TestConfidenceIntervals:
    """Comprehensive tests for Confidence Intervals"""
    
    @pytest.fixture
    def bootstrap_result(self):
        """Create a bootstrap result for testing"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        return bootstrap.estimate(data=data, statistic='mean')
    
    def test_percentile_method(self, bootstrap_result):
        """Test percentile method CI"""
        ci = bootstrap_result.confidence_interval(0.95, method='percentile')
        
        assert isinstance(ci, ConfidenceInterval)
        assert ci.method == 'percentile'
        assert ci.lower < ci.upper
    
    def test_normal_method(self, bootstrap_result):
        """Test normal approximation CI"""
        ci = bootstrap_result.confidence_interval(0.95, method='normal')
        
        assert ci.method == 'normal'
        assert ci.lower < ci.upper
    
    def test_different_confidence_levels(self, bootstrap_result):
        """Test different confidence levels"""
        ci_90 = bootstrap_result.confidence_interval(0.90)
        ci_95 = bootstrap_result.confidence_interval(0.95)
        ci_99 = bootstrap_result.confidence_interval(0.99)
        
        # Wider confidence level should give wider interval
        width_90 = ci_90.upper - ci_90.lower
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        
        assert width_90 < width_95 < width_99
    
    def test_ci_contains_estimate(self, bootstrap_result):
        """Test CI contains point estimate"""
        ci = bootstrap_result.confidence_interval(0.95)
        
        # Original statistic should often be in CI
        assert ci.lower <= bootstrap_result.original_statistic <= ci.upper
    
    def test_ci_attributes(self, bootstrap_result):
        """Test CI object attributes"""
        ci = bootstrap_result.confidence_interval(0.95)
        
        assert hasattr(ci, 'lower')
        assert hasattr(ci, 'upper')
        assert hasattr(ci, 'level')
        assert hasattr(ci, 'method')
        assert hasattr(ci, 'width')
        
        # Width calculation
        assert abs(ci.width - (ci.upper - ci.lower)) < 1e-10
    
    def test_ci_representation(self, bootstrap_result):
        """Test CI string representation"""
        ci = bootstrap_result.confidence_interval(0.95)
        
        ci_str = str(ci)
        assert '95%' in ci_str or '0.95' in ci_str
        assert str(ci.lower) in ci_str
        assert str(ci.upper) in ci_str


# ============================================================================
# TEST: STANDARD ERROR ESTIMATION
# ============================================================================

class TestStandardErrorEstimation:
    """Tests for standard error estimation via bootstrap"""
    
    def test_se_convergence(self):
        """Test SE estimate converges with more bootstrap samples"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        # Few bootstrap samples
        bootstrap_100 = NonParametricBootstrap(n_bootstrap=100)
        result_100 = bootstrap_100.estimate(data, statistic='mean')
        se_100 = result_100.standard_error()
        
        # Many bootstrap samples
        bootstrap_2000 = NonParametricBootstrap(n_bootstrap=2000)
        result_2000 = bootstrap_2000.estimate(data, statistic='mean')
        se_2000 = result_2000.standard_error()
        
        # Should be similar (more stable with more samples)
        assert abs(se_100 - se_2000) < se_2000 * 0.5
    
    def test_se_formula_consistency(self):
        """Test SE matches theoretical formula for simple case"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        bootstrap = NonParametricBootstrap(n_bootstrap=2000)
        result = bootstrap.estimate(data, statistic='mean')
        se_bootstrap = result.standard_error()
        
        # Theoretical SE for mean
        se_theoretical = np.std(data, ddof=1) / np.sqrt(len(data))
        
        # Should be close
        assert abs(se_bootstrap - se_theoretical) / se_theoretical < 0.3
    
    def test_se_for_different_statistics(self):
        """Test SE for various statistics"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        
        # SE for mean
        result_mean = bootstrap.estimate(data, statistic='mean')
        se_mean = result_mean.standard_error()
        assert se_mean > 0
        
        # SE for median
        result_median = bootstrap.estimate(data, statistic='median')
        se_median = result_median.standard_error()
        assert se_median > 0
        
        # Typically SE(median) > SE(mean) for normal data
        assert se_median > se_mean * 0.8


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestBootstrapIntegration:
    """Integration tests for bootstrap framework"""
    
    def test_complete_bootstrap_pipeline(self):
        """Test complete bootstrap analysis pipeline"""
        np.random.seed(42)
        data = np.random.weibull(a=2, size=100) * 5
        
        # 1. Fit distribution
        dist = WeibullDistribution()
        dist.fit(data, method='mle')
        
        # 2. Parametric bootstrap
        bootstrap = ParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data, dist, statistic='mean')
        
        # 3. Get confidence interval
        ci = result.confidence_interval(0.95)
        
        # 4. Get standard error
        se = result.standard_error()
        
        # 5. Estimate bias
        bias = result.bias()
        
        # All should be valid
        assert ci.lower < ci.upper
        assert se > 0
        assert abs(bias) < 1
    
    def test_parametric_vs_nonparametric(self):
        """Compare parametric and non-parametric bootstrap"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        # Parametric
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        param_bootstrap = ParametricBootstrap(n_bootstrap=1000)
        param_result = param_bootstrap.estimate(data, dist, statistic='mean')
        param_ci = param_result.confidence_interval(0.95)
        
        # Non-parametric
        nonparam_bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        nonparam_result = nonparam_bootstrap.estimate(data, statistic='mean')
        nonparam_ci = nonparam_result.confidence_interval(0.95)
        
        # Should be similar for correctly specified model
        assert abs(param_ci.lower - nonparam_ci.lower) < 1
        assert abs(param_ci.upper - nonparam_ci.upper) < 1
    
    def test_bootstrap_for_complex_statistic(self):
        """Test bootstrap for complex custom statistic"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        # Complex statistic: coefficient of variation
        def cv(data):
            return np.std(data) / np.mean(data)
        
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(data, statistic=cv)
        
        ci = result.confidence_interval(0.95)
        
        # Should produce valid CI
        assert ci.lower < ci.upper
        assert ci.lower > 0  # CV is positive
    
    def test_small_sample_bootstrap(self):
        """Test bootstrap with small sample size"""
        np.random.seed(42)
        small_data = np.random.normal(10, 2, size=20)
        
        bootstrap = NonParametricBootstrap(n_bootstrap=1000)
        result = bootstrap.estimate(small_data, statistic='mean')
        
        ci = result.confidence_interval(0.95)
        
        # Should still work but CI will be wider
        assert ci.lower < ci.upper
        width = ci.upper - ci.lower
        assert width > 0.5  # Relatively wide
    
    def test_bootstrap_reproducibility(self):
        """Test bootstrap is reproducible with seed"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)
        
        # First run
        np.random.seed(100)
        bootstrap1 = NonParametricBootstrap(n_bootstrap=500)
        result1 = bootstrap1.estimate(data, statistic='mean')
        ci1 = result1.confidence_interval(0.95)
        
        # Second run with same seed
        np.random.seed(100)
        bootstrap2 = NonParametricBootstrap(n_bootstrap=500)
        result2 = bootstrap2.estimate(data, statistic='mean')
        ci2 = result2.confidence_interval(0.95)
        
        # Should be identical
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper
    
    def test_all_bootstrap_methods_comparison(self):
        """Compare all three bootstrap methods"""
        np.random.seed(42)
        data = np.random.exponential(scale=2, size=100)
        
        dist = ExponentialDistribution()
        dist.fit(data, method='mle')
        
        # Parametric
        param_bs = ParametricBootstrap(n_bootstrap=1000)
        param_result = param_bs.estimate(data, dist, statistic='mean')
        
        # Non-parametric
        nonparam_bs = NonParametricBootstrap(n_bootstrap=1000)
        nonparam_result = nonparam_bs.estimate(data, statistic='mean')
        
        # BCa
        bca_bs = BCaBootstrap(n_bootstrap=1000)
        bca_result = bca_bs.estimate(data, statistic='mean')
        
        # All should produce valid results
        assert param_result.original_statistic is not None
        assert nonparam_result.original_statistic is not None
        assert bca_result.original_statistic is not None
        
        # CIs should be reasonably similar
        param_ci = param_result.confidence_interval(0.95)
        nonparam_ci = nonparam_result.confidence_interval(0.95)
        bca_ci = bca_result.confidence_interval(0.95, method='bca')
        
        # Check all are valid
        assert param_ci.lower < param_ci.upper
        assert nonparam_ci.lower < nonparam_ci.upper
        assert bca_ci.lower < bca_ci.upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
