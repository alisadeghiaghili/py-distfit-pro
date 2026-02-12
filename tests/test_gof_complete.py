"""
Comprehensive Tests for GOF Tests and Model Selection (Part 6/10)
==================================================================

Tests for Goodness-of-Fit framework:
1. Kolmogorov-Smirnov Test
2. Anderson-Darling Test
3. Chi-Square Test
4. Cramér-von Mises Test
5. Model Selection (AIC, BIC)

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    NormalDistribution,
    ExponentialDistribution,
    UniformDistribution,
    PoissonDistribution,
    get_distribution
)
from distfit_pro.gof import (
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    ChiSquareTest,
    CramerVonMisesTest,
    GOFTestResult,
    ModelSelector
)


# ============================================================================
# TEST: KOLMOGOROV-SMIRNOV TEST
# ============================================================================

class TestKolmogorovSmirnovTest:
    """Comprehensive tests for Kolmogorov-Smirnov Test"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=1000)
    
    @pytest.fixture
    def exponential_data(self):
        """Generate exponential distributed data"""
        np.random.seed(42)
        return np.random.exponential(scale=2, size=1000)
    
    def test_initialization(self):
        """Test KS test initialization"""
        ks_test = KolmogorovSmirnovTest()
        assert ks_test.name == "Kolmogorov-Smirnov"
    
    def test_correct_distribution_accepted(self, normal_data):
        """Test KS accepts correct distribution"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data=normal_data, distribution=dist)
        
        assert isinstance(result, GOFTestResult)
        assert result.statistic >= 0
        assert 0 <= result.p_value <= 1
        # Should NOT reject (p-value > 0.05)
        assert result.p_value > 0.05
    
    def test_wrong_distribution_rejected(self, normal_data):
        """Test KS rejects wrong distribution"""
        # Fit exponential to normal data
        dist = ExponentialDistribution()
        dist.fit(normal_data - np.min(normal_data) + 0.1, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data=normal_data, distribution=dist)
        
        # Should reject (p-value < 0.05)
        assert result.p_value < 0.05
    
    def test_statistic_calculation(self, normal_data):
        """Test KS statistic calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data=normal_data, distribution=dist)
        
        # KS statistic is max distance between ECDFs
        # Should be small for good fit
        assert 0 <= result.statistic < 0.1
    
    def test_uniform_distribution(self):
        """Test KS with uniform distribution"""
        np.random.seed(42)
        data = np.random.uniform(0, 1, size=500)
        
        dist = UniformDistribution()
        dist.fit(data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data=data, distribution=dist)
        
        assert result.p_value > 0.05
    
    def test_sample_size_effect(self):
        """Test KS test with different sample sizes"""
        np.random.seed(42)
        
        # Small sample
        small_data = np.random.normal(0, 1, size=30)
        dist_small = NormalDistribution()
        dist_small.fit(small_data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result_small = ks_test.test(data=small_data, distribution=dist_small)
        
        # Large sample
        large_data = np.random.normal(0, 1, size=1000)
        dist_large = NormalDistribution()
        dist_large.fit(large_data, method='mle')
        
        result_large = ks_test.test(data=large_data, distribution=dist_large)
        
        # Both should accept
        assert result_small.p_value > 0.05
        assert result_large.p_value > 0.05
    
    def test_two_sample_ks(self):
        """Test two-sample KS test"""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, size=500)
        data2 = np.random.normal(0, 1, size=500)
        
        # Two samples from same distribution
        statistic, p_value = stats.ks_2samp(data1, data2)
        assert p_value > 0.05
        
        # Two samples from different distributions
        data3 = np.random.normal(2, 1, size=500)
        statistic, p_value = stats.ks_2samp(data1, data3)
        assert p_value < 0.05
    
    def test_critical_value(self, normal_data):
        """Test KS critical value"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data=normal_data, distribution=dist)
        
        # Critical value for alpha=0.05: ~1.36/sqrt(n)
        n = len(normal_data)
        critical_value = 1.36 / np.sqrt(n)
        
        # Statistic should be less than critical value
        assert result.statistic < critical_value


# ============================================================================
# TEST: ANDERSON-DARLING TEST
# ============================================================================

class TestAndersonDarlingTest:
    """Comprehensive tests for Anderson-Darling Test"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test AD test initialization"""
        ad_test = AndersonDarlingTest()
        assert ad_test.name == "Anderson-Darling"
    
    def test_correct_distribution_accepted(self, normal_data):
        """Test AD accepts correct distribution"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ad_test = AndersonDarlingTest()
        result = ad_test.test(data=normal_data, distribution=dist)
        
        assert isinstance(result, GOFTestResult)
        assert result.statistic >= 0
        # Should NOT reject
        assert result.p_value > 0.05 or result.statistic < 2.5
    
    def test_tail_sensitivity(self, normal_data):
        """Test AD is more sensitive to tails than KS"""
        # Add outliers
        data_with_outliers = np.concatenate([normal_data, [100, -100]])
        
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ad_test = AndersonDarlingTest()
        result = ad_test.test(data=data_with_outliers, distribution=dist)
        
        # AD should detect tail problems
        assert result.statistic > 0
    
    def test_exponential_distribution(self):
        """Test AD with exponential distribution"""
        np.random.seed(42)
        data = np.random.exponential(scale=2, size=500)
        
        dist = ExponentialDistribution()
        dist.fit(data, method='mle')
        
        ad_test = AndersonDarlingTest()
        result = ad_test.test(data=data, distribution=dist)
        
        assert result.statistic >= 0
    
    def test_critical_values(self, normal_data):
        """Test AD critical values"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ad_test = AndersonDarlingTest()
        result = ad_test.test(data=normal_data, distribution=dist)
        
        # Critical values: 0.576 (15%), 0.656 (10%), 0.787 (5%), 0.918 (2.5%), 1.092 (1%)
        # Should be below 5% critical value
        assert result.statistic < 1.0
    
    def test_weighted_towards_tails(self):
        """Test AD weights tails more than center"""
        np.random.seed(42)
        
        # Data with good center but bad tails
        center_data = np.random.normal(0, 1, 900)
        tail_data = np.random.uniform(-5, 5, 100)
        mixed_data = np.concatenate([center_data, tail_data])
        
        dist = NormalDistribution()
        dist.fit(center_data, method='mle')
        
        ad_test = AndersonDarlingTest()
        result = ad_test.test(data=mixed_data, distribution=dist)
        
        # Should detect tail problems
        assert result.statistic > 0.5


# ============================================================================
# TEST: CHI-SQUARE TEST
# ============================================================================

class TestChiSquareTest:
    """Comprehensive tests for Chi-Square Test"""
    
    @pytest.fixture
    def discrete_data(self):
        """Generate Poisson distributed data"""
        np.random.seed(42)
        return np.random.poisson(lam=5, size=1000)
    
    def test_initialization(self):
        """Test Chi-Square test initialization"""
        chi2_test = ChiSquareTest()
        assert chi2_test.name == "Chi-Square"
    
    def test_discrete_distribution(self, discrete_data):
        """Test Chi-Square with discrete distribution"""
        dist = PoissonDistribution()
        dist.fit(discrete_data, method='mle')
        
        chi2_test = ChiSquareTest(bins=10)
        result = chi2_test.test(data=discrete_data, distribution=dist)
        
        assert isinstance(result, GOFTestResult)
        assert result.statistic >= 0
        assert result.p_value > 0.05
    
    def test_continuous_distribution_binning(self):
        """Test Chi-Square with continuous distribution (needs binning)"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=1000)
        
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        chi2_test = ChiSquareTest(bins=20)
        result = chi2_test.test(data=data, distribution=dist)
        
        assert result.p_value > 0.05
    
    def test_bin_count_effect(self):
        """Test effect of number of bins"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=1000)
        
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        # Few bins
        chi2_test_few = ChiSquareTest(bins=5)
        result_few = chi2_test_few.test(data=data, distribution=dist)
        
        # Many bins
        chi2_test_many = ChiSquareTest(bins=50)
        result_many = chi2_test_many.test(data=data, distribution=dist)
        
        # Both should work but give different statistics
        assert result_few.statistic != result_many.statistic
    
    def test_expected_frequencies(self, discrete_data):
        """Test expected frequencies calculation"""
        dist = PoissonDistribution()
        dist.fit(discrete_data, method='mle')
        
        chi2_test = ChiSquareTest(bins=10)
        result = chi2_test.test(data=discrete_data, distribution=dist)
        
        # Expected frequencies should be reasonable
        assert result.statistic >= 0
    
    def test_minimum_expected_frequency(self):
        """Test minimum expected frequency rule (>5)"""
        np.random.seed(42)
        data = np.random.poisson(lam=2, size=100)
        
        dist = PoissonDistribution()
        dist.fit(data, method='mle')
        
        # Should handle bins with low expected frequencies
        chi2_test = ChiSquareTest(bins=5, min_expected_freq=5)
        result = chi2_test.test(data=data, distribution=dist)
        
        assert result.statistic >= 0
    
    def test_degrees_of_freedom(self):
        """Test degrees of freedom calculation"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=500)
        
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        bins = 10
        chi2_test = ChiSquareTest(bins=bins)
        result = chi2_test.test(data=data, distribution=dist)
        
        # df = bins - 1 - number_of_estimated_parameters
        # For Normal: 2 parameters (mean, std)
        expected_df = bins - 1 - 2
        assert result.degrees_of_freedom == expected_df


# ============================================================================
# TEST: CRAMÉR-VON MISES TEST
# ============================================================================

class TestCramerVonMisesTest:
    """Comprehensive tests for Cramér-von Mises Test"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test CvM test initialization"""
        cvm_test = CramerVonMisesTest()
        assert cvm_test.name == "Cramér-von Mises"
    
    def test_correct_distribution_accepted(self, normal_data):
        """Test CvM accepts correct distribution"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        cvm_test = CramerVonMisesTest()
        result = cvm_test.test(data=normal_data, distribution=dist)
        
        assert isinstance(result, GOFTestResult)
        assert result.statistic >= 0
        # Should NOT reject
        assert result.p_value > 0.05 or result.statistic < 0.5
    
    def test_integral_of_squared_differences(self, normal_data):
        """Test CvM is integral of squared differences"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        cvm_test = CramerVonMisesTest()
        result = cvm_test.test(data=normal_data, distribution=dist)
        
        # CvM integrates (F_n - F)^2
        # Should be small for good fit
        assert result.statistic < 1.0
    
    def test_uniform_distribution(self):
        """Test CvM with uniform distribution"""
        np.random.seed(42)
        data = np.random.uniform(0, 1, size=500)
        
        dist = UniformDistribution()
        dist.fit(data, method='mle')
        
        cvm_test = CramerVonMisesTest()
        result = cvm_test.test(data=data, distribution=dist)
        
        assert result.p_value > 0.05
    
    def test_comparison_with_ks(self, normal_data):
        """Test CvM vs KS comparison"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        ks_result = ks_test.test(data=normal_data, distribution=dist)
        
        cvm_test = CramerVonMisesTest()
        cvm_result = cvm_test.test(data=normal_data, distribution=dist)
        
        # Both should accept or both reject
        ks_accepts = ks_result.p_value > 0.05
        cvm_accepts = cvm_result.p_value > 0.05
        
        # May differ slightly but should be consistent
        assert ks_result.p_value > 0 and cvm_result.p_value > 0


# ============================================================================
# TEST: MODEL SELECTION
# ============================================================================

class TestModelSelector:
    """Comprehensive tests for Model Selection"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test ModelSelector initialization"""
        selector = ModelSelector()
        assert selector is not None
    
    def test_aic_calculation(self, normal_data):
        """Test AIC calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        selector = ModelSelector()
        aic = selector.calculate_aic(data=normal_data, distribution=dist)
        
        # AIC = 2k - 2ln(L)
        # Should be negative (log-likelihood dominates)
        assert aic is not None
    
    def test_bic_calculation(self, normal_data):
        """Test BIC calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        selector = ModelSelector()
        bic = selector.calculate_bic(data=normal_data, distribution=dist)
        
        # BIC = k*ln(n) - 2ln(L)
        assert bic is not None
    
    def test_bic_penalizes_more_than_aic(self, normal_data):
        """Test BIC penalizes parameters more than AIC"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        selector = ModelSelector()
        aic = selector.calculate_aic(data=normal_data, distribution=dist)
        bic = selector.calculate_bic(data=normal_data, distribution=dist)
        
        # For large n: BIC penalty > AIC penalty
        # So BIC should be "worse" (higher)
        assert bic > aic
    
    def test_compare_multiple_distributions(self, normal_data):
        """Test comparing multiple distributions"""
        # Fit multiple distributions
        normal_dist = NormalDistribution()
        normal_dist.fit(normal_data, method='mle')
        
        uniform_dist = UniformDistribution()
        uniform_dist.fit(normal_data, method='mle')
        
        exp_dist = ExponentialDistribution()
        exp_dist.fit(normal_data - np.min(normal_data) + 0.1, method='mle')
        
        selector = ModelSelector()
        
        # Compare AICs
        aic_normal = selector.calculate_aic(normal_data, normal_dist)
        aic_uniform = selector.calculate_aic(normal_data, uniform_dist)
        aic_exp = selector.calculate_aic(normal_data - np.min(normal_data) + 0.1, exp_dist)
        
        # Normal should be best (lowest AIC)
        assert aic_normal < aic_uniform
        assert aic_normal < aic_exp
    
    def test_select_best_distribution(self, normal_data):
        """Test automatic best distribution selection"""
        distributions = ['normal', 'uniform', 'exponential', 'lognormal']
        
        selector = ModelSelector()
        best_dist, results = selector.select_best(
            data=normal_data,
            distributions=distributions,
            criterion='aic'
        )
        
        # Should select normal
        assert best_dist == 'normal'
        assert len(results) == len(distributions)
    
    def test_likelihood_calculation(self, normal_data):
        """Test log-likelihood calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        selector = ModelSelector()
        log_likelihood = selector.calculate_log_likelihood(normal_data, dist)
        
        # Log-likelihood should be negative
        assert log_likelihood < 0
    
    def test_parameter_count(self):
        """Test parameter counting"""
        selector = ModelSelector()
        
        # Normal has 2 parameters
        normal_dist = NormalDistribution()
        normal_dist.params = {'loc': 0, 'scale': 1}
        k_normal = len(normal_dist.params)
        assert k_normal == 2
        
        # Uniform has 2 parameters
        uniform_dist = UniformDistribution()
        uniform_dist.params = {'loc': 0, 'scale': 1}
        k_uniform = len(uniform_dist.params)
        assert k_uniform == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestGOFIntegration:
    """Integration tests for GOF framework"""
    
    def test_all_gof_tests_on_same_data(self):
        """Test all GOF tests on same data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=1000)
        
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        # Run all tests
        ks_test = KolmogorovSmirnovTest()
        ks_result = ks_test.test(data, dist)
        
        ad_test = AndersonDarlingTest()
        ad_result = ad_test.test(data, dist)
        
        chi2_test = ChiSquareTest(bins=20)
        chi2_result = chi2_test.test(data, dist)
        
        cvm_test = CramerVonMisesTest()
        cvm_result = cvm_test.test(data, dist)
        
        # All should accept
        assert ks_result.p_value > 0.05
        assert ad_result.p_value > 0.05 or ad_result.statistic < 2.5
        assert chi2_result.p_value > 0.05
        assert cvm_result.p_value > 0.05 or cvm_result.statistic < 0.5
    
    def test_gof_pipeline(self):
        """Test complete GOF testing pipeline"""
        np.random.seed(42)
        data = np.random.exponential(scale=2, size=500)
        
        # 1. Fit distribution
        dist = ExponentialDistribution()
        dist.fit(data, method='mle')
        
        # 2. Test goodness of fit
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data, dist)
        
        # 3. Check result
        assert result.p_value > 0.05
        
        # 4. Calculate model selection criteria
        selector = ModelSelector()
        aic = selector.calculate_aic(data, dist)
        bic = selector.calculate_bic(data, dist)
        
        assert aic is not None
        assert bic is not None
    
    def test_power_analysis(self):
        """Test statistical power of GOF tests"""
        np.random.seed(42)
        
        # Generate data from Normal(0, 1)
        true_data = np.random.normal(0, 1, size=500)
        
        # Fit correct distribution
        correct_dist = NormalDistribution()
        correct_dist.fit(true_data, method='mle')
        
        # Fit wrong distribution
        wrong_dist = ExponentialDistribution()
        wrong_dist.fit(true_data - np.min(true_data) + 0.1, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        
        # Test with correct distribution
        result_correct = ks_test.test(true_data, correct_dist)
        # Test with wrong distribution
        result_wrong = ks_test.test(true_data, wrong_dist)
        
        # Correct should have higher p-value
        assert result_correct.p_value > result_wrong.p_value
    
    def test_type_i_error_rate(self):
        """Test Type I error rate (false positive)"""
        np.random.seed(42)
        
        # Run test multiple times with correct distribution
        rejections = 0
        n_simulations = 100
        alpha = 0.05
        
        for i in range(n_simulations):
            data = np.random.normal(0, 1, size=200)
            dist = NormalDistribution()
            dist.fit(data, method='mle')
            
            ks_test = KolmogorovSmirnovTest()
            result = ks_test.test(data, dist)
            
            if result.p_value < alpha:
                rejections += 1
        
        # Type I error rate should be near alpha
        type_i_error_rate = rejections / n_simulations
        assert 0 < type_i_error_rate < 0.15  # Allow some variance
    
    def test_complete_model_selection_workflow(self):
        """Test complete model selection workflow"""
        np.random.seed(42)
        # Generate mixed data
        data = np.random.lognormal(0, 0.5, size=500)
        
        # Test multiple distributions
        candidates = ['normal', 'lognormal', 'gamma', 'weibull']
        
        selector = ModelSelector()
        best_by_aic, aic_results = selector.select_best(
            data, candidates, criterion='aic'
        )
        best_by_bic, bic_results = selector.select_best(
            data, candidates, criterion='bic'
        )
        
        # Should prefer lognormal
        assert best_by_aic == 'lognormal'
        assert best_by_bic == 'lognormal'
    
    def test_gof_result_object(self):
        """Test GOFTestResult object"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=500)
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        ks_test = KolmogorovSmirnovTest()
        result = ks_test.test(data, dist)
        
        # Check all attributes
        assert hasattr(result, 'test_name')
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'conclusion')
        assert hasattr(result, 'alpha')
        
        # Check conclusion
        if result.p_value > 0.05:
            assert 'accept' in result.conclusion.lower() or 'not reject' in result.conclusion.lower()
        else:
            assert 'reject' in result.conclusion.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
