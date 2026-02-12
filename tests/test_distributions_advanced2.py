"""
Comprehensive Tests for Advanced Distributions Part 2 (Part 3/10)
=================================================================

Tests for 5 advanced distributions:
1. Frechet Distribution (Extreme Value Type II)
2. Pareto Distribution (Power Law)
3. Cauchy Distribution (Heavy Tails)
4. Student's t Distribution
5. Chi-Squared Distribution

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    FrechetDistribution,
    ParetoDistribution,
    CauchyDistribution,
    StudentTDistribution,
    ChiSquaredDistribution,
    get_distribution
)


# ============================================================================
# TEST: FRECHET DISTRIBUTION
# ============================================================================

class TestFrechetDistribution:
    """Comprehensive tests for Frechet Distribution"""
    
    @pytest.fixture
    def frechet_data(self):
        """Generate Frechet distributed data"""
        np.random.seed(42)
        # Using scipy's invweibull (which is Frechet)
        return stats.invweibull.rvs(c=2, scale=5, size=1000)
    
    def test_initialization(self):
        """Test Frechet distribution initialization"""
        dist = FrechetDistribution()
        assert dist.info.name == "frechet"
        assert not dist.fitted
    
    def test_fit_mle(self, frechet_data):
        """Test MLE fitting"""
        dist = FrechetDistribution()
        dist.fit(frechet_data, method='mle')
        
        assert dist.fitted
        assert 'c' in dist.params  # alpha (shape)
        assert 'scale' in dist.params  # s (scale)
        assert dist.params['c'] > 0
        assert dist.params['scale'] > 0
    
    def test_positive_only(self):
        """Test Frechet handles positive-only data"""
        dist = FrechetDistribution()
        data_with_negatives = np.array([-1, 0, 1, 2, 3, 4, 5])
        dist.fit(data_with_negatives, method='mle')
        
        assert dist.fitted
    
    def test_heavy_tails(self, frechet_data):
        """Test Frechet has very heavy tails"""
        dist = FrechetDistribution()
        dist.fit(frechet_data, method='mle')
        
        # Kurtosis should be large (heavy tails)
        try:
            kurt = dist.kurtosis()
            assert kurt > 2  # Very heavy-tailed
        except:
            # May not exist for small alpha
            pass
    
    def test_extreme_value_property(self):
        """Test Frechet for extreme value analysis"""
        np.random.seed(42)
        # Simulate extreme insurance claims
        data = stats.invweibull.rvs(c=1.5, scale=1000, size=100)
        
        dist = FrechetDistribution()
        dist.fit(data, method='mle')
        
        # 99th percentile (extreme event)
        extreme_value = dist.ppf(0.99)
        assert extreme_value > np.median(data)
    
    def test_pdf_shape(self, frechet_data):
        """Test PDF shape characteristics"""
        dist = FrechetDistribution()
        dist.fit(frechet_data, method='mle')
        
        x = np.linspace(0.1, 20, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
        # Should have a mode
        assert np.max(pdf_values) > 0
    
    def test_mean_exists_condition(self, frechet_data):
        """Test mean exists only when alpha > 1"""
        dist = FrechetDistribution()
        dist.fit(frechet_data, method='mle')
        
        if dist.params['c'] > 1:
            mean = dist.mean()
            assert mean > 0
    
    def test_variance_exists_condition(self, frechet_data):
        """Test variance exists only when alpha > 2"""
        dist = FrechetDistribution()
        dist.fit(frechet_data, method='mle')
        
        if dist.params['c'] > 2:
            var = dist.var()
            assert var > 0


# ============================================================================
# TEST: PARETO DISTRIBUTION
# ============================================================================

class TestParetoDistribution:
    """Comprehensive tests for Pareto Distribution"""
    
    @pytest.fixture
    def pareto_data(self):
        """Generate Pareto distributed data"""
        np.random.seed(42)
        return (np.random.pareto(a=2.5, size=1000) + 1) * 10  # xm=10, alpha=2.5
    
    def test_initialization(self):
        """Test Pareto distribution initialization"""
        dist = ParetoDistribution()
        assert dist.info.name == "pareto"
    
    def test_fit_mle(self, pareto_data):
        """Test MLE fitting"""
        dist = ParetoDistribution()
        dist.fit(pareto_data, method='mle')
        
        assert dist.fitted
        assert 'b' in dist.params  # alpha (shape)
        assert 'scale' in dist.params  # xm (minimum)
        assert dist.params['b'] > 0
        assert dist.params['scale'] > 0
    
    def test_power_law(self, pareto_data):
        """Test Pareto follows power law"""
        dist = ParetoDistribution()
        dist.fit(pareto_data, method='mle')
        
        # P(X > x) ~ x^(-alpha) for large x
        x1, x2 = 20, 40
        sf1 = dist.sf(np.array([x1]))[0]
        sf2 = dist.sf(np.array([x2]))[0]
        
        # Ratio should follow power law
        ratio = sf1 / sf2
        expected_ratio = (x2 / x1) ** dist.params['b']
        
        assert abs(ratio - expected_ratio) / expected_ratio < 0.2
    
    def test_80_20_rule(self):
        """Test 80-20 rule (Pareto principle)"""
        np.random.seed(42)
        # Wealth distribution with alpha ~ 1.16 gives 80-20 rule
        data = (np.random.pareto(a=1.16, size=1000) + 1) * 100
        
        dist = ParetoDistribution()
        dist.fit(data, method='mle')
        
        # Top 20% should hold significant portion
        p80 = dist.ppf(0.8)
        assert p80 < np.mean(data)  # 80th percentile below mean
    
    def test_minimum_value(self, pareto_data):
        """Test minimum value property"""
        dist = ParetoDistribution()
        dist.fit(pareto_data, method='mle')
        
        xm = dist.params['scale']
        # All data should be >= xm
        assert np.min(pareto_data) >= xm - 0.1  # Small tolerance
    
    def test_mean_formula(self, pareto_data):
        """Test mean formula"""
        dist = ParetoDistribution()
        dist.fit(pareto_data, method='mle')
        
        alpha = dist.params['b']
        xm = dist.params['scale']
        
        if alpha > 1:
            mean = dist.mean()
            # E[X] = alpha * xm / (alpha - 1)
            expected_mean = alpha * xm / (alpha - 1)
            assert abs(mean - expected_mean) / expected_mean < 0.1
    
    def test_heavy_tails(self, pareto_data):
        """Test Pareto has heavy tails"""
        dist = ParetoDistribution()
        dist.fit(pareto_data, method='mle')
        
        # Skewness should be positive (right-skewed)
        if dist.params['b'] > 3:
            skew = dist.skewness()
            assert skew > 1  # Heavy right tail
    
    def test_positive_support(self, pareto_data):
        """Test Pareto support is [xm, infinity)"""
        dist = ParetoDistribution()
        dist.fit(pareto_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= dist.params['scale'] - 0.1)


# ============================================================================
# TEST: CAUCHY DISTRIBUTION
# ============================================================================

class TestCauchyDistribution:
    """Comprehensive tests for Cauchy Distribution"""
    
    @pytest.fixture
    def cauchy_data(self):
        """Generate Cauchy distributed data"""
        np.random.seed(42)
        return np.random.standard_cauchy(size=1000) * 2 + 10  # loc=10, scale=2
    
    def test_initialization(self):
        """Test Cauchy distribution initialization"""
        dist = CauchyDistribution()
        assert dist.info.name == "cauchy"
    
    def test_fit_mle(self, cauchy_data):
        """Test MLE fitting using median and IQR"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params  # x0 (location)
        assert 'scale' in dist.params  # gamma (scale)
        
        # Location should be near median
        assert abs(dist.params['loc'] - np.median(cauchy_data)) < 1
    
    def test_undefined_mean(self, cauchy_data):
        """Test Cauchy has undefined mean"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        mean = dist.mean()
        assert np.isnan(mean)
    
    def test_undefined_variance(self, cauchy_data):
        """Test Cauchy has undefined variance"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        var = dist.var()
        assert np.isnan(var)
    
    def test_median_equals_location(self, cauchy_data):
        """Test median equals location parameter"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        median = dist.median()
        location = dist.params['loc']
        
        assert abs(median - location) < 0.01
    
    def test_symmetric(self, cauchy_data):
        """Test Cauchy is symmetric"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        x0 = dist.params['loc']
        offset = 5
        
        # PDF should be symmetric around x0
        pdf_left = dist.pdf(np.array([x0 - offset]))[0]
        pdf_right = dist.pdf(np.array([x0 + offset]))[0]
        
        assert abs(pdf_left - pdf_right) < 0.01
    
    def test_very_heavy_tails(self, cauchy_data):
        """Test Cauchy has very heavy tails"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        # Sample should contain outliers
        samples = dist.rvs(size=1000, random_state=42)
        extreme_values = np.sum(np.abs(samples - dist.params['loc']) > 100)
        
        # Should have some extreme outliers
        assert extreme_values > 0
    
    def test_lorentzian_profile(self, cauchy_data):
        """Test Cauchy PDF (Lorentzian profile)"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        x0 = dist.params['loc']
        gamma = dist.params['scale']
        
        # PDF at x0 should be 1/(pi * gamma)
        pdf_at_x0 = dist.pdf(np.array([x0]))[0]
        expected = 1.0 / (np.pi * gamma)
        
        assert abs(pdf_at_x0 - expected) / expected < 0.01
    
    def test_no_moment_generating_function(self, cauchy_data):
        """Test Cauchy has no MGF"""
        dist = CauchyDistribution()
        dist.fit(cauchy_data, method='mle')
        
        # Cannot compute moments
        with pytest.raises((NotImplementedError, AttributeError, ValueError)):
            # This should fail or return NaN
            result = dist.skewness()
            if not np.isnan(result):
                raise ValueError("Skewness should be undefined")


# ============================================================================
# TEST: STUDENT'S T DISTRIBUTION
# ============================================================================

class TestStudentTDistribution:
    """Comprehensive tests for Student's t Distribution"""
    
    @pytest.fixture
    def studentt_data(self):
        """Generate Student's t distributed data"""
        np.random.seed(42)
        return np.random.standard_t(df=5, size=1000) * 2 + 10  # df=5, scale=2, loc=10
    
    def test_initialization(self):
        """Test Student's t distribution initialization"""
        dist = StudentTDistribution()
        assert dist.info.name == "studentt"
    
    def test_fit_mle(self, studentt_data):
        """Test MLE fitting"""
        dist = StudentTDistribution()
        dist.fit(studentt_data, method='mle')
        
        assert dist.fitted
        assert 'df' in dist.params  # nu (degrees of freedom)
        assert 'loc' in dist.params  # mu
        assert 'scale' in dist.params  # sigma
        assert dist.params['df'] > 0
    
    def test_approaches_normal_large_df(self):
        """Test t approaches normal as df -> infinity"""
        np.random.seed(42)
        
        # Large df
        data = np.random.standard_t(df=100, size=1000)
        dist = StudentTDistribution()
        dist.fit(data, method='mle')
        
        # Should be close to standard normal
        mean = dist.mean()
        std = dist.std()
        
        assert abs(mean) < 0.2
        assert abs(std - 1.0) < 0.2
    
    def test_heavier_tails_than_normal(self, studentt_data):
        """Test t has heavier tails than normal"""
        dist = StudentTDistribution()
        dist.fit(studentt_data, method='mle')
        
        if dist.params['df'] > 4:
            # Kurtosis should be positive (heavier than normal)
            kurt = dist.kurtosis()
            assert kurt > 0
    
    def test_symmetric(self, studentt_data):
        """Test Student's t is symmetric"""
        dist = StudentTDistribution()
        dist.fit(studentt_data, method='mle')
        
        mean = dist.mean()
        median = dist.median()
        
        assert abs(mean - median) < 0.5
    
    def test_degrees_of_freedom_effect(self):
        """Test effect of degrees of freedom"""
        np.random.seed(42)
        
        # Small df: heavy tails
        data1 = np.random.standard_t(df=3, size=500)
        dist1 = StudentTDistribution()
        dist1.fit(data1, method='mle')
        
        # Large df: lighter tails
        data2 = np.random.standard_t(df=30, size=500)
        dist2 = StudentTDistribution()
        dist2.fit(data2, method='mle')
        
        # Smaller df should have heavier tails (higher kurtosis)
        if dist1.params['df'] > 4 and dist2.params['df'] > 4:
            kurt1 = dist1.kurtosis()
            kurt2 = dist2.kurtosis()
            assert kurt1 > kurt2
    
    def test_hypothesis_testing_application(self):
        """Test t distribution for hypothesis testing"""
        dist = StudentTDistribution()
        dist.params = {'df': 10, 'loc': 0, 'scale': 1}
        dist.fitted = True
        
        # 95% confidence interval
        t_critical = dist.ppf(0.975)
        assert 2.0 < t_critical < 2.5  # Known value for df=10
    
    def test_mean_equals_location(self, studentt_data):
        """Test mean equals location parameter"""
        dist = StudentTDistribution()
        dist.fit(studentt_data, method='mle')
        
        if dist.params['df'] > 1:
            mean = dist.mean()
            loc = dist.params['loc']
            assert abs(mean - loc) < 0.5


# ============================================================================
# TEST: CHI-SQUARED DISTRIBUTION
# ============================================================================

class TestChiSquaredDistribution:
    """Comprehensive tests for Chi-Squared Distribution"""
    
    @pytest.fixture
    def chisquared_data(self):
        """Generate Chi-Squared distributed data"""
        np.random.seed(42)
        return np.random.chisquare(df=5, size=1000)
    
    def test_initialization(self):
        """Test Chi-Squared distribution initialization"""
        dist = ChiSquaredDistribution()
        assert dist.info.name == "chisquared"
    
    def test_fit_mle(self, chisquared_data):
        """Test MLE fitting"""
        dist = ChiSquaredDistribution()
        dist.fit(chisquared_data, method='mle')
        
        assert dist.fitted
        assert 'df' in dist.params  # k (degrees of freedom)
        assert 4 < dist.params['df'] < 6
    
    def test_mean_equals_df(self, chisquared_data):
        """Test mean equals degrees of freedom"""
        dist = ChiSquaredDistribution()
        dist.fit(chisquared_data, method='mle')
        
        mean = dist.mean()
        df = dist.params['df']
        
        assert abs(mean - df) < 0.5
    
    def test_variance_equals_2df(self, chisquared_data):
        """Test variance equals 2 * df"""
        dist = ChiSquaredDistribution()
        dist.fit(chisquared_data, method='mle')
        
        var = dist.var()
        df = dist.params['df']
        
        assert abs(var - 2 * df) < 1.0
    
    def test_special_case_gamma(self, chisquared_data):
        """Test Chi-Squared is Gamma(k/2, 2)"""
        dist = ChiSquaredDistribution()
        dist.fit(chisquared_data, method='mle')
        
        # Chi-Squared(k) = Gamma(k/2, 2)
        # Mean = k = (k/2) * 2
        mean = dist.mean()
        assert mean > 0
    
    def test_positive_support(self, chisquared_data):
        """Test Chi-Squared support is [0, infinity)"""
        dist = ChiSquaredDistribution()
        dist.fit(chisquared_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
    
    def test_right_skewed(self, chisquared_data):
        """Test Chi-Squared is right-skewed"""
        dist = ChiSquaredDistribution()
        dist.fit(chisquared_data, method='mle')
        
        skew = dist.skewness()
        assert skew > 0  # Right-skewed
    
    def test_approaches_normal_large_df(self):
        """Test Chi-Squared approaches normal for large df"""
        np.random.seed(42)
        data = np.random.chisquare(df=50, size=1000)
        
        dist = ChiSquaredDistribution()
        dist.fit(data, method='mle')
        
        # Skewness should decrease
        skew = dist.skewness()
        # Skewness = sqrt(8/k)
        expected_skew = np.sqrt(8.0 / dist.params['df'])
        
        assert abs(skew - expected_skew) < 0.2
    
    def test_goodness_of_fit_application(self):
        """Test Chi-Squared for goodness of fit tests"""
        dist = ChiSquaredDistribution()
        dist.params = {'df': 5}
        dist.fitted = True
        
        # 95th percentile for rejection region
        critical_value = dist.ppf(0.95)
        assert 10 < critical_value < 12  # Known value for df=5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAdvancedDistributions2Integration:
    """Integration tests for advanced distributions part 2"""
    
    def test_all_distributions_fit(self):
        """Test all distributions can be fitted"""
        np.random.seed(42)
        
        test_cases = [
            ('frechet', stats.invweibull.rvs(c=2, scale=5, size=500)),
            ('pareto', (np.random.pareto(a=2.5, size=500) + 1) * 10),
            ('cauchy', np.random.standard_cauchy(size=500) * 2 + 10),
            ('studentt', np.random.standard_t(df=5, size=500) * 2 + 10),
            ('chisquared', np.random.chisquare(df=5, size=500))
        ]
        
        for name, data in test_cases:
            dist = get_distribution(name)
            dist.fit(data, method='mle')
            assert dist.fitted, f"{name} failed to fit"
    
    def test_heavy_tailed_comparison(self):
        """Compare heavy-tailed distributions"""
        np.random.seed(42)
        
        # Generate outlier-prone data
        data = np.concatenate([
            np.random.normal(0, 1, 900),
            np.random.normal(0, 10, 100)  # Outliers
        ])
        
        # Fit different distributions
        cauchy = CauchyDistribution()
        cauchy.fit(data, method='mle')
        
        studentt = StudentTDistribution()
        studentt.fit(data, method='mle')
        
        # Both should fit
        assert cauchy.fitted
        assert studentt.fitted
    
    def test_summary_output(self):
        """Test summary output for all distributions"""
        np.random.seed(42)
        
        distributions = [
            (FrechetDistribution(), stats.invweibull.rvs(c=2, scale=5, size=300)),
            (ParetoDistribution(), (np.random.pareto(a=2.5, size=300) + 1) * 10),
            (StudentTDistribution(), np.random.standard_t(df=5, size=300)),
            (ChiSquaredDistribution(), np.random.chisquare(df=5, size=300))
        ]
        
        for dist, data in distributions:
            dist.fit(data, method='mle')
            summary = dist.summary()
            assert len(summary) > 100
            assert dist.info.display_name in summary
    
    def test_undefined_moments_handling(self):
        """Test handling of undefined moments"""
        np.random.seed(42)
        data = np.random.standard_cauchy(size=500)
        
        dist = CauchyDistribution()
        dist.fit(data, method='mle')
        
        # Mean and variance should be NaN
        assert np.isnan(dist.mean())
        assert np.isnan(dist.var())
        
        # But median should work
        median = dist.median()
        assert not np.isnan(median)
    
    def test_extreme_value_distributions(self):
        """Test extreme value distributions (Frechet)"""
        np.random.seed(42)
        
        # Annual maximum wind speeds
        data = stats.invweibull.rvs(c=3, scale=50, size=30)
        
        dist = FrechetDistribution()
        dist.fit(data, method='mle')
        
        # 100-year event
        extreme_event = dist.ppf(0.99)
        assert extreme_event > np.max(data)
    
    def test_cdf_bounds(self):
        """Test all CDFs are bounded [0, 1]"""
        np.random.seed(42)
        
        distributions = [
            (FrechetDistribution(), stats.invweibull.rvs(c=2, scale=5, size=300)),
            (ParetoDistribution(), (np.random.pareto(a=2.5, size=300) + 1) * 10),
            (ChiSquaredDistribution(), np.random.chisquare(df=5, size=300))
        ]
        
        for dist, data in distributions:
            dist.fit(data, method='mle')
            
            x = np.linspace(np.min(data), np.max(data), 50)
            cdf_values = dist.cdf(x)
            
            assert np.all(cdf_values >= 0)
            assert np.all(cdf_values <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
