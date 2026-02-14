"""
Comprehensive Test Suite for BaseDistribution
==============================================

Tests all features of base.py to ensure correct implementation:
- Fitting (MLE & MOM)
- Probability functions (PDF, CDF, PPF, SF, ISF)
- Statistics (mean, var, std, median, mode, skewness, kurtosis)
- Model selection (AIC, BIC, log-likelihood)
- Random sampling
- Summary and explain
- i18n support
- Error handling
- Data validation

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from distfit_pro.core.distributions import get_distribution, list_distributions
from distfit_pro.core.config import set_language, config
from distfit_pro.locales import t


class TestFitting:
    """Test distribution fitting methods"""
    
    def test_mle_fitting(self):
        """Test Maximum Likelihood Estimation"""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        
        dist = get_distribution('normal')
        dist.fit(data, method='mle')
        
        assert dist.fitted
        assert 'loc' in dist.params
        assert 'scale' in dist.params
        assert_almost_equal(dist.params['loc'], 10, decimal=1)
        assert_almost_equal(dist.params['scale'], 2, decimal=1)
    
    def test_mom_fitting(self):
        """Test Method of Moments"""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        
        dist = get_distribution('normal')
        dist.fit(data, method='mom')
        
        assert dist.fitted
        assert_almost_equal(dist.params['loc'], np.mean(data), decimal=6)
        assert_almost_equal(dist.params['scale'], np.std(data, ddof=1), decimal=6)
    
    def test_method_chaining(self):
        """Test that fit() returns self for chaining"""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        
        dist = get_distribution('normal')
        result = dist.fit(data)
        
        assert result is dist
        assert dist.fitted
    
    def test_fitting_preserves_data(self):
        """Test that original data is stored"""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        
        dist = get_distribution('normal')
        dist.fit(data)
        
        assert dist.data is not None
        assert len(dist.data) == len(data)
        assert_array_almost_equal(dist.data, data)


class TestDataValidation:
    """Test data validation and error handling"""
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError"""
        dist = get_distribution('normal')
        
        with pytest.raises(ValueError, match="empty"):
            dist.fit([])
    
    def test_nan_data_raises_error(self):
        """Test that NaN values raise ValueError"""
        dist = get_distribution('normal')
        data = np.array([1, 2, np.nan, 4, 5])
        
        with pytest.raises(ValueError, match="NaN"):
            dist.fit(data)
    
    def test_inf_data_raises_error(self):
        """Test that infinite values raise ValueError"""
        dist = get_distribution('normal')
        data = np.array([1, 2, np.inf, 4, 5])
        
        with pytest.raises(ValueError, match="infinite"):
            dist.fit(data)
    
    def test_invalid_method_raises_error(self):
        """Test that invalid fitting method raises ValueError"""
        dist = get_distribution('normal')
        data = np.random.normal(0, 1, 100)
        
        with pytest.raises(ValueError, match="Unknown method"):
            dist.fit(data, method='invalid_method')


class TestProbabilityFunctions:
    """Test probability density and distribution functions"""
    
    @pytest.fixture
    def fitted_normal(self):
        """Fixture providing a fitted normal distribution"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        return dist
    
    def test_pdf(self, fitted_normal):
        """Test probability density function"""
        x = np.array([0, 1, -1])
        pdf_vals = fitted_normal.pdf(x)
        
        assert pdf_vals.shape == x.shape
        assert np.all(pdf_vals > 0)
        assert pdf_vals[0] > pdf_vals[1]  # PDF highest at mean
    
    def test_logpdf(self, fitted_normal):
        """Test log probability density function"""
        x = np.array([0, 1, -1])
        logpdf_vals = fitted_normal.logpdf(x)
        pdf_vals = fitted_normal.pdf(x)
        
        assert_array_almost_equal(logpdf_vals, np.log(pdf_vals))
    
    def test_cdf(self, fitted_normal):
        """Test cumulative distribution function"""
        x = np.array([-3, 0, 3])
        cdf_vals = fitted_normal.cdf(x)
        
        assert cdf_vals.shape == x.shape
        assert np.all((cdf_vals >= 0) & (cdf_vals <= 1))
        assert cdf_vals[0] < cdf_vals[1] < cdf_vals[2]
        assert_almost_equal(cdf_vals[1], 0.5, decimal=1)  # CDF(mean) ≈ 0.5
    
    def test_ppf(self, fitted_normal):
        """Test percent point function (inverse CDF)"""
        q = np.array([0.1, 0.5, 0.9])
        ppf_vals = fitted_normal.ppf(q)
        
        assert ppf_vals.shape == q.shape
        assert ppf_vals[0] < ppf_vals[1] < ppf_vals[2]
        
        # Test that PPF inverts CDF
        cdf_vals = fitted_normal.cdf(ppf_vals)
        assert_array_almost_equal(cdf_vals, q, decimal=4)
    
    def test_sf(self, fitted_normal):
        """Test survival function"""
        x = np.array([-3, 0, 3])
        sf_vals = fitted_normal.sf(x)
        cdf_vals = fitted_normal.cdf(x)
        
        assert_array_almost_equal(sf_vals, 1 - cdf_vals, decimal=10)
    
    def test_isf(self, fitted_normal):
        """Test inverse survival function"""
        q = np.array([0.1, 0.5, 0.9])
        isf_vals = fitted_normal.isf(q)
        ppf_vals = fitted_normal.ppf(1 - q)
        
        assert_array_almost_equal(isf_vals, ppf_vals, decimal=10)
    
    def test_unfitted_raises_error(self):
        """Test that calling methods before fit() raises error"""
        dist = get_distribution('normal')
        
        with pytest.raises(ValueError, match="Must call fit"):
            dist.pdf(np.array([0]))
        
        with pytest.raises(ValueError, match="Must call fit"):
            dist.cdf(np.array([0]))


class TestStatistics:
    """Test statistical moments and properties"""
    
    @pytest.fixture
    def fitted_normal(self):
        """Fixture providing a fitted normal distribution"""
        np.random.seed(42)
        data = np.random.normal(5, 2, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        return dist
    
    def test_mean(self, fitted_normal):
        """Test mean calculation"""
        mean_val = fitted_normal.mean()
        assert_almost_equal(mean_val, 5, decimal=1)
    
    def test_variance(self, fitted_normal):
        """Test variance calculation"""
        var_val = fitted_normal.var()
        assert_almost_equal(var_val, 4, decimal=0)  # 2^2 = 4
    
    def test_std(self, fitted_normal):
        """Test standard deviation calculation"""
        std_val = fitted_normal.std()
        assert_almost_equal(std_val, 2, decimal=1)
    
    def test_median(self, fitted_normal):
        """Test median calculation"""
        median_val = fitted_normal.median()
        mean_val = fitted_normal.mean()
        # For normal distribution, median ≈ mean
        assert_almost_equal(median_val, mean_val, decimal=2)
    
    def test_mode(self, fitted_normal):
        """Test mode calculation"""
        mode_val = fitted_normal.mode()
        mean_val = fitted_normal.mean()
        # For normal distribution, mode = mean
        assert_almost_equal(mode_val, mean_val, decimal=6)
    
    def test_skewness(self, fitted_normal):
        """Test skewness calculation"""
        skew_val = fitted_normal.skewness()
        # Normal distribution has zero skewness
        assert_almost_equal(skew_val, 0, decimal=1)
    
    def test_kurtosis(self, fitted_normal):
        """Test kurtosis calculation"""
        kurt_val = fitted_normal.kurtosis()
        # Normal distribution has zero excess kurtosis
        assert_almost_equal(kurt_val, 0, decimal=1)
    
    def test_entropy(self, fitted_normal):
        """Test differential entropy"""
        entropy_val = fitted_normal.entropy()
        assert entropy_val > 0


class TestModelSelection:
    """Test model selection criteria"""
    
    @pytest.fixture
    def fitted_dist_with_data(self):
        """Fixture with fitted distribution and data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        dist = get_distribution('normal')
        dist.fit(data)
        return dist, data
    
    def test_log_likelihood(self, fitted_dist_with_data):
        """Test log-likelihood calculation"""
        dist, data = fitted_dist_with_data
        ll = dist.log_likelihood()
        
        assert isinstance(ll, (int, float))
        assert ll < 0  # Log-likelihood is typically negative
    
    def test_log_likelihood_with_custom_data(self, fitted_dist_with_data):
        """Test log-likelihood with custom data"""
        dist, original_data = fitted_dist_with_data
        new_data = np.random.normal(0, 1, 100)
        
        ll = dist.log_likelihood(new_data)
        assert isinstance(ll, (int, float))
    
    def test_aic(self, fitted_dist_with_data):
        """Test Akaike Information Criterion"""
        dist, data = fitted_dist_with_data
        aic_val = dist.aic()
        
        assert isinstance(aic_val, (int, float))
        # AIC = 2k - 2*log(L), should be positive for most cases
        assert aic_val > 0
    
    def test_bic(self, fitted_dist_with_data):
        """Test Bayesian Information Criterion"""
        dist, data = fitted_dist_with_data
        bic_val = dist.bic()
        
        assert isinstance(bic_val, (int, float))
        assert bic_val > 0
    
    def test_bic_penalizes_more_than_aic(self, fitted_dist_with_data):
        """Test that BIC penalizes complexity more than AIC"""
        dist, data = fitted_dist_with_data
        aic_val = dist.aic()
        bic_val = dist.bic()
        
        # For large samples, BIC > AIC
        if len(data) > 10:
            assert bic_val > aic_val


class TestRandomSampling:
    """Test random sample generation"""
    
    def test_rvs_basic(self):
        """Test basic random variate generation"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        samples = dist.rvs(size=100, random_state=42)
        
        assert len(samples) == 100
        assert isinstance(samples, np.ndarray)
    
    def test_rvs_reproducibility(self):
        """Test that random_state ensures reproducibility"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        samples1 = dist.rvs(size=100, random_state=42)
        samples2 = dist.rvs(size=100, random_state=42)
        
        assert_array_almost_equal(samples1, samples2)
    
    def test_rvs_matches_distribution(self):
        """Test that generated samples match fitted distribution"""
        np.random.seed(42)
        data = np.random.normal(5, 2, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        samples = dist.rvs(size=10000, random_state=42)
        
        # Check that sample statistics are close to fitted parameters
        assert_almost_equal(np.mean(samples), 5, decimal=1)
        assert_almost_equal(np.std(samples), 2, decimal=1)


class TestSummaryAndExplain:
    """Test output formatting methods"""
    
    def test_summary_unfitted(self):
        """Test summary for unfitted distribution"""
        dist = get_distribution('normal')
        summary = dist.summary()
        
        assert 'not fitted' in summary.lower()
        assert 'Normal' in summary
    
    def test_summary_fitted(self):
        """Test summary for fitted distribution"""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        summary = dist.summary()
        
        assert '╔' in summary and '╗' in summary  # Box characters
        assert 'Normal' in summary
        assert 'μ' in summary or 'loc' in summary
        assert 'σ' in summary or 'scale' in summary
        assert len(summary.split('\n')) > 10  # Multiple lines
    
    def test_explain(self):
        """Test explain method"""
        dist = get_distribution('normal')
        explanation = dist.explain()
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'normal' in explanation.lower() or 'gaussian' in explanation.lower()
    
    def test_str_representation(self):
        """Test __str__ returns summary"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        dist = get_distribution('normal')
        dist.fit(data)
        
        str_repr = str(dist)
        summary = dist.summary()
        
        assert str_repr == summary
    
    def test_repr(self):
        """Test __repr__ representation"""
        dist = get_distribution('normal')
        repr_str = repr(dist)
        
        assert 'Normal' in repr_str
        assert 'not fitted' in repr_str
        
        dist.fit(np.random.normal(0, 1, 100))
        repr_str = repr(dist)
        
        assert 'fitted' in repr_str


class TestInternationalization:
    """Test i18n support"""
    
    def test_summary_english(self):
        """Test summary in English"""
        set_language('en')
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        summary = dist.summary()
        
        assert 'Mean' in summary or 'LOCATION' in summary
        assert 'Std' in summary or 'SPREAD' in summary
    
    def test_summary_persian(self):
        """Test summary in Persian"""
        set_language('fa')
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        summary = dist.summary()
        
        # Check for Persian text (RTL characters)
        assert any(ord(c) > 1536 for c in summary)  # Persian Unicode range
    
    def test_summary_german(self):
        """Test summary in German"""
        set_language('de')
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        summary = dist.summary()
        
        # Reset to English
        set_language('en')
        
        assert 'Mittelwert' in summary or 'LAGE' in summary


class TestParameterManagement:
    """Test parameter access and manipulation"""
    
    def test_params_before_fit_raises_error(self):
        """Test that accessing params before fit raises error"""
        dist = get_distribution('normal')
        
        with pytest.raises(ValueError, match="not fitted"):
            _ = dist.params
    
    def test_params_after_fit(self):
        """Test params property after fitting"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        params = dist.params
        
        assert isinstance(params, dict)
        assert 'loc' in params
        assert 'scale' in params
    
    def test_params_returns_copy(self):
        """Test that params returns a copy, not reference"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        dist = get_distribution('normal')
        dist.fit(data)
        
        params1 = dist.params
        params2 = dist.params
        
        assert params1 is not params2
        assert params1 == params2
    
    def test_manual_parameter_setting(self):
        """Test setting parameters manually"""
        dist = get_distribution('normal')
        dist.params = {'loc': 5.0, 'scale': 2.0}
        
        assert dist.fitted
        assert dist.params['loc'] == 5.0
        assert dist.params['scale'] == 2.0
    
    def test_fitted_property(self):
        """Test fitted property"""
        dist = get_distribution('normal')
        
        assert not dist.fitted
        
        dist.fit(np.random.normal(0, 1, 100))
        
        assert dist.fitted


class TestDistributionRegistry:
    """Test distribution registry and factory"""
    
    def test_list_distributions(self):
        """Test listing all available distributions"""
        dists = list_distributions()
        
        assert isinstance(dists, list)
        assert len(dists) > 0
        assert 'normal' in dists
        assert 'exponential' in dists
    
    def test_get_distribution_valid(self):
        """Test getting a valid distribution"""
        dist = get_distribution('normal')
        
        assert dist is not None
        assert dist.info.name == 'normal'
    
    def test_get_distribution_invalid(self):
        """Test getting invalid distribution raises error"""
        with pytest.raises((ValueError, KeyError)):
            get_distribution('nonexistent_distribution')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
