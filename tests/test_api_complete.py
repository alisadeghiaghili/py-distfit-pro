"""
Comprehensive Tests for High-Level API (Part 9/10)
===================================================

Tests for high-level API framework:
1. DistributionFitter (main API)
2. Auto-selection
3. One-line workflows
4. Result objects
5. Batch operations
6. Pipeline integration

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
import pickle
from scipy import stats
from distfit_pro import (
    DistributionFitter,
    fit,
    compare_distributions,
    find_best_distribution,
    FitResult,
    ComparisonResult
)
from distfit_pro.core.distributions import get_distribution


# ============================================================================
# TEST: DISTRIBUTION FITTER (MAIN API)
# ============================================================================

class TestDistributionFitter:
    """Comprehensive tests for DistributionFitter"""
    
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
        """Test DistributionFitter initialization"""
        fitter = DistributionFitter()
        assert fitter is not None
    
    def test_simple_fit(self, normal_data):
        """Test simple fit with specified distribution"""
        fitter = DistributionFitter()
        result = fitter.fit(data=normal_data, distribution='normal')
        
        assert isinstance(result, FitResult)
        assert result.distribution_name == 'normal'
        assert result.fitted is True
    
    def test_auto_selection(self, normal_data):
        """Test automatic distribution selection"""
        fitter = DistributionFitter()
        result = fitter.fit(
            data=normal_data,
            distributions=['normal', 'exponential', 'uniform'],
            auto_select=True
        )
        
        # Should select normal as best fit
        assert result.distribution_name == 'normal'
        assert result.is_best_fit is True
    
    def test_fit_with_gof_test(self, normal_data):
        """Test fit with GOF test"""
        fitter = DistributionFitter()
        result = fitter.fit(
            data=normal_data,
            distribution='normal',
            gof_test='ks'
        )
        
        assert hasattr(result, 'gof_result')
        assert result.gof_result is not None
        assert result.gof_result.p_value > 0.05
    
    def test_fit_with_bootstrap(self, normal_data):
        """Test fit with bootstrap CI"""
        fitter = DistributionFitter()
        result = fitter.fit(
            data=normal_data,
            distribution='normal',
            bootstrap=True,
            n_bootstrap=500,
            confidence_level=0.95
        )
        
        assert hasattr(result, 'confidence_intervals')
        assert result.confidence_intervals is not None
    
    def test_fit_with_visualization(self, normal_data):
        """Test fit with automatic visualization"""
        fitter = DistributionFitter()
        result = fitter.fit(
            data=normal_data,
            distribution='normal',
            visualize=True,
            plot_types=['qq', 'histogram', 'cdf']
        )
        
        assert hasattr(result, 'plots')
        assert len(result.plots) == 3
    
    def test_method_chaining(self, normal_data):
        """Test method chaining"""
        fitter = DistributionFitter()
        result = (fitter
                  .fit(normal_data, 'normal')
                  .run_gof_test('ks')
                  .calculate_bootstrap_ci(n_bootstrap=500)
                  .generate_plots(['qq', 'pp']))
        
        assert result.fitted is True
        assert result.gof_result is not None
        assert result.confidence_intervals is not None
        assert len(result.plots) >= 2
    
    def test_parameter_extraction(self, normal_data):
        """Test parameter extraction from result"""
        fitter = DistributionFitter()
        result = fitter.fit(normal_data, 'normal')
        
        params = result.get_parameters()
        assert 'loc' in params
        assert 'scale' in params
        assert 9 < params['loc'] < 11
        assert 1.5 < params['scale'] < 2.5
    
    def test_prediction(self, normal_data):
        """Test prediction with fitted distribution"""
        fitter = DistributionFitter()
        result = fitter.fit(normal_data, 'normal')
        
        # Predict probability
        prob = result.pdf(np.array([10]))
        assert prob[0] > 0
        
        # Predict CDF
        cdf_val = result.cdf(np.array([10]))
        assert 0 < cdf_val[0] < 1
    
    def test_sampling_from_fit(self, normal_data):
        """Test sampling from fitted distribution"""
        fitter = DistributionFitter()
        result = fitter.fit(normal_data, 'normal')
        
        samples = result.sample(size=100, random_state=42)
        assert len(samples) == 100
        # Samples should be similar to original data
        assert abs(np.mean(samples) - np.mean(normal_data)) < 1


# ============================================================================
# TEST: ONE-LINE WORKFLOWS
# ============================================================================

class TestOneLineWorkflows:
    """Tests for convenient one-line API functions"""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data"""
        np.random.seed(42)
        return np.random.gamma(shape=2, scale=2, size=500)
    
    def test_simple_fit_function(self, test_data):
        """Test simple fit() function"""
        result = fit(test_data, distribution='gamma')
        
        assert isinstance(result, FitResult)
        assert result.fitted is True
    
    def test_find_best_distribution(self, test_data):
        """Test find_best_distribution() function"""
        best_name, best_result = find_best_distribution(
            data=test_data,
            candidates=['normal', 'exponential', 'gamma', 'weibull'],
            criterion='aic'
        )
        
        # Gamma should be best (data is gamma-distributed)
        assert best_name == 'gamma'
        assert isinstance(best_result, FitResult)
    
    def test_compare_distributions_function(self, test_data):
        """Test compare_distributions() function"""
        comparison = compare_distributions(
            data=test_data,
            distributions=['normal', 'exponential', 'gamma'],
            metrics=['aic', 'bic', 'ks']
        )
        
        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.rankings) == 3
    
    def test_quick_fit_with_plot(self, test_data):
        """Test quick fit with visualization"""
        result = fit(test_data, 'gamma', plot=True)
        
        assert result.fitted is True
        assert hasattr(result, 'plots')
    
    def test_quick_bootstrap(self, test_data):
        """Test quick bootstrap CI"""
        result = fit(test_data, 'gamma', bootstrap=True, n_bootstrap=500)
        
        assert result.confidence_intervals is not None


# ============================================================================
# TEST: FIT RESULT
# ============================================================================

class TestFitResult:
    """Comprehensive tests for FitResult object"""
    
    @pytest.fixture
    def fit_result(self):
        """Create a fit result for testing"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=500)
        fitter = DistributionFitter()
        return fitter.fit(data, 'normal')
    
    def test_result_attributes(self, fit_result):
        """Test result has all expected attributes"""
        assert hasattr(fit_result, 'distribution_name')
        assert hasattr(fit_result, 'parameters')
        assert hasattr(fit_result, 'fitted')
        assert hasattr(fit_result, 'data')
        assert hasattr(fit_result, 'distribution')
    
    def test_result_summary(self, fit_result):
        """Test result summary generation"""
        summary = fit_result.summary()
        
        assert isinstance(summary, str)
        assert 'normal' in summary.lower()
        assert 'parameters' in summary.lower()
    
    def test_result_to_dict(self, fit_result):
        """Test result serialization to dict"""
        result_dict = fit_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'distribution_name' in result_dict
        assert 'parameters' in result_dict
    
    def test_result_from_dict(self, fit_result):
        """Test result deserialization from dict"""
        result_dict = fit_result.to_dict()
        restored = FitResult.from_dict(result_dict)
        
        assert restored.distribution_name == fit_result.distribution_name
        assert restored.fitted == fit_result.fitted
    
    def test_result_pickle(self, fit_result):
        """Test result can be pickled"""
        pickled = pickle.dumps(fit_result)
        restored = pickle.loads(pickled)
        
        assert restored.distribution_name == fit_result.distribution_name
    
    def test_result_repr(self, fit_result):
        """Test result string representation"""
        repr_str = repr(fit_result)
        
        assert 'FitResult' in repr_str
        assert 'normal' in repr_str.lower()
    
    def test_result_comparison(self):
        """Test result comparison"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=500)
        
        fitter = DistributionFitter()
        result1 = fitter.fit(data, 'normal')
        result2 = fitter.fit(data, 'exponential')
        
        # Should be able to compare based on AIC/BIC
        assert result1.is_better_than(result2, criterion='aic')


# ============================================================================
# TEST: COMPARISON RESULT
# ============================================================================

class TestComparisonResult:
    """Tests for ComparisonResult object"""
    
    @pytest.fixture
    def comparison_result(self):
        """Create a comparison result"""
        np.random.seed(42)
        data = np.random.gamma(2, 2, size=500)
        
        return compare_distributions(
            data=data,
            distributions=['normal', 'exponential', 'gamma', 'weibull'],
            metrics=['aic', 'bic']
        )
    
    def test_rankings(self, comparison_result):
        """Test distribution rankings"""
        rankings = comparison_result.rankings
        
        assert len(rankings) == 4
        # Gamma should be best
        assert rankings[0]['distribution'] == 'gamma'
    
    def test_get_best(self, comparison_result):
        """Test get best distribution"""
        best = comparison_result.get_best()
        
        assert best['distribution'] == 'gamma'
    
    def test_get_by_rank(self, comparison_result):
        """Test get distribution by rank"""
        second_best = comparison_result.get_by_rank(2)
        
        assert second_best is not None
        assert 'distribution' in second_best
    
    def test_comparison_table(self, comparison_result):
        """Test comparison table generation"""
        table = comparison_result.to_table()
        
        assert isinstance(table, str)
        assert 'gamma' in table.lower()
    
    def test_comparison_plot(self, comparison_result):
        """Test comparison visualization"""
        fig = comparison_result.plot(metric='aic')
        
        assert fig is not None


# ============================================================================
# TEST: BATCH OPERATIONS
# ============================================================================

class TestBatchOperations:
    """Tests for batch fitting operations"""
    
    @pytest.fixture
    def multiple_datasets(self):
        """Generate multiple datasets"""
        np.random.seed(42)
        return {
            'dataset1': np.random.normal(10, 2, 500),
            'dataset2': np.random.exponential(2, 500),
            'dataset3': np.random.gamma(2, 2, 500)
        }
    
    def test_batch_fit(self, multiple_datasets):
        """Test batch fitting"""
        fitter = DistributionFitter()
        results = fitter.fit_multiple(
            datasets=multiple_datasets,
            distributions=['normal', 'exponential', 'gamma']
        )
        
        assert len(results) == 3
        assert all(r.fitted for r in results.values())
    
    def test_batch_auto_select(self, multiple_datasets):
        """Test batch auto-selection"""
        fitter = DistributionFitter()
        results = fitter.auto_fit_multiple(
            datasets=multiple_datasets,
            candidates=['normal', 'exponential', 'gamma', 'weibull']
        )
        
        # Each should get best distribution
        assert results['dataset1'].distribution_name == 'normal'
        assert results['dataset2'].distribution_name == 'exponential'
        assert results['dataset3'].distribution_name == 'gamma'
    
    def test_parallel_fitting(self, multiple_datasets):
        """Test parallel fitting"""
        fitter = DistributionFitter(n_jobs=2)
        results = fitter.fit_multiple(
            datasets=multiple_datasets,
            distributions=['normal'],
            parallel=True
        )
        
        assert len(results) == 3


# ============================================================================
# TEST: PIPELINE INTEGRATION
# ============================================================================

class TestPipelineIntegration:
    """Tests for pipeline integration"""
    
    def test_complete_pipeline(self):
        """Test complete analysis pipeline"""
        np.random.seed(42)
        data = np.random.weibull(2, size=500) * 5
        
        # 1. Find best distribution
        best_name, best_result = find_best_distribution(
            data=data,
            candidates=['normal', 'exponential', 'weibull', 'gamma']
        )
        
        # 2. Run GOF test
        best_result.run_gof_test('ks')
        
        # 3. Calculate bootstrap CI
        best_result.calculate_bootstrap_ci(n_bootstrap=500)
        
        # 4. Generate plots
        best_result.generate_plots(['qq', 'histogram'])
        
        # All should be successful
        assert best_result.fitted is True
        assert best_result.gof_result is not None
        assert best_result.confidence_intervals is not None
        assert len(best_result.plots) >= 2
    
    def test_sklearn_style_api(self):
        """Test scikit-learn style API"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=500)
        
        fitter = DistributionFitter()
        
        # Fit
        fitter.fit(data, 'normal')
        
        # Transform (to standard normal)
        transformed = fitter.transform(data)
        
        # Inverse transform
        restored = fitter.inverse_transform(transformed)
        
        assert np.allclose(data, restored, atol=0.01)
    
    def test_cross_validation(self):
        """Test cross-validation for distribution selection"""
        np.random.seed(42)
        data = np.random.gamma(2, 2, size=500)
        
        fitter = DistributionFitter()
        cv_results = fitter.cross_validate(
            data=data,
            distributions=['normal', 'exponential', 'gamma'],
            cv=5
        )
        
        # Gamma should have best CV score
        assert cv_results['gamma']['mean_score'] > cv_results['normal']['mean_score']


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and validation"""
    
    def test_empty_data(self):
        """Test handling of empty data"""
        fitter = DistributionFitter()
        
        with pytest.raises(ValueError):
            fitter.fit(data=np.array([]), distribution='normal')
    
    def test_invalid_distribution(self):
        """Test handling of invalid distribution name"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)
        
        fitter = DistributionFitter()
        
        with pytest.raises(ValueError):
            fitter.fit(data=data, distribution='invalid_dist')
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        fitter = DistributionFitter()
        small_data = np.array([1, 2, 3])
        
        with pytest.warns(UserWarning):
            result = fitter.fit(small_data, 'normal')
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)
        data[::10] = np.nan
        
        fitter = DistributionFitter(handle_nan='remove')
        result = fitter.fit(data, 'normal')
        
        assert result.fitted is True
    
    def test_infinite_values(self):
        """Test handling of infinite values"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)
        data[0] = np.inf
        
        fitter = DistributionFitter(handle_inf='remove')
        result = fitter.fit(data, 'normal')
        
        assert result.fitted is True
    
    def test_negative_data_for_positive_dist(self):
        """Test handling of negative data for positive distributions"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)  # Contains negative values
        
        fitter = DistributionFitter()
        
        with pytest.raises(ValueError):
            fitter.fit(data, 'exponential')
    
    def test_graceful_failure(self):
        """Test graceful failure in auto-selection"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)
        
        fitter = DistributionFitter()
        result = fitter.fit(
            data=data,
            distributions=['normal', 'invalid_dist', 'exponential'],
            auto_select=True,
            handle_errors='skip'
        )
        
        # Should succeed with valid distributions
        assert result.fitted is True


# ============================================================================
# TEST: PERFORMANCE
# ============================================================================

class TestPerformance:
    """Tests for performance optimization"""
    
    def test_caching(self):
        """Test result caching"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=1000)
        
        fitter = DistributionFitter(cache=True)
        
        # First fit
        result1 = fitter.fit(data, 'normal')
        
        # Second fit (should use cache)
        result2 = fitter.fit(data, 'normal')
        
        assert result1.parameters == result2.parameters
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        np.random.seed(42)
        large_data = np.random.normal(0, 1, size=100000)
        
        fitter = DistributionFitter()
        result = fitter.fit(large_data, 'normal')
        
        assert result.fitted is True
    
    def test_memory_efficiency(self):
        """Test memory-efficient operations"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=10000)
        
        fitter = DistributionFitter(memory_efficient=True)
        result = fitter.fit(
            data=data,
            distribution='normal',
            bootstrap=True,
            n_bootstrap=1000
        )
        
        assert result.fitted is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAPIIntegration:
    """Integration tests for complete API"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, size=500)
        
        # Step 1: Find best distribution
        comparison = compare_distributions(
            data=data,
            distributions=['normal', 'lognormal', 'gamma', 'weibull'],
            metrics=['aic', 'bic', 'ks']
        )
        
        best_dist = comparison.get_best()['distribution']
        assert best_dist == 'lognormal'
        
        # Step 2: Fit best distribution with all features
        fitter = DistributionFitter()
        result = fitter.fit(
            data=data,
            distribution=best_dist,
            gof_test='ks',
            bootstrap=True,
            n_bootstrap=500,
            visualize=True,
            plot_types=['qq', 'histogram', 'cdf']
        )
        
        # Step 3: Verify all components
        assert result.fitted is True
        assert result.gof_result.p_value > 0.05
        assert result.confidence_intervals is not None
        assert len(result.plots) == 3
        
        # Step 4: Use for prediction
        new_samples = result.sample(size=100, random_state=42)
        assert len(new_samples) == 100
        
        # Step 5: Export results
        summary = result.summary()
        assert len(summary) > 100
    
    def test_api_consistency(self):
        """Test API consistency across methods"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=500)
        
        # Method 1: DistributionFitter class
        fitter = DistributionFitter()
        result1 = fitter.fit(data, 'normal')
        
        # Method 2: Convenience function
        result2 = fit(data, 'normal')
        
        # Results should be equivalent
        assert result1.distribution_name == result2.distribution_name
        assert np.allclose(
            list(result1.parameters.values()),
            list(result2.parameters.values()),
            rtol=0.01
        )
    
    def test_real_world_scenario(self):
        """Test real-world analysis scenario"""
        np.random.seed(42)
        
        # Simulate real-world data (e.g., response times)
        response_times = np.random.lognormal(1, 0.5, size=1000)
        
        # Analyst workflow
        # 1. Quick exploration
        quick_fit = fit(response_times, 'lognormal', plot=True)
        assert quick_fit.fitted is True
        
        # 2. Comprehensive comparison
        comparison = compare_distributions(
            data=response_times,
            distributions=['normal', 'lognormal', 'gamma', 'weibull'],
            metrics=['aic', 'bic']
        )
        best_name = comparison.get_best()['distribution']
        
        # 3. Detailed analysis of best fit
        fitter = DistributionFitter()
        final_result = fitter.fit(
            data=response_times,
            distribution=best_name,
            gof_test='ks',
            bootstrap=True,
            n_bootstrap=1000,
            confidence_level=0.95
        )
        
        # 4. Generate report
        report = final_result.summary()
        
        assert best_name == 'lognormal'
        assert final_result.gof_result.p_value > 0.05
        assert len(report) > 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
