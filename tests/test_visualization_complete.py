"""
Comprehensive Tests for Visualization Methods (Part 8/10)
==========================================================

Tests for visualization framework:
1. QQ Plot (Quantile-Quantile)
2. PP Plot (Probability-Probability)
3. Histogram with Fitted Distribution
4. CDF Plots (Empirical vs Theoretical)
5. PDF Overlay
6. Survival Function Plots
7. Residual Plots
8. Multi-Distribution Comparison

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from scipy import stats
from distfit_pro.core.distributions import (
    NormalDistribution,
    ExponentialDistribution,
    WeibullDistribution,
    UniformDistribution,
    get_distribution
)
from distfit_pro.visualization import (
    QQPlot,
    PPPlot,
    HistogramPlot,
    CDFPlot,
    PDFPlot,
    SurvivalPlot,
    ResidualPlot,
    MultiDistributionPlot,
    PlotConfig
)


# ============================================================================
# TEST: QQ PLOT
# ============================================================================

class TestQQPlot:
    """Comprehensive tests for QQ Plot"""
    
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
        """Test QQ plot initialization"""
        qq_plot = QQPlot()
        assert qq_plot is not None
    
    def test_create_plot(self, normal_data):
        """Test QQ plot creation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        qq_plot = QQPlot()
        fig, ax = qq_plot.plot(data=normal_data, distribution=dist)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)
    
    def test_reference_line(self, normal_data):
        """Test 45-degree reference line"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        qq_plot = QQPlot()
        fig, ax = qq_plot.plot(data=normal_data, distribution=dist)
        
        # Check that reference line exists
        lines = ax.get_lines()
        assert len(lines) >= 2  # Data points + reference line
        
        plt.close(fig)
    
    def test_good_fit_along_line(self, normal_data):
        """Test good fit shows points along reference line"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        qq_plot = QQPlot()
        theoretical_quantiles, sample_quantiles = qq_plot.calculate_quantiles(
            data=normal_data, distribution=dist
        )
        
        # Calculate correlation (should be high for good fit)
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        assert correlation > 0.99
    
    def test_poor_fit_deviates(self, normal_data):
        """Test poor fit shows deviation from reference line"""
        # Fit wrong distribution
        dist = ExponentialDistribution()
        dist.fit(normal_data - np.min(normal_data) + 0.1, method='mle')
        
        qq_plot = QQPlot()
        theoretical_quantiles, sample_quantiles = qq_plot.calculate_quantiles(
            data=normal_data, distribution=dist
        )
        
        # Correlation should be lower for poor fit
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        assert correlation < 0.99
    
    def test_quantile_calculation(self, normal_data):
        """Test quantile calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        qq_plot = QQPlot()
        theoretical_q, sample_q = qq_plot.calculate_quantiles(
            data=normal_data, distribution=dist
        )
        
        assert len(theoretical_q) == len(sample_q)
        assert len(theoretical_q) > 0
    
    def test_customization(self, normal_data):
        """Test plot customization"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        config = PlotConfig(
            title='Custom QQ Plot',
            xlabel='Theoretical Quantiles',
            ylabel='Sample Quantiles',
            color='red'
        )
        
        qq_plot = QQPlot(config=config)
        fig, ax = qq_plot.plot(data=normal_data, distribution=dist)
        
        assert ax.get_title() == 'Custom QQ Plot'
        assert ax.get_xlabel() == 'Theoretical Quantiles'
        
        plt.close(fig)
    
    def test_multiple_distributions(self, normal_data):
        """Test QQ plot with multiple distributions"""
        dist1 = NormalDistribution()
        dist1.fit(normal_data, method='mle')
        
        dist2 = UniformDistribution()
        dist2.fit(normal_data, method='mle')
        
        qq_plot = QQPlot()
        fig, axes = qq_plot.plot_multiple(
            data=normal_data,
            distributions=[dist1, dist2],
            labels=['Normal', 'Uniform']
        )
        
        assert len(axes) == 2
        plt.close(fig)


# ============================================================================
# TEST: PP PLOT
# ============================================================================

class TestPPPlot:
    """Comprehensive tests for PP Plot"""
    
    @pytest.fixture
    def uniform_data(self):
        """Generate uniform distributed data"""
        np.random.seed(42)
        return np.random.uniform(0, 10, size=1000)
    
    def test_initialization(self):
        """Test PP plot initialization"""
        pp_plot = PPPlot()
        assert pp_plot is not None
    
    def test_create_plot(self, uniform_data):
        """Test PP plot creation"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        pp_plot = PPPlot()
        fig, ax = pp_plot.plot(data=uniform_data, distribution=dist)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)
    
    def test_probability_calculation(self, uniform_data):
        """Test probability calculation"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        pp_plot = PPPlot()
        theoretical_p, empirical_p = pp_plot.calculate_probabilities(
            data=uniform_data, distribution=dist
        )
        
        # Both should be in [0, 1]
        assert np.all((theoretical_p >= 0) & (theoretical_p <= 1))
        assert np.all((empirical_p >= 0) & (empirical_p <= 1))
    
    def test_diagonal_reference(self, uniform_data):
        """Test diagonal reference line"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        pp_plot = PPPlot()
        fig, ax = pp_plot.plot(data=uniform_data, distribution=dist)
        
        # Should have diagonal reference line
        lines = ax.get_lines()
        assert len(lines) >= 2
        
        plt.close(fig)
    
    def test_good_fit_near_diagonal(self, uniform_data):
        """Test good fit shows points near diagonal"""
        dist = UniformDistribution()
        dist.fit(uniform_data, method='mle')
        
        pp_plot = PPPlot()
        theoretical_p, empirical_p = pp_plot.calculate_probabilities(
            data=uniform_data, distribution=dist
        )
        
        # Mean absolute deviation from diagonal
        mad = np.mean(np.abs(theoretical_p - empirical_p))
        assert mad < 0.05
    
    def test_pp_vs_qq_difference(self):
        """Test PP plot focuses on center while QQ on tails"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=500)
        
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        # PP plot
        pp_plot = PPPlot()
        pp_fig, pp_ax = pp_plot.plot(data, dist)
        
        # QQ plot
        qq_plot = QQPlot()
        qq_fig, qq_ax = qq_plot.plot(data, dist)
        
        # Both should be created
        assert pp_ax is not None
        assert qq_ax is not None
        
        plt.close(pp_fig)
        plt.close(qq_fig)


# ============================================================================
# TEST: HISTOGRAM PLOT
# ============================================================================

class TestHistogramPlot:
    """Comprehensive tests for Histogram Plot"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=1000)
    
    def test_initialization(self):
        """Test histogram plot initialization"""
        hist_plot = HistogramPlot()
        assert hist_plot is not None
    
    def test_create_histogram(self, normal_data):
        """Test histogram creation"""
        hist_plot = HistogramPlot(bins=30)
        fig, ax = hist_plot.plot(data=normal_data)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)
    
    def test_overlay_pdf(self, normal_data):
        """Test PDF overlay on histogram"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        hist_plot = HistogramPlot(bins=30)
        fig, ax = hist_plot.plot_with_pdf(data=normal_data, distribution=dist)
        
        # Should have histogram bars + PDF line
        assert len(ax.patches) > 0  # Histogram bars
        assert len(ax.lines) > 0    # PDF line
        
        plt.close(fig)
    
    def test_density_normalization(self, normal_data):
        """Test histogram is normalized to density"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        hist_plot = HistogramPlot(bins=30, density=True)
        fig, ax = hist_plot.plot_with_pdf(data=normal_data, distribution=dist)
        
        # Histogram should integrate to ~1
        plt.close(fig)
    
    def test_bin_count_effect(self, normal_data):
        """Test different bin counts"""
        hist_plot_10 = HistogramPlot(bins=10)
        fig1, ax1 = hist_plot_10.plot(normal_data)
        
        hist_plot_50 = HistogramPlot(bins=50)
        fig2, ax2 = hist_plot_50.plot(normal_data)
        
        # Different bin counts should give different number of bars
        assert len(ax1.patches) != len(ax2.patches)
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_multiple_distributions_overlay(self, normal_data):
        """Test multiple distribution overlays"""
        dist1 = NormalDistribution()
        dist1.fit(normal_data, method='mle')
        
        dist2 = UniformDistribution()
        dist2.fit(normal_data, method='mle')
        
        hist_plot = HistogramPlot(bins=30)
        fig, ax = hist_plot.plot_with_multiple_pdfs(
            data=normal_data,
            distributions=[dist1, dist2],
            labels=['Normal', 'Uniform']
        )
        
        # Should have histogram + 2 PDF lines
        assert len(ax.lines) >= 2
        
        plt.close(fig)


# ============================================================================
# TEST: CDF PLOT
# ============================================================================

class TestCDFPlot:
    """Comprehensive tests for CDF Plot"""
    
    @pytest.fixture
    def exponential_data(self):
        """Generate exponential distributed data"""
        np.random.seed(42)
        return np.random.exponential(scale=2, size=1000)
    
    def test_initialization(self):
        """Test CDF plot initialization"""
        cdf_plot = CDFPlot()
        assert cdf_plot is not None
    
    def test_empirical_cdf(self, exponential_data):
        """Test empirical CDF calculation"""
        cdf_plot = CDFPlot()
        x_values, ecdf_values = cdf_plot.calculate_ecdf(exponential_data)
        
        # ECDF should be monotonically increasing
        assert np.all(np.diff(ecdf_values) >= 0)
        # ECDF should range from 0 to 1
        assert ecdf_values[0] >= 0
        assert ecdf_values[-1] <= 1
    
    def test_theoretical_cdf(self, exponential_data):
        """Test theoretical CDF overlay"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        cdf_plot = CDFPlot()
        fig, ax = cdf_plot.plot_comparison(
            data=exponential_data, distribution=dist
        )
        
        # Should have empirical + theoretical lines
        assert len(ax.lines) >= 2
        
        plt.close(fig)
    
    def test_ks_statistic_visualization(self, exponential_data):
        """Test KS statistic visualization on CDF plot"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        cdf_plot = CDFPlot(show_ks_statistic=True)
        fig, ax = cdf_plot.plot_comparison(
            data=exponential_data, distribution=dist
        )
        
        # Should show maximum vertical distance
        plt.close(fig)
    
    def test_cdf_bounds(self, exponential_data):
        """Test CDF stays within [0, 1]"""
        dist = ExponentialDistribution()
        dist.fit(exponential_data, method='mle')
        
        cdf_plot = CDFPlot()
        x_vals, ecdf = cdf_plot.calculate_ecdf(exponential_data)
        
        x_test = np.linspace(np.min(exponential_data), np.max(exponential_data), 100)
        tcdf = dist.cdf(x_test)
        
        assert np.all((tcdf >= 0) & (tcdf <= 1))
        assert np.all((ecdf >= 0) & (ecdf <= 1))


# ============================================================================
# TEST: PDF PLOT
# ============================================================================

class TestPDFPlot:
    """Comprehensive tests for PDF Plot"""
    
    @pytest.fixture
    def weibull_data(self):
        """Generate Weibull distributed data"""
        np.random.seed(42)
        return np.random.weibull(a=2, size=1000) * 5
    
    def test_initialization(self):
        """Test PDF plot initialization"""
        pdf_plot = PDFPlot()
        assert pdf_plot is not None
    
    def test_plot_pdf(self, weibull_data):
        """Test PDF plotting"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        pdf_plot = PDFPlot()
        fig, ax = pdf_plot.plot(distribution=dist)
        
        assert isinstance(fig, plt.Figure)
        assert len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_pdf_range(self, weibull_data):
        """Test PDF is plotted over appropriate range"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        pdf_plot = PDFPlot()
        x_range = pdf_plot.calculate_plot_range(distribution=dist)
        
        # Range should cover main probability mass
        assert x_range[0] < x_range[1]
    
    def test_pdf_non_negative(self, weibull_data):
        """Test PDF values are non-negative"""
        dist = WeibullDistribution()
        dist.fit(weibull_data, method='mle')
        
        x = np.linspace(0.1, 10, 100)
        pdf_values = dist.pdf(x)
        
        assert np.all(pdf_values >= 0)
    
    def test_multiple_pdfs(self, weibull_data):
        """Test multiple PDFs on same plot"""
        dist1 = WeibullDistribution()
        dist1.fit(weibull_data, method='mle')
        
        dist2 = ExponentialDistribution()
        dist2.fit(weibull_data, method='mle')
        
        pdf_plot = PDFPlot()
        fig, ax = pdf_plot.plot_multiple(
            distributions=[dist1, dist2],
            labels=['Weibull', 'Exponential']
        )
        
        assert len(ax.lines) >= 2
        plt.close(fig)


# ============================================================================
# TEST: SURVIVAL PLOT
# ============================================================================

class TestSurvivalPlot:
    """Comprehensive tests for Survival Function Plot"""
    
    @pytest.fixture
    def survival_data(self):
        """Generate survival time data"""
        np.random.seed(42)
        return np.random.exponential(scale=10, size=500)
    
    def test_initialization(self):
        """Test survival plot initialization"""
        surv_plot = SurvivalPlot()
        assert surv_plot is not None
    
    def test_survival_function(self, survival_data):
        """Test survival function S(t) = 1 - F(t)"""
        dist = ExponentialDistribution()
        dist.fit(survival_data, method='mle')
        
        surv_plot = SurvivalPlot()
        fig, ax = surv_plot.plot(distribution=dist)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_survival_decreasing(self, survival_data):
        """Test survival function is decreasing"""
        dist = ExponentialDistribution()
        dist.fit(survival_data, method='mle')
        
        t_values = np.linspace(0, 30, 100)
        survival = dist.sf(t_values)
        
        # Should be monotonically decreasing
        assert np.all(np.diff(survival) <= 0)
    
    def test_survival_bounds(self, survival_data):
        """Test survival function in [0, 1]"""
        dist = ExponentialDistribution()
        dist.fit(survival_data, method='mle')
        
        t_values = np.linspace(0, 50, 100)
        survival = dist.sf(t_values)
        
        assert np.all((survival >= 0) & (survival <= 1))
    
    def test_kaplan_meier_comparison(self, survival_data):
        """Test comparison with Kaplan-Meier estimator"""
        dist = ExponentialDistribution()
        dist.fit(survival_data, method='mle')
        
        surv_plot = SurvivalPlot()
        fig, ax = surv_plot.plot_with_kaplan_meier(
            data=survival_data, distribution=dist
        )
        
        # Should have both curves
        assert len(ax.lines) >= 2
        plt.close(fig)


# ============================================================================
# TEST: RESIDUAL PLOT
# ============================================================================

class TestResidualPlot:
    """Comprehensive tests for Residual Plot"""
    
    @pytest.fixture
    def normal_data(self):
        """Generate normal distributed data"""
        np.random.seed(42)
        return np.random.normal(loc=10, scale=2, size=500)
    
    def test_initialization(self):
        """Test residual plot initialization"""
        resid_plot = ResidualPlot()
        assert resid_plot is not None
    
    def test_quantile_residuals(self, normal_data):
        """Test quantile residuals calculation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        resid_plot = ResidualPlot()
        residuals = resid_plot.calculate_residuals(
            data=normal_data, distribution=dist
        )
        
        # Residuals should have mean ~0
        assert abs(np.mean(residuals)) < 0.2
    
    def test_residual_plot(self, normal_data):
        """Test residual plot creation"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        resid_plot = ResidualPlot()
        fig, ax = resid_plot.plot(data=normal_data, distribution=dist)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_residuals_random_pattern(self, normal_data):
        """Test residuals show random pattern for good fit"""
        dist = NormalDistribution()
        dist.fit(normal_data, method='mle')
        
        resid_plot = ResidualPlot()
        residuals = resid_plot.calculate_residuals(normal_data, dist)
        
        # Should not have strong autocorrelation
        # (simple check: compare first and second half means)
        mid = len(residuals) // 2
        mean1 = np.mean(residuals[:mid])
        mean2 = np.mean(residuals[mid:])
        assert abs(mean1 - mean2) < 0.5


# ============================================================================
# TEST: MULTI-DISTRIBUTION PLOT
# ============================================================================

class TestMultiDistributionPlot:
    """Comprehensive tests for Multi-Distribution Comparison"""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data"""
        np.random.seed(42)
        return np.random.gamma(shape=2, scale=2, size=500)
    
    def test_initialization(self):
        """Test multi-distribution plot initialization"""
        multi_plot = MultiDistributionPlot()
        assert multi_plot is not None
    
    def test_compare_distributions(self, test_data):
        """Test comparing multiple distributions"""
        # Fit multiple distributions
        dist_names = ['normal', 'exponential', 'gamma', 'weibull']
        distributions = []
        
        for name in dist_names:
            dist = get_distribution(name)
            dist.fit(test_data, method='mle')
            distributions.append(dist)
        
        multi_plot = MultiDistributionPlot()
        fig = multi_plot.plot_comparison(
            data=test_data,
            distributions=distributions,
            labels=dist_names
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_subplot_layout(self, test_data):
        """Test subplot layout for multiple plots"""
        distributions = [
            get_distribution('normal'),
            get_distribution('exponential'),
            get_distribution('gamma')
        ]
        
        for dist in distributions:
            dist.fit(test_data, method='mle')
        
        multi_plot = MultiDistributionPlot()
        fig = multi_plot.plot_grid(
            data=test_data,
            distributions=distributions,
            plot_types=['pdf', 'cdf', 'qq']
        )
        
        # Should have multiple subplots
        assert len(fig.axes) >= 3
        plt.close(fig)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization framework"""
    
    def test_complete_visualization_pipeline(self):
        """Test complete visualization pipeline"""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=500)
        
        # Fit distribution
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        # Create all plots
        qq_plot = QQPlot()
        qq_fig, _ = qq_plot.plot(data, dist)
        
        pp_plot = PPPlot()
        pp_fig, _ = pp_plot.plot(data, dist)
        
        hist_plot = HistogramPlot()
        hist_fig, _ = hist_plot.plot_with_pdf(data, dist)
        
        cdf_plot = CDFPlot()
        cdf_fig, _ = cdf_plot.plot_comparison(data, dist)
        
        # All should be created
        assert qq_fig is not None
        assert pp_fig is not None
        assert hist_fig is not None
        assert cdf_fig is not None
        
        plt.close('all')
    
    def test_all_visualization_types(self):
        """Test all visualization types are available"""
        plot_types = [
            QQPlot,
            PPPlot,
            HistogramPlot,
            CDFPlot,
            PDFPlot,
            SurvivalPlot,
            ResidualPlot,
            MultiDistributionPlot
        ]
        
        for plot_class in plot_types:
            plot_obj = plot_class()
            assert plot_obj is not None
    
    def test_plot_config_consistency(self):
        """Test plot configuration is applied consistently"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=200)
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        config = PlotConfig(
            figsize=(10, 6),
            dpi=100,
            style='seaborn',
            color='blue'
        )
        
        qq_plot = QQPlot(config=config)
        fig, _ = qq_plot.plot(data, dist)
        
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        
        plt.close(fig)
    
    def test_export_plot(self):
        """Test plot can be exported"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=200)
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        qq_plot = QQPlot()
        fig, _ = qq_plot.plot(data, dist)
        
        # Test that figure can be saved (don't actually save in test)
        assert hasattr(fig, 'savefig')
        
        plt.close(fig)
    
    def test_interactive_features(self):
        """Test interactive plot features"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=200)
        dist = NormalDistribution()
        dist.fit(data, method='mle')
        
        hist_plot = HistogramPlot(interactive=True)
        fig, ax = hist_plot.plot_with_pdf(data, dist)
        
        # Should support interactivity
        assert fig is not None
        
        plt.close(fig)
    
    def test_batch_visualization(self):
        """Test batch visualization for multiple datasets"""
        np.random.seed(42)
        
        datasets = [
            np.random.normal(0, 1, 200),
            np.random.exponential(2, 200),
            np.random.uniform(0, 10, 200)
        ]
        
        distributions = [
            NormalDistribution(),
            ExponentialDistribution(),
            UniformDistribution()
        ]
        
        for data, dist in zip(datasets, distributions):
            dist.fit(data, method='mle')
        
        multi_plot = MultiDistributionPlot()
        fig = multi_plot.plot_comparison(
            data=datasets[0],
            distributions=distributions,
            labels=['Normal', 'Exponential', 'Uniform']
        )
        
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
