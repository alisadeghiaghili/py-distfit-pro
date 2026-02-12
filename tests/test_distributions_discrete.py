"""
Comprehensive Tests for All Discrete Distributions (Part 5/10)
===============================================================

Tests for 5 discrete distributions:
1. Poisson Distribution (Event Counting)
2. Binomial Distribution (Success/Failure)
3. Negative Binomial Distribution (Over-Dispersion)
4. Geometric Distribution (Waiting Time)
5. Hypergeometric Distribution (Sampling)

Author: Ali Sadeghi Aghili
"""

import pytest
import numpy as np
from scipy import stats
from distfit_pro.core.distributions import (
    PoissonDistribution,
    BinomialDistribution,
    NegativeBinomialDistribution,
    GeometricDistribution,
    HypergeometricDistribution,
    get_distribution
)


# ============================================================================
# TEST: POISSON DISTRIBUTION
# ============================================================================

class TestPoissonDistribution:
    """Comprehensive tests for Poisson Distribution"""
    
    @pytest.fixture
    def poisson_data(self):
        """Generate Poisson distributed data"""
        np.random.seed(42)
        return np.random.poisson(lam=5, size=1000)
    
    def test_initialization(self):
        """Test Poisson distribution initialization"""
        dist = PoissonDistribution()
        assert dist.info.name == "poisson"
        assert dist._is_discrete
    
    def test_fit_mle(self, poisson_data):
        """Test MLE fitting"""
        dist = PoissonDistribution()
        dist.fit(poisson_data, method='mle')
        
        assert dist.fitted
        assert 'mu' in dist.params  # lambda (rate)
        assert 4.5 < dist.params['mu'] < 5.5
    
    def test_mle_equals_sample_mean(self, poisson_data):
        """Test MLE equals sample mean"""
        dist = PoissonDistribution()
        dist.fit(poisson_data, method='mle')
        
        # MLE for lambda is sample mean
        assert abs(dist.params['mu'] - np.mean(poisson_data)) < 0.01
    
    def test_mean_equals_variance(self, poisson_data):
        """Test mean equals variance property"""
        dist = PoissonDistribution()
        dist.fit(poisson_data, method='mle')
        
        mean = dist.mean()
        var = dist.var()
        
        # For Poisson: E[X] = Var[X] = lambda
        assert abs(mean - var) / mean < 0.15
    
    def test_non_negative_integers(self, poisson_data):
        """Test Poisson produces non-negative integers"""
        dist = PoissonDistribution()
        dist.fit(poisson_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
        assert np.all(samples == np.floor(samples))  # Integer values
    
    def test_pmf_sums_to_one(self, poisson_data):
        """Test PMF sums to approximately 1"""
        dist = PoissonDistribution()
        dist.fit(poisson_data, method='mle')
        
        # Sum PMF over reasonable range
        k_values = np.arange(0, 30)
        pmf_sum = np.sum(dist.pdf(k_values))  # pdf returns PMF for discrete
        
        assert 0.95 < pmf_sum < 1.05
    
    def test_rare_events(self):
        """Test Poisson for rare event modeling"""
        np.random.seed(42)
        # Simulate rare events (e.g., accidents per day)
        data = np.random.poisson(lam=0.5, size=365)
        
        dist = PoissonDistribution()
        dist.fit(data, method='mle')
        
        # Rate should be small
        assert 0.3 < dist.params['mu'] < 0.7
    
    def test_event_counting_application(self, poisson_data):
        """Test Poisson for event counting"""
        dist = PoissonDistribution()
        dist.fit(poisson_data, method='mle')
        
        # Probability of exactly k events
        k = 5
        prob_k = dist.pdf(np.array([k]))[0]
        assert 0 < prob_k < 1
        
        # Probability of at least k events
        prob_at_least_k = dist.sf(np.array([k-1]))[0]
        assert 0 < prob_at_least_k < 1
    
    def test_binomial_approximation(self):
        """Test Poisson approximates Binomial for large n, small p"""
        np.random.seed(42)
        n, p = 1000, 0.005
        # lambda = n*p = 5
        data = np.random.binomial(n, p, size=500)
        
        dist = PoissonDistribution()
        dist.fit(data, method='mle')
        
        # Should approximate lambda = 5
        assert 4 < dist.params['mu'] < 6


# ============================================================================
# TEST: BINOMIAL DISTRIBUTION
# ============================================================================

class TestBinomialDistribution:
    """Comprehensive tests for Binomial Distribution"""
    
    @pytest.fixture
    def binomial_data(self):
        """Generate Binomial distributed data"""
        np.random.seed(42)
        return np.random.binomial(n=10, p=0.6, size=1000)
    
    def test_initialization(self):
        """Test Binomial distribution initialization"""
        dist = BinomialDistribution()
        assert dist.info.name == "binomial"
        assert dist._is_discrete
    
    def test_fit_mle(self, binomial_data):
        """Test MLE fitting"""
        # Binomial needs n to be specified
        dist = BinomialDistribution()
        dist.fit(binomial_data, method='mle', n=10)
        
        assert dist.fitted
        assert 'n' in dist.params  # number of trials
        assert 'p' in dist.params  # success probability
        assert dist.params['n'] == 10
        assert 0.55 < dist.params['p'] < 0.65
    
    def test_mle_formula(self, binomial_data):
        """Test MLE formula: p = sample_mean / n"""
        dist = BinomialDistribution()
        n = 10
        dist.fit(binomial_data, method='mle', n=n)
        
        expected_p = np.mean(binomial_data) / n
        assert abs(dist.params['p'] - expected_p) < 0.01
    
    def test_bounded_support(self, binomial_data):
        """Test Binomial support is {0, 1, ..., n}"""
        dist = BinomialDistribution()
        dist.fit(binomial_data, method='mle', n=10)
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)
        assert np.all(samples == np.floor(samples))
    
    def test_mean_formula(self, binomial_data):
        """Test mean formula"""
        dist = BinomialDistribution()
        dist.fit(binomial_data, method='mle', n=10)
        
        mean = dist.mean()
        # E[X] = n * p
        expected_mean = dist.params['n'] * dist.params['p']
        
        assert abs(mean - expected_mean) < 0.01
    
    def test_variance_formula(self, binomial_data):
        """Test variance formula"""
        dist = BinomialDistribution()
        dist.fit(binomial_data, method='mle', n=10)
        
        var = dist.var()
        # Var[X] = n * p * (1 - p)
        n, p = dist.params['n'], dist.params['p']
        expected_var = n * p * (1 - p)
        
        assert abs(var - expected_var) < 0.01
    
    def test_symmetric_case(self):
        """Test symmetric Binomial (p=0.5)"""
        np.random.seed(42)
        data = np.random.binomial(n=20, p=0.5, size=1000)
        
        dist = BinomialDistribution()
        dist.fit(data, method='mle', n=20)
        
        # Should be symmetric
        assert 0.45 < dist.params['p'] < 0.55
        
        # Mean should be near n/2
        mean = dist.mean()
        assert 9 < mean < 11
    
    def test_skewness(self):
        """Test skewness formula"""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.3, size=1000)
        
        dist = BinomialDistribution()
        dist.fit(data, method='mle', n=10)
        
        skew = dist.skewness()
        # Skewness = (1 - 2p) / sqrt(np(1-p))
        n, p = dist.params['n'], dist.params['p']
        expected_skew = (1 - 2*p) / np.sqrt(n * p * (1-p))
        
        assert abs(skew - expected_skew) < 0.2
    
    def test_coin_flipping_application(self):
        """Test Binomial for coin flipping"""
        np.random.seed(42)
        # Flip fair coin 100 times
        data = np.random.binomial(n=100, p=0.5, size=500)
        
        dist = BinomialDistribution()
        dist.fit(data, method='mle', n=100)
        
        # Should recover p ≈ 0.5
        assert 0.45 < dist.params['p'] < 0.55


# ============================================================================
# TEST: NEGATIVE BINOMIAL DISTRIBUTION
# ============================================================================

class TestNegativeBinomialDistribution:
    """Comprehensive tests for Negative Binomial Distribution"""
    
    @pytest.fixture
    def negbinom_data(self):
        """Generate Negative Binomial distributed data"""
        np.random.seed(42)
        return np.random.negative_binomial(n=5, p=0.5, size=1000)
    
    def test_initialization(self):
        """Test Negative Binomial distribution initialization"""
        dist = NegativeBinomialDistribution()
        assert dist.info.name == "negative_binomial"
        assert dist._is_discrete
    
    def test_fit_mle(self, negbinom_data):
        """Test MLE fitting"""
        dist = NegativeBinomialDistribution()
        dist.fit(negbinom_data, method='mle')
        
        assert dist.fitted
        assert 'n' in dist.params  # r (number of successes)
        assert 'p' in dist.params  # success probability
        assert dist.params['n'] > 0
        assert 0 < dist.params['p'] < 1
    
    def test_overdispersion(self):
        """Test Negative Binomial handles over-dispersed count data"""
        np.random.seed(42)
        # Over-dispersed data (variance > mean)
        data = np.random.negative_binomial(n=2, p=0.3, size=1000)
        
        dist = NegativeBinomialDistribution()
        dist.fit(data, method='mle')
        
        mean = dist.mean()
        var = dist.var()
        
        # Variance should be greater than mean
        assert var > mean
    
    def test_mean_formula(self, negbinom_data):
        """Test mean formula"""
        dist = NegativeBinomialDistribution()
        dist.fit(negbinom_data, method='mle')
        
        mean = dist.mean()
        # E[X] = r * (1-p) / p
        r, p = dist.params['n'], dist.params['p']
        expected_mean = r * (1 - p) / p
        
        assert abs(mean - expected_mean) / mean < 0.2
    
    def test_variance_formula(self, negbinom_data):
        """Test variance formula"""
        dist = NegativeBinomialDistribution()
        dist.fit(negbinom_data, method='mle')
        
        var = dist.var()
        # Var[X] = r * (1-p) / p^2
        r, p = dist.params['n'], dist.params['p']
        expected_var = r * (1 - p) / (p ** 2)
        
        assert abs(var - expected_var) / var < 0.3
    
    def test_geometric_special_case(self):
        """Test that NegBinom(r=1) is Geometric"""
        np.random.seed(42)
        data = np.random.geometric(p=0.3, size=1000) - 1  # NegBinom counts failures
        
        dist = NegativeBinomialDistribution()
        dist.fit(data, method='mle')
        
        # r should be close to 1
        assert 0.5 < dist.params['n'] < 1.5
    
    def test_non_negative_integers(self, negbinom_data):
        """Test produces non-negative integers"""
        dist = NegativeBinomialDistribution()
        dist.fit(negbinom_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
        assert np.all(samples == np.floor(samples))
    
    def test_waiting_time_interpretation(self, negbinom_data):
        """Test waiting time interpretation"""
        dist = NegativeBinomialDistribution()
        dist.fit(negbinom_data, method='mle')
        
        # Number of failures before r successes
        assert dist.fitted


# ============================================================================
# TEST: GEOMETRIC DISTRIBUTION
# ============================================================================

class TestGeometricDistribution:
    """Comprehensive tests for Geometric Distribution"""
    
    @pytest.fixture
    def geometric_data(self):
        """Generate Geometric distributed data"""
        np.random.seed(42)
        return np.random.geometric(p=0.3, size=1000)
    
    def test_initialization(self):
        """Test Geometric distribution initialization"""
        dist = GeometricDistribution()
        assert dist.info.name == "geometric"
        assert dist._is_discrete
    
    def test_fit_mle(self, geometric_data):
        """Test MLE fitting"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        assert dist.fitted
        assert 'p' in dist.params  # success probability
        assert 0.25 < dist.params['p'] < 0.35
    
    def test_mle_formula(self, geometric_data):
        """Test MLE formula: p = 1 / sample_mean"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        expected_p = 1.0 / np.mean(geometric_data)
        assert abs(dist.params['p'] - expected_p) < 0.02
    
    def test_memoryless_property(self, geometric_data):
        """Test memoryless property"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        # P(X > s+t | X > s) = P(X > t)
        s, t = 3, 2
        prob1 = dist.sf(np.array([s + t]))[0] / dist.sf(np.array([s]))[0]
        prob2 = dist.sf(np.array([t]))[0]
        
        assert abs(prob1 - prob2) < 0.05
    
    def test_mean_formula(self, geometric_data):
        """Test mean formula"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        mean = dist.mean()
        # E[X] = 1/p
        expected_mean = 1.0 / dist.params['p']
        
        assert abs(mean - expected_mean) / mean < 0.01
    
    def test_variance_formula(self, geometric_data):
        """Test variance formula"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        var = dist.var()
        # Var[X] = (1-p) / p^2
        p = dist.params['p']
        expected_var = (1 - p) / (p ** 2)
        
        assert abs(var - expected_var) / var < 0.02
    
    def test_positive_integers(self, geometric_data):
        """Test Geometric produces positive integers"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 1)  # Support is {1, 2, 3, ...}
        assert np.all(samples == np.floor(samples))
    
    def test_waiting_time_application(self, geometric_data):
        """Test Geometric for waiting time"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        # Expected waiting time for first success
        expected_wait = dist.mean()
        assert expected_wait > 0
    
    def test_mode_equals_one(self, geometric_data):
        """Test mode is always 1"""
        dist = GeometricDistribution()
        dist.fit(geometric_data, method='mle')
        
        # PMF should be highest at k=1
        pmf_1 = dist.pdf(np.array([1]))[0]
        pmf_2 = dist.pdf(np.array([2]))[0]
        
        assert pmf_1 > pmf_2


# ============================================================================
# TEST: HYPERGEOMETRIC DISTRIBUTION
# ============================================================================

class TestHypergeometricDistribution:
    """Comprehensive tests for Hypergeometric Distribution"""
    
    @pytest.fixture
    def hypergeometric_data(self):
        """Generate Hypergeometric distributed data"""
        np.random.seed(42)
        return np.random.hypergeometric(ngood=50, nbad=50, nsample=10, size=1000)
    
    def test_initialization(self):
        """Test Hypergeometric distribution initialization"""
        dist = HypergeometricDistribution()
        assert dist.info.name == "hypergeometric"
        assert dist._is_discrete
    
    def test_fit_mle(self, hypergeometric_data):
        """Test MLE fitting"""
        # Hypergeometric needs N and n to be specified
        dist = HypergeometricDistribution()
        dist.fit(hypergeometric_data, method='mle', N=100, n=10)
        
        assert dist.fitted
        assert 'M' in dist.params  # number of success states in population
        assert 'n' in dist.params  # number of draws
        assert 'N' in dist.params  # population size
    
    def test_bounded_support(self, hypergeometric_data):
        """Test Hypergeometric bounded support"""
        dist = HypergeometricDistribution()
        dist.fit(hypergeometric_data, method='mle', N=100, n=10)
        
        samples = dist.rvs(size=100, random_state=42)
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)  # Can't draw more than n
        assert np.all(samples == np.floor(samples))
    
    def test_mean_formula(self, hypergeometric_data):
        """Test mean formula"""
        dist = HypergeometricDistribution()
        dist.fit(hypergeometric_data, method='mle', N=100, n=10)
        
        mean = dist.mean()
        # E[X] = n * M / N
        n = dist.params['n']
        M = dist.params['M']
        N = dist.params['N']
        expected_mean = n * M / N
        
        assert abs(mean - expected_mean) < 0.1
    
    def test_variance_formula(self, hypergeometric_data):
        """Test variance formula"""
        dist = HypergeometricDistribution()
        dist.fit(hypergeometric_data, method='mle', N=100, n=10)
        
        var = dist.var()
        # Var[X] = n * (M/N) * (1 - M/N) * (N-n)/(N-1)
        n = dist.params['n']
        M = dist.params['M']
        N = dist.params['N']
        p = M / N
        expected_var = n * p * (1 - p) * (N - n) / (N - 1)
        
        assert abs(var - expected_var) / var < 0.2
    
    def test_binomial_approximation_large_N(self):
        """Test Hypergeometric approximates Binomial for large N"""
        np.random.seed(42)
        # Large population, small sample
        N, M, n = 10000, 5000, 50
        data = np.random.hypergeometric(ngood=M, nbad=N-M, nsample=n, size=500)
        
        dist = HypergeometricDistribution()
        dist.fit(data, method='mle', N=N, n=n)
        
        # Should behave like Binomial(n, M/N)
        mean = dist.mean()
        expected_mean = n * (M / N)
        assert abs(mean - expected_mean) < 1
    
    def test_sampling_without_replacement(self):
        """Test Hypergeometric for sampling without replacement"""
        # Drawing cards without replacement
        dist = HypergeometricDistribution()
        # 52 cards, 13 hearts, draw 5
        dist.params = {'N': 52, 'M': 13, 'n': 5}
        dist.fitted = True
        
        # Probability of exactly 2 hearts
        prob_2_hearts = dist.pdf(np.array([2]))[0]
        assert 0 < prob_2_hearts < 1
    
    def test_quality_control_application(self, hypergeometric_data):
        """Test Hypergeometric for quality control"""
        # 100 items, 10 defective, sample 20
        dist = HypergeometricDistribution()
        dist.params = {'N': 100, 'M': 10, 'n': 20}
        dist.fitted = True
        
        # Expected defectives in sample
        expected_defectives = dist.mean()
        assert 1.5 < expected_defectives < 2.5  # Should be 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDiscreteDistributionsIntegration:
    """Integration tests for discrete distributions"""
    
    def test_all_distributions_fit(self):
        """Test all discrete distributions can be fitted"""
        np.random.seed(42)
        
        test_cases = [
            ('poisson', np.random.poisson(5, 500)),
            ('binomial', np.random.binomial(10, 0.6, 500)),
            ('negative_binomial', np.random.negative_binomial(5, 0.5, 500)),
            ('geometric', np.random.geometric(0.3, 500)),
            ('hypergeometric', np.random.hypergeometric(50, 50, 10, 500))
        ]
        
        for name, data in test_cases:
            dist = get_distribution(name)
            if name == 'binomial':
                dist.fit(data, method='mle', n=10)
            elif name == 'hypergeometric':
                dist.fit(data, method='mle', N=100, n=10)
            else:
                dist.fit(data, method='mle')
            assert dist.fitted, f"{name} failed to fit"
    
    def test_all_produce_integers(self):
        """Test all discrete distributions produce integers"""
        np.random.seed(42)
        
        distributions = [
            (PoissonDistribution(), np.random.poisson(5, 300)),
            (GeometricDistribution(), np.random.geometric(0.3, 300)),
        ]
        
        for dist, data in distributions:
            dist.fit(data, method='mle')
            samples = dist.rvs(size=50, random_state=42)
            assert np.all(samples == np.floor(samples))
    
    def test_pmf_properties(self):
        """Test PMF properties for discrete distributions"""
        np.random.seed(42)
        data = np.random.poisson(5, 500)
        
        dist = PoissonDistribution()
        dist.fit(data, method='mle')
        
        # PMF should be non-negative
        k_values = np.arange(0, 20)
        pmf_values = dist.pdf(k_values)
        assert np.all(pmf_values >= 0)
        
        # PMF should sum to approximately 1
        pmf_sum = np.sum(pmf_values)
        assert 0.9 < pmf_sum < 1.05
    
    def test_cdf_monotonic(self):
        """Test CDF is monotonically increasing"""
        np.random.seed(42)
        data = np.random.binomial(10, 0.6, 500)
        
        dist = BinomialDistribution()
        dist.fit(data, method='mle', n=10)
        
        k_values = np.arange(0, 11)
        cdf_values = dist.cdf(k_values)
        
        # Check monotonicity
        assert np.all(np.diff(cdf_values) >= 0)
    
    def test_summary_output(self):
        """Test summary output for discrete distributions"""
        np.random.seed(42)
        
        distributions = [
            (PoissonDistribution(), np.random.poisson(5, 300)),
            (GeometricDistribution(), np.random.geometric(0.3, 300)),
        ]
        
        for dist, data in distributions:
            dist.fit(data, method='mle')
            summary = dist.summary()
            explain = dist.explain()
            
            assert len(summary) > 100
            assert len(explain) > 50
    
    def test_all_25_distributions_complete(self):
        """Verify all 25 distributions (20 continuous + 5 discrete) are tested"""
        all_distributions = [
            # Continuous (20)
            'normal', 'lognormal', 'weibull', 'gamma', 'exponential',
            'beta', 'uniform', 'triangular', 'logistic', 'gumbel',
            'frechet', 'pareto', 'cauchy', 'studentt', 'chisquared',
            'f', 'rayleigh', 'laplace', 'invgamma', 'loglogistic',
            # Discrete (5)
            'poisson', 'binomial', 'negative_binomial', 'geometric', 'hypergeometric'
        ]
        
        assert len(all_distributions) == 25
        
        # All should be accessible
        for name in all_distributions:
            dist = get_distribution(name)
            assert dist is not None
    
    def test_discrete_vs_continuous_flag(self):
        """Test _is_discrete flag is set correctly"""
        continuous = get_distribution('normal')
        discrete = get_distribution('poisson')
        
        assert not continuous._is_discrete
        assert discrete._is_discrete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
