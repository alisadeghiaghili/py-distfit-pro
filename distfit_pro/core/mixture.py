"""
Mixture Models
=============

Fit mixture distributions using Expectation-Maximization (EM) algorithm.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class MixtureComponent:
    """
    Single component of a mixture model
    
    Attributes:
    -----------
    distribution : BaseDistribution
        Fitted distribution
    weight : float
        Mixture weight (proportion)
    """
    distribution: any  # BaseDistribution
    weight: float


class MixtureModel:
    """
    Mixture of distributions using EM algorithm
    
    Example:
    --------
    >>> from distfit_pro.core.distributions import get_distribution
    >>> from distfit_pro.core.mixture import MixtureModel
    >>> 
    >>> # Bimodal data
    >>> data1 = np.random.normal(0, 1, 500)
    >>> data2 = np.random.normal(5, 1, 500)
    >>> data = np.concatenate([data1, data2])
    >>> 
    >>> # Fit mixture of 2 normals
    >>> mixture = MixtureModel(n_components=2, distribution_name='normal')
    >>> mixture.fit(data, max_iter=100)
    >>> print(mixture.summary())
    """
    
    def __init__(self,
                 n_components: int,
                 distribution_name: str,
                 random_state: Optional[int] = None):
        """
        Initialize mixture model
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        distribution_name : str
            Name of distribution (e.g., 'normal', 'gamma')
        random_state : int, optional
            Random seed
        """
        from distfit_pro.core.distributions import get_distribution
        
        self.n_components = n_components
        self.distribution_name = distribution_name
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Components
        self.components: List[MixtureComponent] = []
        self.fitted = False
    
    def fit(self,
            data: np.ndarray,
            max_iter: int = 100,
            tol: float = 1e-4) -> 'MixtureModel':
        """
        Fit mixture model using EM algorithm
        
        Parameters:
        -----------
        data : array-like
            Observed data
        max_iter : int
            Maximum EM iterations
        tol : float
            Convergence tolerance (change in log-likelihood)
            
        Returns:
        --------
        self : MixtureModel
            Fitted model
        """
        from distfit_pro.core.distributions import get_distribution
        
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Initialize components randomly
        weights = np.ones(self.n_components) / self.n_components
        components = []
        
        for k in range(self.n_components):
            # Random subset for initialization
            subset_size = max(int(n / self.n_components), 50)
            subset_idx = np.random.choice(n, size=min(subset_size, n), replace=False)
            subset_data = data[subset_idx]
            
            dist = get_distribution(self.distribution_name)
            try:
                dist.fit(subset_data, method='mle')
            except:
                # Fallback: use perturbed overall fit
                dist.fit(data, method='mle')
                for param_name in dist.params:
                    dist.params[param_name] *= (1 + 0.1 * np.random.randn())
            
            components.append(dist)
        
        # EM iterations
        log_likelihood_old = -np.inf
        
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            responsibilities = np.zeros((n, self.n_components))
            
            for k in range(self.n_components):
                responsibilities[:, k] = weights[k] * components[k].pdf(data)
            
            # Normalize
            responsibilities_sum = np.sum(responsibilities, axis=1, keepdims=True)
            responsibilities = responsibilities / (responsibilities_sum + 1e-10)
            
            # M-step: update parameters
            for k in range(self.n_components):
                # Update weight
                weights[k] = np.mean(responsibilities[:, k])
                
                # Update distribution parameters (weighted MLE)
                r_k = responsibilities[:, k]
                r_k = r_k / np.sum(r_k)  # Normalize
                
                # Refit with weights
                try:
                    from distfit_pro.core.weighted import WeightedDistributionFitter
                    fitter = WeightedDistributionFitter()
                    params = fitter.fit(data, r_k, components[k], method='mle')
                    components[k].params = params
                except:
                    # Fallback: refit without weights
                    sample_idx = np.random.choice(n, size=n, replace=True, p=r_k)
                    sample_data = data[sample_idx]
                    components[k].fit(sample_data, method='mle')
            
            # Check convergence
            log_likelihood = self._compute_log_likelihood(data, components, weights)
            
            if np.abs(log_likelihood - log_likelihood_old) < tol:
                break
            
            log_likelihood_old = log_likelihood
        
        # Store results
        self.components = [
            MixtureComponent(distribution=comp, weight=w)
            for comp, w in zip(components, weights)
        ]
        self.fitted = True
        
        return self
    
    def _compute_log_likelihood(self,
                               data: np.ndarray,
                               components: List,
                               weights: np.ndarray) -> float:
        """
        Compute log-likelihood of mixture
        """
        n = len(data)
        likelihood = np.zeros(n)
        
        for k, (comp, w) in enumerate(zip(components, weights)):
            likelihood += w * comp.pdf(data)
        
        return np.sum(np.log(likelihood + 1e-10))
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Mixture PDF
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        result = np.zeros_like(x, dtype=float)
        for comp in self.components:
            result += comp.weight * comp.distribution.pdf(x)
        return result
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Mixture CDF
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        result = np.zeros_like(x, dtype=float)
        for comp in self.components:
            result += comp.weight * comp.distribution.cdf(x)
        return result
    
    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples from mixture
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Sample component assignments
        weights = [comp.weight for comp in self.components]
        component_assignments = np.random.choice(
            self.n_components, size=size, p=weights
        )
        
        # Sample from each component
        samples = np.zeros(size)
        for k in range(self.n_components):
            mask = component_assignments == k
            n_k = np.sum(mask)
            if n_k > 0:
                samples[mask] = self.components[k].distribution.rvs(size=n_k)
        
        return samples
    
    def summary(self) -> str:
        """
        Summary of mixture model
        """
        if not self.fitted:
            return "Mixture model not fitted"
        
        output = []
        output.append("\n" + "="*70)
        output.append(f"MIXTURE MODEL ({self.n_components} components)")
        output.append("="*70)
        
        for i, comp in enumerate(self.components, 1):
            output.append(f"\nComponent {i}: {comp.distribution.info.display_name}")
            output.append(f"  Weight: {comp.weight:.4f}")
            output.append(f"  Parameters:")
            for param_name, param_value in comp.distribution.params.items():
                output.append(f"    {param_name}: {param_value:.6f}")
        
        output.append("\n" + "="*70)
        return "\n".join(output)
