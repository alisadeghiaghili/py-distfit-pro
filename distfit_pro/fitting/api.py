"""
High-Level API Functions
========================

Convenience functions for one-line workflows.
"""

from typing import List, Optional, Tuple, Dict, Union
import numpy as np
from .fitter import DistributionFitter, FitResults
from ..core.base import BaseDistribution
from ..core.distributions import get_distribution
from ..core.model_selection import ModelSelection


# Alias for compatibility
FitResult = FitResults


def fit(
    data: Union[np.ndarray, list],
    distribution: str = 'normal',
    method: str = 'mle',
    plot: bool = False,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    **kwargs
) -> FitResults:
    """
    Quick one-line distribution fitting.
    
    Parameters
    ----------
    data : array-like
        Observed data
    distribution : str, default='normal'
        Distribution name
    method : str, default='mle'
        Fitting method: 'mle' or 'mom'
    plot : bool, default=False
        Show diagnostic plots
    bootstrap : bool, default=False
        Calculate bootstrap confidence intervals
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    **kwargs
        Additional arguments
        
    Returns
    -------
    result : FitResults
        Fitting results
        
    Examples
    --------
    >>> import numpy as np
    >>> from distfit_pro import fit
    >>> 
    >>> data = np.random.normal(10, 2, 500)
    >>> result = fit(data, 'normal', plot=True)
    >>> print(result.summary())
    """
    fitter = DistributionFitter(data)
    result = fitter.fit(
        distributions=[distribution],
        method=method,
        verbose=False
    )
    
    if plot:
        result.plot(kind='comparison', show=True)
    
    if bootstrap:
        # TODO: Add bootstrap CI calculation
        pass
    
    return result


def find_best_distribution(
    data: Union[np.ndarray, list],
    candidates: Optional[List[str]] = None,
    criterion: str = 'aic',
    method: str = 'mle',
    verbose: bool = True
) -> Tuple[str, FitResults]:
    """
    Find best-fitting distribution from candidates.
    
    Parameters
    ----------
    data : array-like
        Observed data
    candidates : list of str, optional
        Candidate distributions. If None, uses suggested ones.
    criterion : str, default='aic'
        Selection criterion: 'aic', 'bic', 'loo_cv'
    method : str, default='mle'
        Fitting method
    verbose : bool, default=True
        Show progress
        
    Returns
    -------
    best_name : str
        Name of best distribution
    best_result : FitResults
        Full fitting results
        
    Examples
    --------
    >>> from distfit_pro import find_best_distribution
    >>> 
    >>> data = np.random.gamma(2, 2, 500)
    >>> best_name, result = find_best_distribution(
    ...     data,
    ...     candidates=['normal', 'gamma', 'weibull']
    ... )
    >>> print(f"Best: {best_name}")
    """
    fitter = DistributionFitter(data)
    result = fitter.fit(
        distributions=candidates,
        method=method,
        criterion=criterion,
        verbose=verbose
    )
    
    best_name = result.best_model.info.name
    return best_name, result


class ComparisonResult:
    """
    Results from comparing multiple distributions.
    """
    
    def __init__(self, data, model_scores, fitted_models):
        self.data = data
        self.model_scores = model_scores
        self.fitted_models = fitted_models
        self.rankings = [
            {
                'distribution': score.distribution_name,
                'score': score.score,
                'criterion': score.criterion
            }
            for score in model_scores
        ]
    
    def get_best(self) -> Dict:
        """Get best distribution"""
        return self.rankings[0]
    
    def get_by_rank(self, rank: int) -> Dict:
        """Get distribution by rank (1-indexed)"""
        if rank < 1 or rank > len(self.rankings):
            raise ValueError(f"Rank must be between 1 and {len(self.rankings)}")
        return self.rankings[rank - 1]
    
    def to_table(self) -> str:
        """Generate comparison table"""
        lines = []
        lines.append(f"{'Rank':<6} {'Distribution':<15} {'Score':<12} {'Delta':<10}")
        lines.append("-" * 50)
        
        best_score = self.rankings[0]['score']
        for i, rank in enumerate(self.rankings, 1):
            delta = rank['score'] - best_score
            lines.append(
                f"{i:<6} {rank['distribution']:<15} {rank['score']:<12.2f} {delta:<10.2f}"
            )
        
        return "\n".join(lines)
    
    def plot(self, metric: str = 'aic', **kwargs):
        """Plot comparison"""
        import matplotlib.pyplot as plt
        
        names = [r['distribution'] for r in self.rankings]
        scores = [r['score'] for r in self.rankings]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, scores)
        ax.set_xlabel(metric.upper())
        ax.set_title('Distribution Comparison')
        ax.invert_yaxis()
        
        return fig


def compare_distributions(
    data: Union[np.ndarray, list],
    distributions: List[str],
    metrics: List[str] = ['aic'],
    method: str = 'mle',
    verbose: bool = True
) -> ComparisonResult:
    """
    Compare multiple distributions.
    
    Parameters
    ----------
    data : array-like
        Observed data
    distributions : list of str
        Distributions to compare
    metrics : list of str, default=['aic']
        Comparison metrics: 'aic', 'bic', 'ks'
    method : str, default='mle'
        Fitting method
    verbose : bool, default=True
        Show progress
        
    Returns
    -------
    comparison : ComparisonResult
        Comparison results with rankings
        
    Examples
    --------
    >>> from distfit_pro import compare_distributions
    >>> 
    >>> data = np.random.normal(0, 1, 500)
    >>> comparison = compare_distributions(
    ...     data,
    ...     distributions=['normal', 'laplace', 'logistic']
    ... )
    >>> print(comparison.to_table())
    """
    fitter = DistributionFitter(data)
    result = fitter.fit(
        distributions=distributions,
        method=method,
        criterion=metrics[0],  # Use first metric for ranking
        verbose=verbose
    )
    
    return ComparisonResult(
        data=fitter.data,
        model_scores=result.model_scores,
        fitted_models=result.fitted_models
    )


__all__ = [
    'fit',
    'find_best_distribution',
    'compare_distributions',
    'FitResult',
    'FitResults',
    'ComparisonResult',
]
