"""Model Selection Criteria
========================

This module implements various model selection criteria:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)  
- AICc (Corrected AIC for small samples)
- LOO-CV (Leave-One-Out Cross-Validation)

Each criterion explains why one model is better than another.

Examples
--------
Basic model comparison:

>>> from distfit_pro import get_distribution
>>> from distfit_pro.core.model_selection import ModelSelection
>>> import numpy as np
>>> 
>>> # Generate data
>>> data = np.random.lognormal(0, 1, 500)
>>> 
>>> # Fit candidate models
>>> candidates = ['normal', 'lognormal', 'gamma', 'weibull']
>>> fitted_models = []
>>> for name in candidates:
...     dist = get_distribution(name)
...     try:
...         dist.fit(data)
...         fitted_models.append(dist)
...     except:
...         print(f"Failed to fit {name}")
>>> 
>>> # Compare with AIC
>>> aic_scores = ModelSelection.compare_models(data, fitted_models, criterion='aic')
>>> for score in aic_scores:
...     print(f"{score.rank}. {score.distribution_name}: {score.score:.2f}")
1. lognormal: 1234.56
2. gamma: 1245.78
3. weibull: 1256.89
4. normal: 1289.12
>>> 
>>> # Print detailed comparison
>>> from distfit_pro.core.model_selection import DeltaComparison
>>> DeltaComparison.print_comparison(aic_scores)

Cross-validation example:

>>> # LOO-CV (more accurate but slower)
>>> loo_scores = ModelSelection.compare_models(data, fitted_models, criterion='loo_cv')
>>> best_model = loo_scores[0]
>>> print(f"Best model by LOO-CV: {best_model.distribution_name}")
Best model by LOO-CV: lognormal

Small sample correction:

>>> # Small dataset - use AICc
>>> small_data = np.random.exponential(5, 30)
>>> dist = get_distribution('exponential')
>>> dist.fit(small_data)
>>> 
>>> n = len(small_data)
>>> k = len(dist.params)
>>> print(f"n/k = {n/k:.1f}")  # Should use AICc if < 40
n/k = 30.0
>>> 
>>> log_lik = dist.log_likelihood(small_data)
>>> aicc = ModelSelection.compute_aic_c(log_lik, k, n)
>>> print(f"AICc = {aicc:.2f}")
AICc = 123.45

Multiple criteria comparison:

>>> # Compare using multiple criteria
>>> criteria_results = {}
>>> for criterion in ['aic', 'bic', 'loo_cv']:
...     scores = ModelSelection.compare_models(data, fitted_models, criterion=criterion)
...     criteria_results[criterion] = scores[0].distribution_name
>>> 
>>> print("Best model by criterion:")
>>> for crit, best in criteria_results.items():
...     print(f"  {crit.upper()}: {best}")
Best model by criterion:
  AIC: lognormal
  BIC: lognormal
  LOO_CV: lognormal
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats
import warnings

try:
    from ..locales import t
except ImportError:
    # Fallback if locales not available
    def t(key):
        return key


@dataclass
class ModelScore:
    """
    Model score with explanations
    
    Attributes
    ----------
    distribution_name : str
        Name of the distribution
    criterion : str
        Criterion used (AIC, BIC, etc)
    score : float
        Criterion value (lower is better)
    n_params : int
        Number of parameters
    sample_size : int
        Size of dataset
    explanation : str
        Human-readable explanation
    rank : int, optional
        Rank among compared models (1 = best)
    
    Examples
    --------
    >>> score = ModelScore(
    ...     distribution_name='normal',
    ...     criterion='AIC',
    ...     score=1234.56,
    ...     n_params=2,
    ...     sample_size=1000,
    ...     explanation='Lower is better',
    ...     rank=1
    ... )
    >>> print(score)
    normal: AIC=1234.56 (rank 1)
    """
    distribution_name: str
    criterion: str
    score: float
    n_params: int
    sample_size: int
    explanation: str
    rank: Optional[int] = None
    
    def __repr__(self) -> str:
        return f"{self.distribution_name}: {self.criterion}={self.score:.2f} (rank {self.rank})"


class ModelSelection:
    """
    Model selection and comparison using information criteria
    
    This class provides methods to:
    - Compute AIC, BIC, AICc, LOO-CV
    - Compare multiple distributions
    - Explain differences between models
    - Handle failed fits gracefully
    
    All criteria follow "lower is better" convention.
    
    Methods
    -------
    compute_aic(log_likelihood, n_params)
        Calculate Akaike Information Criterion
    compute_aic_c(log_likelihood, n_params, n_samples)
        Calculate corrected AIC for small samples
    compute_bic(log_likelihood, n_params, n_samples)
        Calculate Bayesian Information Criterion
    compute_loo_cv(data, distribution)
        Calculate Leave-One-Out Cross-Validation score
    compare_models(data, fitted_distributions, criterion)
        Compare multiple models using specified criterion
    
    Examples
    --------
    Basic usage:
    
    >>> from distfit_pro import get_distribution
    >>> import numpy as np
    >>> 
    >>> # Generate test data
    >>> np.random.seed(42)
    >>> data = np.random.gamma(2, 2, 1000)
    >>> 
    >>> # Fit candidate distributions
    >>> models = {}
    >>> for name in ['gamma', 'weibull', 'lognormal', 'exponential']:
    ...     dist = get_distribution(name)
    ...     try:
    ...         dist.fit(data)
    ...         models[name] = dist
    ...     except Exception as e:
    ...         print(f"Failed to fit {name}: {e}")
    >>> 
    >>> # Compare with AIC
    >>> scores = ModelSelection.compare_models(
    ...     data, 
    ...     list(models.values()),
    ...     criterion='aic'
    ... )
    >>> 
    >>> # Print results
    >>> print("Model Comparison (AIC):")
    >>> for score in scores:
    ...     delta = score.score - scores[0].score
    ...     print(f"{score.rank}. {score.distribution_name}: "
    ...           f"{score.score:.2f} (Δ={delta:.2f})")
    Model Comparison (AIC):
    1. gamma: 4567.12 (Δ=0.00)
    2. weibull: 4589.34 (Δ=22.22)
    3. lognormal: 4601.56 (Δ=34.44)
    4. exponential: 4678.90 (Δ=111.78)
    
    Small sample correction:
    
    >>> # Small dataset (n=25)
    >>> small_data = np.random.exponential(2, 25)
    >>> dist = get_distribution('exponential')
    >>> dist.fit(small_data)
    >>> 
    >>> # Check if AICc is needed
    >>> n, k = len(small_data), len(dist.params)
    >>> if n / k < 40:
    ...     print(f"Use AICc (n/k = {n/k:.1f} < 40)")
    ...     log_lik = dist.log_likelihood()
    ...     score = ModelSelection.compute_aic_c(log_lik, k, n)
    ... else:
    ...     print("Use AIC")
    ...     score = dist.aic()
    Use AICc (n/k = 25.0 < 40)
    >>> print(f"AICc = {score:.2f}")
    AICc = 78.45
    
    Cross-validation:
    
    >>> # LOO-CV for robust comparison (slower)
    >>> loo_scores = ModelSelection.compare_models(
    ...     data[:100],  # Subset for speed
    ...     [models['gamma'], models['weibull']],
    ...     criterion='loo_cv'
    ... )
    >>> best = loo_scores[0]
    >>> print(f"Best by LOO-CV: {best.distribution_name}")
    Best by LOO-CV: gamma
    
    Notes
    -----
    **When to use each criterion:**
    
    - **AIC**: General purpose, prediction-focused, n > 40
    - **AICc**: Small samples (n/k < 40), prevents overfitting
    - **BIC**: Finding "true" model, penalizes complexity more
    - **LOO-CV**: Most robust, computationally expensive
    
    **Interpretation:**
    
    - Δ < 2: Models essentially equivalent
    - 2 < Δ < 7: Some evidence for best model
    - Δ > 10: Strong evidence for best model
    
    **Common pitfalls:**
    
    - Don't compare AIC across different datasets
    - Don't use BIC for very small samples (n < 10)
    - LOO-CV may fail if some folds don't converge
    - Always check if fits succeeded before comparing
    """
    
    @staticmethod
    def compute_aic(log_likelihood: float, n_params: int) -> float:
        """
        Akaike Information Criterion (AIC)
        
        Formula: AIC = 2k - 2ln(L)
        
        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of fitted model
        n_params : int
            Number of parameters in model
        
        Returns
        -------
        aic : float
            AIC value (lower is better)
        
        Notes
        -----
        **Interpretation:**
        - k: number of model parameters
        - L: likelihood
        - Lower AIC is better
        - Penalty for complexity: 2k
        
        **When to use:**
        - Suitable for medium to large samples (n > 40)
        - Better for prediction
        - Prefers more complex models compared to BIC
        
        **Difference from BIC:**
        - AIC: asymptotically efficient (best for prediction)
        - BIC: consistent (finds true model as n → ∞)
        
        Examples
        --------
        >>> dist = get_distribution('normal')
        >>> dist.fit(data)
        >>> log_lik = dist.log_likelihood()
        >>> k = len(dist.params)
        >>> aic = ModelSelection.compute_aic(log_lik, k)
        >>> print(f"AIC = {aic:.2f}")
        AIC = 1234.56
        
        See Also
        --------
        compute_aic_c : Corrected AIC for small samples
        compute_bic : Bayesian Information Criterion
        """
        if not np.isfinite(log_likelihood):
            warnings.warn(
                "Log-likelihood is not finite. Check if model fitting succeeded.",
                RuntimeWarning
            )
            return np.inf
        
        if n_params < 1:
            raise ValueError(f"n_params must be >= 1, got {n_params}")
        
        return 2 * n_params - 2 * log_likelihood
    
    @staticmethod
    def compute_aic_c(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """
        Corrected AIC (AICc) for small samples
        
        Formula: AICc = AIC + [2k² + 2k] / [n - k - 1]
        
        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of fitted model
        n_params : int
            Number of parameters
        n_samples : int
            Sample size
        
        Returns
        -------
        aicc : float
            Corrected AIC (lower is better)
        
        Raises
        ------
        ValueError
            If n_samples <= n_params + 1 (correction undefined)
        
        Notes
        -----
        **When to use:**
        - Small samples where n/k < 40
        - Prevents overfitting better than AIC
        - Converges to AIC as n → ∞
        
        **Rule of thumb:**
        - n/k < 40: Use AICc
        - n/k >= 40: AIC and AICc are essentially equivalent
        
        Examples
        --------
        >>> # Small sample
        >>> small_data = np.random.normal(0, 1, 30)
        >>> dist = get_distribution('normal')
        >>> dist.fit(small_data)
        >>> 
        >>> n, k = len(small_data), len(dist.params)
        >>> log_lik = dist.log_likelihood()
        >>> 
        >>> # Check ratio
        >>> print(f"n/k = {n/k:.1f}")
        n/k = 15.0
        >>> 
        >>> # Use AICc for small sample
        >>> aicc = ModelSelection.compute_aic_c(log_lik, k, n)
        >>> aic = ModelSelection.compute_aic(log_lik, k)
        >>> print(f"AIC  = {aic:.2f}")
        >>> print(f"AICc = {aicc:.2f} (penalty = {aicc-aic:.2f})")
        AIC  = 78.45
        AICc = 79.67 (penalty = 1.22)
        
        See Also
        --------
        compute_aic : Standard AIC
        """
        if n_samples <= n_params + 1:
            raise ValueError(
                f"Sample size ({n_samples}) must be > n_params + 1 ({n_params + 1}) "
                "for AICc correction. Use more data or fewer parameters."
            )
        
        aic = ModelSelection.compute_aic(log_likelihood, n_params)
        
        # Correction term
        correction = (2 * n_params**2 + 2 * n_params) / (n_samples - n_params - 1)
        
        return aic + correction
    
    @staticmethod
    def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """
        Bayesian Information Criterion (BIC)
        
        Formula: BIC = k·ln(n) - 2ln(L)
        
        Parameters
        ----------
        log_likelihood : float
            Log-likelihood of fitted model
        n_params : int
            Number of parameters
        n_samples : int
            Sample size
        
        Returns
        -------
        bic : float
            BIC value (lower is better)
        
        Notes
        -----
        **Interpretation:**
        - Stronger penalty for complexity: k·ln(n) vs 2k for AIC
        - Suitable for large samples
        - Lower BIC is better
        
        **When to use:**
        - Goal is identification of "true" model
        - Prefer simpler models
        - Large sample sizes (n > 100)
        
        **BIC vs AIC:**
        - AIC: better for prediction
        - BIC: better for model selection (finding true model)
        - For large n, BIC penalty >> AIC penalty
        - At n=8, penalties are equal (ln(8) ≈ 2)
        - At n=100, BIC penalty is 2.3x stronger
        
        Examples
        --------
        >>> # Compare AIC vs BIC penalty
        >>> dist = get_distribution('gamma')
        >>> dist.fit(data)
        >>> 
        >>> n, k = len(data), len(dist.params)
        >>> log_lik = dist.log_likelihood()
        >>> 
        >>> aic = ModelSelection.compute_aic(log_lik, k)
        >>> bic = ModelSelection.compute_bic(log_lik, k, n)
        >>> 
        >>> aic_penalty = 2 * k
        >>> bic_penalty = k * np.log(n)
        >>> ratio = bic_penalty / aic_penalty
        >>> 
        >>> print(f"AIC penalty: {aic_penalty}")
        >>> print(f"BIC penalty: {bic_penalty:.2f}")
        >>> print(f"BIC/AIC ratio: {ratio:.2f}x")
        AIC penalty: 4
        BIC penalty: 27.63
        BIC/AIC ratio: 6.91x
        
        See Also
        --------
        compute_aic : Akaike Information Criterion
        """
        if not np.isfinite(log_likelihood):
            warnings.warn(
                "Log-likelihood is not finite. Check if model fitting succeeded.",
                RuntimeWarning
            )
            return np.inf
        
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        
        if n_params < 1:
            raise ValueError(f"n_params must be >= 1, got {n_params}")
        
        return n_params * np.log(n_samples) - 2 * log_likelihood
    
    @staticmethod
    def compute_likelihood(data: np.ndarray, distribution) -> float:
        """
        Compute log-likelihood of data under fitted distribution
        
        Parameters
        ----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
        
        Returns
        -------
        log_likelihood : float
            Sum of log(pdf(x)) over all data points
        
        Notes
        -----
        Log is used for numerical stability. The actual likelihood
        can be extremely small (e.g., 10^-500), causing underflow.
        
        Examples
        --------
        >>> dist = get_distribution('normal')
        >>> dist.fit(data)
        >>> log_lik = ModelSelection.compute_likelihood(data, dist)
        >>> print(f"Log-likelihood: {log_lik:.2f}")
        Log-likelihood: -1234.56
        """
        try:
            log_lik = np.sum(distribution.logpdf(data))
            if not np.isfinite(log_lik):
                warnings.warn(
                    f"Non-finite log-likelihood for {distribution.info.name}. "
                    "Model may be inappropriate for this data.",
                    RuntimeWarning
                )
            return log_lik
        except Exception as e:
            warnings.warn(
                f"Failed to compute likelihood for {distribution.info.name}: {e}",
                RuntimeWarning
            )
            return -np.inf
    
    @staticmethod
    def compare_models(
        data: np.ndarray,
        fitted_distributions: List,
        criterion: str = 'aic'
    ) -> List[ModelScore]:
        """
        Compare multiple fitted distributions using specified criterion
        
        Parameters
        ----------
        data : array-like
            Observed data (same data used to fit all models)
        fitted_distributions : list of BaseDistribution
            List of fitted distribution objects
        criterion : {'aic', 'aicc', 'bic', 'loo_cv'}, default='aic'
            Selection criterion to use
        
        Returns
        -------
        scores : list of ModelScore
            Sorted list (best model first)
        
        Raises
        ------
        ValueError
            If criterion is unknown or distributions list is empty
        RuntimeWarning
            If some models fail to compute scores
        
        Examples
        --------
        >>> # Fit multiple models
        >>> from distfit_pro import get_distribution
        >>> 
        >>> data = np.random.lognormal(0, 1, 500)
        >>> candidates = ['normal', 'lognormal', 'gamma', 'weibull']
        >>> 
        >>> fitted = []
        >>> for name in candidates:
        ...     try:
        ...         dist = get_distribution(name)
        ...         dist.fit(data)
        ...         fitted.append(dist)
        ...     except Exception as e:
        ...         print(f"Failed to fit {name}: {e}")
        >>> 
        >>> # Compare with AIC
        >>> scores = ModelSelection.compare_models(data, fitted, 'aic')
        >>> 
        >>> # Print ranking
        >>> for score in scores:
        ...     print(f"{score.rank}. {score.distribution_name}: "
        ...           f"{score.score:.2f}")
        1. lognormal: 1234.56
        2. gamma: 1245.78
        3. weibull: 1267.89
        4. normal: 1298.12
        
        >>> # Compare with multiple criteria
        >>> for crit in ['aic', 'bic']:
        ...     scores = ModelSelection.compare_models(data, fitted, crit)
        ...     print(f"Best by {crit.upper()}: {scores[0].distribution_name}")
        Best by AIC: lognormal
        Best by BIC: lognormal
        
        Notes
        -----
        - All distributions must be fitted on the SAME data
        - Failed score computations result in np.inf (ranked last)
        - LOO-CV is much slower than AIC/BIC
        
        See Also
        --------
        DeltaComparison.print_comparison : Pretty-print results
        """
        if not fitted_distributions:
            raise ValueError("fitted_distributions cannot be empty")
        
        valid_criteria = ['aic', 'aicc', 'bic', 'loo_cv']
        if criterion.lower() not in valid_criteria:
            raise ValueError(
                f"Unknown criterion: {criterion}. "
                f"Must be one of {valid_criteria}"
            )
        
        data = np.asarray(data)
        n_samples = len(data)
        scores = []
        failed_models = []
        
        for dist in fitted_distributions:
            try:
                log_lik = ModelSelection.compute_likelihood(data, dist)
                n_params = len(dist.params)
                
                # Compute score based on criterion
                if criterion == 'aic':
                    score = ModelSelection.compute_aic(log_lik, n_params)
                    expl = ModelSelection._explain_aic(score, n_params, n_samples)
                elif criterion == 'aicc':
                    score = ModelSelection.compute_aic_c(log_lik, n_params, n_samples)
                    expl = ModelSelection._explain_aicc(score, n_params, n_samples)
                elif criterion == 'bic':
                    score = ModelSelection.compute_bic(log_lik, n_params, n_samples)
                    expl = ModelSelection._explain_bic(score, n_params, n_samples)
                elif criterion == 'loo_cv':
                    score = ModelSelection.compute_loo_cv(data, dist)
                    expl = ModelSelection._explain_loo(score)
                
                scores.append(ModelScore(
                    distribution_name=dist.info.name,
                    criterion=criterion.upper(),
                    score=score,
                    n_params=n_params,
                    sample_size=n_samples,
                    explanation=expl
                ))
                
            except Exception as e:
                # Handle failed score computation
                warnings.warn(
                    f"Failed to compute {criterion} for {dist.info.name}: {e}. "
                    "Assigning worst score (inf).",
                    RuntimeWarning
                )
                failed_models.append(dist.info.name)
                
                # Add with infinite score (ranked last)
                scores.append(ModelScore(
                    distribution_name=dist.info.name,
                    criterion=criterion.upper(),
                    score=np.inf,
                    n_params=len(dist.params) if hasattr(dist, 'params') else 0,
                    sample_size=n_samples,
                    explanation="Score computation failed"
                ))
        
        # Sort by score (lowest = best)
        scores.sort(key=lambda x: x.score)
        
        # Assign ranks
        for rank, score_obj in enumerate(scores, 1):
            score_obj.rank = rank
        
        # Warn about failed models
        if failed_models:
            warnings.warn(
                f"{len(failed_models)} model(s) failed: {', '.join(failed_models)}. "
                "Check model fit and data compatibility.",
                RuntimeWarning
            )
        
        return scores
    
    @staticmethod
    def compute_loo_cv(data: np.ndarray, distribution) -> float:
        """
        Leave-One-Out Cross-Validation score
        
        Parameters
        ----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Distribution to evaluate (will be refit n times)
        
        Returns
        -------
        loo_score : float
            Negative sum of log-likelihoods (lower is better)
        
        Notes
        -----
        **How it works:**
        1. For each data point i:
           - Fit model on data without point i
           - Compute log P(x_i | model)
        2. Return: -Σ log P(x_i | model_{-i})
        
        **Advantages:**
        - Directly measures prediction quality
        - No assumptions about model complexity
        - Robust to overfitting
        
        **Disadvantages:**
        - Computationally expensive (n model fits)
        - Can fail if some folds don't converge
        - Slow for large n
        
        **When to use:**
        - Small to medium samples (n < 1000)
        - When computational cost is acceptable
        - When you want most accurate comparison
        
        Examples
        --------
        >>> # Compare LOO-CV vs AIC
        >>> dist = get_distribution('normal')
        >>> dist.fit(data)
        >>> 
        >>> # LOO-CV (slow but accurate)
        >>> loo_score = ModelSelection.compute_loo_cv(data, dist)
        >>> print(f"LOO-CV: {loo_score:.2f}")
        LOO-CV: 1234.56
        >>> 
        >>> # AIC (fast approximation)
        >>> aic_score = dist.aic(data)
        >>> print(f"AIC: {aic_score:.2f}")
        AIC: 1235.12
        >>> 
        >>> # They should be similar
        >>> print(f"Difference: {abs(loo_score - aic_score):.2f}")
        Difference: 0.56
        
        Warnings
        --------
        This method can be very slow for large datasets.
        Consider using AIC/BIC for n > 1000.
        """
        data = np.asarray(data)
        n = len(data)
        loo_scores = []
        failed_folds = 0
        
        for i in range(n):
            # Leave one out
            train_data = np.delete(data, i)
            test_point = data[i:i+1]
            
            # Create new distribution instance
            dist_class = distribution.__class__
            dist_temp = dist_class()
            
            try:
                # Fit on n-1 points
                dist_temp.fit(train_data, method='mle', verbose=False)
                
                # Evaluate on left-out point
                log_lik = dist_temp.logpdf(test_point)[0]
                
                if not np.isfinite(log_lik):
                    # Numerical issue
                    log_lik = -1e10
                    failed_folds += 1
                
                loo_scores.append(log_lik)
                
            except Exception as e:
                # Fit failed for this fold
                warnings.warn(
                    f"LOO-CV fold {i+1}/{n} failed: {e}. Using penalty.",
                    RuntimeWarning
                )
                loo_scores.append(-1e10)  # Heavy penalty
                failed_folds += 1
        
        # Warn if many folds failed
        if failed_folds > n * 0.1:  # More than 10%
            warnings.warn(
                f"{failed_folds}/{n} LOO-CV folds failed. "
                "Results may be unreliable. Consider using AIC/BIC instead.",
                RuntimeWarning
            )
        
        # Return negative sum (so lower = better)
        return -np.sum(loo_scores)
    
    @staticmethod
    def _explain_aic(aic_value: float, n_params: int, n_samples: int) -> str:
        """Generate AIC explanation"""
        return f"""AIC = {aic_value:.2f}

{t('model_sel_aic_components')}:
   • {t('model_sel_complexity_penalty')}: 2×{n_params} = {2*n_params}
   • {t('model_sel_goodness_of_fit')}: -2×log(likelihood)
   
{t('model_sel_interpretation')}:
   • {t('model_sel_lower_better')}
   • {t('model_sel_aic_balance')}
   • {t('model_sel_aic_prediction')}
"""
    
    @staticmethod
    def _explain_aicc(aicc_value: float, n_params: int, n_samples: int) -> str:
        """Generate AICc explanation"""
        ratio = n_samples / n_params
        warning = t('model_sel_aicc_small_sample') if ratio < 40 else t('model_sel_aicc_large_sample')
        return f"""AICc = {aicc_value:.2f}

{t('model_sel_aicc_correction')}:
   • n/k = {ratio:.1f}
   • {warning}
"""
    
    @staticmethod
    def _explain_bic(bic_value: float, n_params: int, n_samples: int) -> str:
        """Generate BIC explanation"""
        penalty_ratio = np.log(n_samples) / 2
        return f"""BIC = {bic_value:.2f}

{t('model_sel_bic_strong_penalty')}:
   • {t('model_sel_penalty')}: {n_params}×ln({n_samples}) = {n_params * np.log(n_samples):.1f}
   • {t('model_sel_bic_aic_ratio')}: {penalty_ratio:.2f}×
   
{t('model_sel_interpretation')}:
   • {t('model_sel_bic_simpler')}
   • {t('model_sel_bic_true_model')}
   • {t('model_sel_bic_increasing_n')}
"""
    
    @staticmethod
    def _explain_loo(loo_value: float) -> str:
        """Generate LOO-CV explanation"""
        return f"""LOO-CV = {loo_value:.2f}

{t('model_sel_loo_cv_score')}:
   • {t('model_sel_loo_direct')}
   • {t('model_sel_loo_each_point')}
   • {t('model_sel_lower_better')}
"""


class DeltaComparison:
    """
    Compare models using Δ (delta) values
    
    Δ_i = criterion_i - criterion_best
    
    Interpretation (Burnham & Anderson, 2002):
    - Δ < 2: Substantial support (models essentially equivalent)
    - 2 ≤ Δ < 7: Considerably less support  
    - Δ ≥ 10: Essentially no support
    
    Examples
    --------
    >>> # After comparing models
    >>> scores = ModelSelection.compare_models(data, fitted_models, 'aic')
    >>> 
    >>> # Compute and display deltas
    >>> DeltaComparison.print_comparison(scores)
    ======================================================================
    Model Comparison: AIC
    ======================================================================
    Rank   Model           Score        Δ          Interpretation
    ----------------------------------------------------------------------
    1      lognormal       1234.56      0.00       Substantial support
    2      gamma           1238.12      3.56       Considerably less support
    3      weibull         1247.89      13.33      Essentially no support
    4      normal          1289.45      54.89      Essentially no support
    ======================================================================
    
    >>> # Programmatic access
    >>> deltas = DeltaComparison.compute_deltas(scores)
    >>> for d in deltas:
    ...     if d['delta'] < 2:
    ...         print(f"{d['model']} is competitive")
    lognormal is competitive
    """
    
    @staticmethod
    def compute_deltas(scores: List[ModelScore]) -> List[Dict]:
        """
        Compute Δ values for each model
        
        Parameters
        ----------
        scores : list of ModelScore
            Sorted model scores (best first)
        
        Returns
        -------
        deltas : list of dict
            Each dict contains:
            - 'model': distribution name
            - 'score': criterion value
            - 'delta': difference from best
            - 'interpretation': text explanation
        
        Examples
        --------
        >>> scores = ModelSelection.compare_models(data, models, 'aic')
        >>> deltas = DeltaComparison.compute_deltas(scores)
        >>> 
        >>> # Find competitive models (Δ < 2)
        >>> competitive = [d for d in deltas if d['delta'] < 2]
        >>> print(f"Found {len(competitive)} competitive models")
        Found 2 competitive models
        """
        if not scores:
            return []
        
        best_score = scores[0].score
        deltas = []
        
        for score in scores:
            delta = score.score - best_score
            interpretation = DeltaComparison._interpret_delta(delta)
            
            deltas.append({
                'model': score.distribution_name,
                'score': score.score,
                'delta': delta,
                'interpretation': interpretation,
                'rank': score.rank
            })
        
        return deltas
    
    @staticmethod
    def _interpret_delta(delta: float) -> str:
        """
        Interpret Δ value (Burnham & Anderson, 2002)
        
        Parameters
        ----------
        delta : float
            Difference from best model
        
        Returns
        -------
        interpretation : str
            Human-readable interpretation
        """
        if delta < 2:
            return t('model_sel_delta_equivalent')
        elif delta < 7:
            return t('model_sel_delta_noticeably_worse')
        else:
            return t('model_sel_delta_substantially_worse')
    
    @staticmethod
    def print_comparison(scores: List[ModelScore]):
        """
        Print beautiful comparison table
        
        Parameters
        ----------
        scores : list of ModelScore
            Sorted model scores from compare_models()
        
        Examples
        --------
        >>> scores = ModelSelection.compare_models(data, models, 'aic')
        >>> DeltaComparison.print_comparison(scores)
        """
        if not scores:
            print("No scores to display")
            return
        
        deltas = DeltaComparison.compute_deltas(scores)
        
        print("\n" + "="*70)
        print(f"{t('model_sel_comparison_title')} {scores[0].criterion}")
        print("="*70)
        print(f"{'Rank':<6} {'Model':<15} {'Score':<12} {'Δ':<10} {t('model_sel_interpretation')}")
        print("-"*70)
        
        for score, delta_info in zip(scores, deltas):
            # Skip models with infinite scores
            if not np.isfinite(score.score):
                print(f"{score.rank:<6} {score.distribution_name:<15} "
                      f"{'FAILED':<12} {'-':<10} {'Fit failed'}")
            else:
                print(f"{score.rank:<6} {score.distribution_name:<15} "
                      f"{score.score:<12.2f} {delta_info['delta']:<10.2f} "
                      f"{delta_info['interpretation']}")
        
        print("="*70)
        
        # Show best model
        if np.isfinite(scores[0].score):
            print(f"\n{t('model_sel_best_model')}: {scores[0].distribution_name}")
            print(f"\n{t('model_sel_explanation')}:")
            print(scores[0].explanation)
        else:
            print("\n⚠️  All models failed to compute scores.")
            print("Check data compatibility and model fitting.")
