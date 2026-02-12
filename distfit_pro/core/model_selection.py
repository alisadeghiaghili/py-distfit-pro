"""Model Selection Criteria
========================

This module implements various model selection criteria:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- WAIC (Watanabe-Akaike Information Criterion)
- LOO-CV (Leave-One-Out Cross-Validation)

Each criterion explains why one model is better than another.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import stats
from ..locales import t


@dataclass
class ModelScore:
    """
    Model score with explanations
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
    Main class for model selection
    
    This class computes and compares different selection criteria.
    """
    
    @staticmethod
    def compute_aic(log_likelihood: float, n_params: int) -> float:
        """
        Akaike Information Criterion (AIC)
        
        Formula: AIC = 2k - 2ln(L)
        
        Explanation:
        -----------
        - k: number of model parameters
        - L: likelihood
        - Lower AIC is better
        - Penalty for complexity: 2k
        
        Usage:
        ------
        - Suitable for medium to large samples (n > 40)
        - Better for prediction
        - Prefers more complex models compared to BIC
        """
        return 2 * n_params - 2 * log_likelihood
    
    @staticmethod
    def compute_aic_c(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """
        Corrected AIC (AICc) for small samples
        
        Formula: AICc = AIC + [2k²+ 2k] / [n - k - 1]
        
        Explanation:
        -----------
        - Correction of AIC for small samples
        - Should be used when n/k < 40
        - Converges to AIC as n → ∞
        
        Usage:
        ------
        - Small samples (n < 40)
        - Prevents overfitting
        """
        aic = ModelSelection.compute_aic(log_likelihood, n_params)
        correction = (2 * n_params**2 + 2 * n_params) / (n_samples - n_params - 1)
        return aic + correction
    
    @staticmethod
    def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """
        Bayesian Information Criterion (BIC)
        
        Formula: BIC = k·ln(n) - 2ln(L)
        
        Explanation:
        -----------
        - Stronger penalty for complexity: k·ln(n)
        - Suitable for large samples
        - Lower BIC is better
        
        Usage:
        ------
        - When goal is identification of true model
        - Prefers simpler models more strongly
        - For large n, penalty is stronger than AIC
        
        Difference from AIC:
        --------------------
        - AIC: better for prediction
        - BIC: better for model selection (finding true model)
        """
        return n_params * np.log(n_samples) - 2 * log_likelihood
    
    @staticmethod
    def compute_likelihood(data: np.ndarray, distribution) -> float:
        """
        Compute log-likelihood
        
        Explanation:
        -----------
        - Probability of observing data under the model
        - Log is used for numerical stability
        """
        log_lik = np.sum(distribution.logpdf(data))
        return log_lik
    
    @staticmethod
    def compare_models(
        data: np.ndarray,
        fitted_distributions: List,
        criterion: str = 'aic'
    ) -> List[ModelScore]:
        """
        Compare multiple models using one criterion
        
        Parameters:
        -----------
        data : array-like
            Data
        fitted_distributions : list
            List of fitted distributions
        criterion : str
            'aic', 'aicc', 'bic', 'loo_cv'
        
        Returns:
        --------
        scores : list of ModelScore
            Sorted scores (best first)
        """
        n_samples = len(data)
        scores = []
        
        for dist in fitted_distributions:
            log_lik = ModelSelection.compute_likelihood(data, dist)
            n_params = len(dist.params)
            
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
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            scores.append(ModelScore(
                distribution_name=dist.info.name,
                criterion=criterion.upper(),
                score=score,
                n_params=n_params,
                sample_size=n_samples,
                explanation=expl
            ))
        
        # Sort (lowest score = best)
        scores.sort(key=lambda x: x.score)
        for rank, score_obj in enumerate(scores, 1):
            score_obj.rank = rank
        
        return scores
    
    @staticmethod
    def compute_loo_cv(data: np.ndarray, distribution) -> float:
        """
        Leave-One-Out Cross-Validation
        
        Explanation:
        -----------
        - For each data point:
          1. Fit model without that point
          2. Compute log-likelihood of that point
        - Sum of negative log-likelihoods = LOO score
        
        Advantages:
        -----------
        - Directly measures prediction quality
        - Sensitive to overfitting
        - No need for data splitting
        
        Disadvantages:
        --------------
        - Computationally expensive (n fits)
        - Slow for large n
        """
        n = len(data)
        loo_scores = []
        
        for i in range(n):
            # Remove one point
            train_data = np.delete(data, i)
            test_point = data[i:i+1]
            
            # Fit on rest
            dist_temp = distribution.__class__()
            try:
                dist_temp.fit(train_data, method='mle')
                # Compute log-likelihood of removed point
                log_lik = dist_temp.logpdf(test_point)[0]
                loo_scores.append(log_lik)
            except:
                # If fit fails, heavy penalty
                loo_scores.append(-1e6)
        
        # Negative sum of log-likelihoods
        return -np.sum(loo_scores)
    
    @staticmethod
    def _explain_aic(aic_value: float, n_params: int, n_samples: int) -> str:
        """Explain AIC"""
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
        """Explain AICc"""
        ratio = n_samples / n_params
        warning = t('model_sel_aicc_small_sample') if ratio < 40 else t('model_sel_aicc_large_sample')
        return f"""AICc = {aicc_value:.2f}

{t('model_sel_aicc_correction')}:
   • n/k = {ratio:.1f}
   • {warning}
"""
    
    @staticmethod
    def _explain_bic(bic_value: float, n_params: int, n_samples: int) -> str:
        """Explain BIC"""
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
        """Explain LOO-CV"""
        return f"""LOO-CV = {loo_value:.2f}

{t('model_sel_loo_cv_score')}:
   • {t('model_sel_loo_direct')}
   • {t('model_sel_loo_each_point')}
   • {t('model_sel_lower_better')}
"""


class DeltaComparison:
    """
    Compare models based on Δ (delta) criteria
    
    Δ_i = criterion_i - criterion_best
    
    Interpretation:
    ---------------
    - Δ < 2: Models are roughly equivalent
    - 2 < Δ < 7: Best model is noticeably better
    - Δ > 10: Best model is substantially better
    """
    
    @staticmethod
    def compute_deltas(scores: List[ModelScore]) -> List[Dict]:
        """
        Compute Δ for each model
        
        Returns:
        --------
        list of dict with explanations
        """
        best_score = scores[0].score  # Lowest
        deltas = []
        
        for score in scores:
            delta = score.score - best_score
            interpretation = DeltaComparison._interpret_delta(delta)
            
            deltas.append({
                'model': score.distribution_name,
                'score': score.score,
                'delta': delta,
                'interpretation': interpretation
            })
        
        return deltas
    
    @staticmethod
    def _interpret_delta(delta: float) -> str:
        """Interpret Δ value"""
        if delta < 2:
            return t('model_sel_delta_equivalent')
        elif delta < 7:
            return t('model_sel_delta_noticeably_worse')
        else:
            return t('model_sel_delta_substantially_worse')
    
    @staticmethod
    def print_comparison(scores: List[ModelScore]):
        """Print beautiful comparison"""
        deltas = DeltaComparison.compute_deltas(scores)
        
        print("\n" + "="*70)
        print(f"{t('model_sel_comparison_title')} {scores[0].criterion}")
        print("="*70)
        print(f"{'Rank':<6} {'Model':<15} {'Score':<12} {'Δ':<10} {t('model_sel_interpretation')}")
        print("-"*70)
        
        for i, (score, delta_info) in enumerate(zip(scores, deltas), 1):
            print(f"{i:<6} {score.distribution_name:<15} "
                  f"{score.score:<12.2f} {delta_info['delta']:<10.2f} "
                  f"{delta_info['interpretation']}")
        
        print("="*70)
        print(f"\n{t('model_sel_best_model')}: {scores[0].distribution_name}")
        print(f"\n{t('model_sel_explanation')}:")
        print(scores[0].explanation)
