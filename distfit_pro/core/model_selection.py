"""Model Selection Criteria
========================

این ماژول معیارهای مختلف انتخاب مدل را پیاده‌سازی می‌کند:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- WAIC (Watanabe-Akaike Information Criterion)
- LOO-CV (Leave-One-Out Cross-Validation)

هر معیار توضیح می‌دهد که چرا یک مدل را بهتر می‌داند.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import stats


@dataclass
class ModelScore:
    """
    امتیاز یک مدل با توضیحات
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
    کلاس اصلی برای انتخاب مدل
    
    این کلاس معیارهای مختلف را محاسبه و مقایسه می‌کند.
    """
    
    @staticmethod
    def compute_aic(log_likelihood: float, n_params: int) -> float:
        """
        Akaike Information Criterion (AIC)
        
        فرمول: AIC = 2k - 2ln(L)
        
        توضیح:
        -------
        - k: تعداد پارامترهای مدل
        - L: likelihood
        - مدل با AIC کمتر بهتر است
        - جریمه برای پیچیدگی: 2k
        
        کاربرد:
        --------
        - مناسب برای نمونه‌های متوسط تا بزرگ (n > 40)
        - برای prediction بهتر است
        - نسبت به BIC، مدل‌های پیچیده‌تر را ترجیح می‌دهد
        """
        return 2 * n_params - 2 * log_likelihood
    
    @staticmethod
    def compute_aic_c(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """
        Corrected AIC (AICc) for small samples
        
        فرمول: AICc = AIC + [2k²+ 2k] / [n - k - 1]
        
        توضیح:
        -------
        - اصلاح AIC برای نمونه‌های کوچک
        - وقتی n/k < 40، استفاده شود
        - برای n → ∞ به AIC میل می‌کند
        
        کاربرد:
        --------
        - نمونه‌های کوچک (n < 40)
        - جلوگیری از overfitting
        """
        aic = ModelSelection.compute_aic(log_likelihood, n_params)
        correction = (2 * n_params**2 + 2 * n_params) / (n_samples - n_params - 1)
        return aic + correction
    
    @staticmethod
    def compute_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
        """
        Bayesian Information Criterion (BIC)
        
        فرمول: BIC = k·ln(n) - 2ln(L)
        
        توضیح:
        -------
        - جریمه قوی‌تر برای پیچیدگی: k·ln(n)
        - مناسب برای نمونه‌های بزرگ
        - مدل با BIC کمتر بهتر است
        
        کاربرد:
        --------
        - وقتی هدف identification مدل واقعی است
        - مدل‌های ساده‌تر را بیشتر ترجیح می‌دهد
        - برای n بزرگ، جریمه شدیدتر از AIC
        
        تفاوت با AIC:
        --------------
        - AIC: بهتر برای prediction
        - BIC: بهتر برای selection (انتخاب مدل درست)
        """
        return n_params * np.log(n_samples) - 2 * log_likelihood
    
    @staticmethod
    def compute_likelihood(data: np.ndarray, distribution) -> float:
        """
        محاسبه log-likelihood
        
        توضیح:
        -------
        - احتمال دیدن داده تحت مدل
        - log استفاده می‌شود برای stability عددی
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
        مقایسه چند مدل با یک معیار
        
        Parameters:
        -----------
        data : array-like
            داده
        fitted_distributions : list
            لیست توزیع‌های فیت‌شده
        criterion : str
            'aic', 'aicc', 'bic', 'loo_cv'
        
        Returns:
        --------
        scores : list of ModelScore
            امتیازات مرتب شده (بهترین اول)
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
        
        # مرتب‌سازی (کمترین امتیاز = بهترین)
        scores.sort(key=lambda x: x.score)
        for rank, score_obj in enumerate(scores, 1):
            score_obj.rank = rank
        
        return scores
    
    @staticmethod
    def compute_loo_cv(data: np.ndarray, distribution) -> float:
        """
        Leave-One-Out Cross-Validation
        
        توضیح:
        -------
        - برای هر نقطه داده:
          1. مدل را بدون آن نقطه فیت کن
          2. log-likelihood آن نقطه را محاسبه کن
        - مجموع log-likelihoods منفی = LOO score
        
        مزایا:
        -------
        - مستقیم کیفیت prediction را می‌سنجد
        - به overfitting حساس است
        - نیاز به تقسیم‌بندی ندارد
        
        معایب:
        -------
        - محاسباتی گران (n بار فیت)
        - برای n بزرگ کند است
        """
        n = len(data)
        loo_scores = []
        
        for i in range(n):
            # حذف یک نقطه
            train_data = np.delete(data, i)
            test_point = data[i:i+1]
            
            # فیت روی بقیه
            dist_temp = distribution.__class__()
            try:
                dist_temp.fit(train_data, method='mle')
                # محاسبه log-likelihood نقطه‌ی حذف‌شده
                log_lik = dist_temp.logpdf(test_point)[0]
                loo_scores.append(log_lik)
            except:
                # اگر فیت ناموفق بود، جریمه سنگین
                loo_scores.append(-1e6)
        
        # منفی مجموع log-likelihoods
        return -np.sum(loo_scores)
    
    @staticmethod
    def _explain_aic(aic_value: float, n_params: int, n_samples: int) -> str:
        """توضیح AIC"""
        return f"""AIC = {aic_value:.2f}

💡 این عدد از دو بخش تشکیل شده:
   • جریمه پیچیدگی: 2×{n_params} = {2*n_params}
   • Goodness of fit: -2×log(likelihood)
   
📊 تفسیر:
   • عدد کوچک‌تر = مدل بهتر
   • تعادل بین fit خوب و سادگی
   • مناسب برای prediction
"""
    
    @staticmethod
    def _explain_aicc(aicc_value: float, n_params: int, n_samples: int) -> str:
        """توضیح AICc"""
        ratio = n_samples / n_params
        return f"""AICc = {aicc_value:.2f}

💡 اصلاح AIC برای نمونه کوچک:
   • n/k = {ratio:.1f}
   • {"⚠️ نمونه کوچک - AICc را استفاده کن" if ratio < 40 else "✅ نمونه بزرگ - AIC کافی است"}
"""
    
    @staticmethod
    def _explain_bic(bic_value: float, n_params: int, n_samples: int) -> str:
        """توضیح BIC"""
        penalty_ratio = np.log(n_samples) / 2
        return f"""BIC = {bic_value:.2f}

💡 این عدد جریمه قوی‌تری برای پیچیدگی دارد:
   • جریمه: {n_params}×ln({n_samples}) = {n_params * np.log(n_samples):.1f}
   • نسبت جریمه BIC/AIC: {penalty_ratio:.2f}×
   
📊 تفسیر:
   • مدل‌های ساده‌تر را بیشتر ترجیح می‌دهد
   • برای یافتن "مدل واقعی" مناسب است
   • با افزایش n، جریمه شدیدتر می‌شود
"""
    
    @staticmethod
    def _explain_loo(loo_value: float) -> str:
        """توضیح LOO-CV"""
        return f"""LOO-CV = {loo_value:.2f}

💡 امتیاز cross-validation:
   • مستقیماً توان prediction را می‌سنجد
   • هر نقطه یک‌بار test می‌شود
   • عدد کوچک‌تر = prediction بهتر
"""


class DeltaComparison:
    """
    مقایسه مدل‌ها بر اساس Δ (delta) criteria
    
    Δ_i = criterion_i - criterion_best
    
    تفسیر:
    -------
    - Δ < 2: مدل‌ها تقریباً یکسان‌اند
    - 2 < Δ < 7: مدل بهترین قابل‌توجه بهتر است
    - Δ > 10: مدل بهترین به‌مراتب بهتر است
    """
    
    @staticmethod
    def compute_deltas(scores: List[ModelScore]) -> List[Dict]:
        """
        محاسبه Δ برای هر مدل
        
        Returns:
        --------
        list of dict با توضیحات
        """
        best_score = scores[0].score  # کمترین
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
        """تفسیر مقدار Δ"""
        if delta < 2:
            return "✅ تقریباً یکسان با بهترین مدل - هر دو قابل استفاده"
        elif delta < 7:
            return "⚠️ قابل‌توجه ضعیف‌تر - اگر دلیل خاصی نباشد، بهترین را بگیر"
        else:
            return "❌ به‌مراتب ضعیف‌تر - استفاده نشود"
    
    @staticmethod
    def print_comparison(scores: List[ModelScore]):
        """چاپ مقایسه زیبا"""
        deltas = DeltaComparison.compute_deltas(scores)
        
        print("\n" + "="*70)
        print("📊 مقایسه مدل‌ها بر اساس", scores[0].criterion)
        print("="*70)
        print(f"{'Rank':<6} {'Model':<15} {'Score':<12} {'Δ':<10} {'تفسیر'}")
        print("-"*70)
        
        for i, (score, delta_info) in enumerate(zip(scores, deltas), 1):
            print(f"{i:<6} {score.distribution_name:<15} "
                  f"{score.score:<12.2f} {delta_info['delta']:<10.2f} "
                  f"{delta_info['interpretation']}")
        
        print("="*70)
        print(f"\n🏆 مدل برتر: {scores[0].distribution_name}")
        print(f"\n💡 توضیح:")
        print(scores[0].explanation)
