"""Creating Custom Distributions

Learn how to:
- Create user-defined distributions
- Implement MLE and MoM fitting
- Use custom distributions with distfit-pro
- Build domain-specific models

Perfect for: Advanced users, custom models
Time: ~20 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from distfit_pro.core.base import ContinuousDistribution, DistributionInfo

print("="*70)
print("🔧 CREATING CUSTOM DISTRIBUTIONS")
print("="*70)

# ============================================================================
# Example 1: Shifted Exponential Distribution
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: Shifted Exponential Distribution")
print("="*70)

print("\n💡 Use Case:")
print("  Model time-to-failure with guaranteed minimum lifetime.")
print("  E.g., warranty period of 100 hours, then exponential failure.")

class ShiftedExponential(ContinuousDistribution):
    """Exponential distribution with location shift.
    
    PDF: f(x) = (1/scale) * exp(-(x-shift)/scale) for x >= shift
    
    Parameters:
    - shift: Minimum value (location parameter)
    - scale: Rate parameter (1/lambda)
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.expon
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name='shifted_exponential',
            scipy_name='expon',
            display_name='Shifted Exponential',
            description='Exponential with guaranteed minimum value',
            parameters=['shift', 'scale'],
            support='[shift, inf)',
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        """Maximum Likelihood Estimation."""
        # MLE: shift = min(data), scale = mean(data) - min(data)
        shift_est = data.min()
        scale_est = data.mean() - shift_est
        
        self._params = {
            'shift': shift_est,
            'scale': scale_est
        }
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """Method of Moments."""
        # For simplicity, use same as MLE
        self._fit_mle(data, **kwargs)
    
    def _get_scipy_params(self):
        """Convert to scipy format."""
        return {
            'loc': self._params['shift'],
            'scale': self._params['scale']
        }

print("\n🛠️  Created custom distribution: ShiftedExponential")

# Generate test data
np.random.seed(100)
true_shift = 100
true_scale = 50
data = np.random.exponential(scale=true_scale, size=300) + true_shift

print(f"\n📊 Generated test data:")
print(f"  True shift: {true_shift}")
print(f"  True scale: {true_scale}")
print(f"  Sample size: {len(data)}")

# Fit custom distribution
dist = ShiftedExponential()
dist.fit(data)

print(f"\n🎯 Estimated parameters:")
for name, value in dist.params.items():
    true_val = true_shift if name == 'shift' else true_scale
    error_pct = abs(value - true_val) / true_val * 100
    print(f"  {name:<10} = {value:>10.4f}  (true: {true_val}, error: {error_pct:.2f}%)")

print(f"\n📊 Distribution statistics:")
print(f"  Mean: {dist.mean():.4f}  (true: {true_shift + true_scale:.4f})")
print(f"  Std: {dist.std():.4f}  (true: {true_scale:.4f})")

# ============================================================================
# Example 2: Truncated Normal Distribution
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Truncated Normal Distribution")
print("="*70)

print("\n💡 Use Case:")
print("  Model normally distributed data with physical bounds.")
print("  E.g., test scores between 0 and 100, heights between 140-210 cm.")

class TruncatedNormal(ContinuousDistribution):
    """Normal distribution truncated to [a, b].
    
    Parameters:
    - loc: Mean of underlying normal
    - scale: Std of underlying normal
    - a: Lower bound
    - b: Upper bound
    """
    
    def __init__(self, a=0, b=100):
        super().__init__()
        self.a = a
        self.b = b
        self._scipy_dist = None  # Will be created after fitting
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name='truncated_normal',
            scipy_name='truncnorm',
            display_name='Truncated Normal',
            description=f'Normal distribution truncated to [{self.a}, {self.b}]',
            parameters=['loc', 'scale'],
            support=f'[{self.a}, {self.b}]',
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        """MLE for truncated normal (simplified)."""
        # Use sample mean/std as starting point
        loc_est = data.mean()
        scale_est = data.std()
        
        self._params = {
            'loc': loc_est,
            'scale': scale_est
        }
        
        # Create scipy truncated normal
        a_std = (self.a - loc_est) / scale_est
        b_std = (self.b - loc_est) / scale_est
        self._scipy_dist = stats.truncnorm(a_std, b_std, loc=loc_est, scale=scale_est)
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """Method of moments."""
        self._fit_mle(data, **kwargs)
    
    def _get_scipy_params(self):
        """Scipy params already set in _scipy_dist."""
        return {}
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Override PDF to use pre-configured scipy dist."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Override CDF."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.cdf(x)
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Override PPF."""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.ppf(q)

print("\n🛠️  Created custom distribution: TruncatedNormal")

# Generate test data (exam scores)
np.random.seed(200)
scores = np.random.normal(loc=70, scale=12, size=500)
scores = np.clip(scores, 0, 100)  # Truncate to [0, 100]

print(f"\n📊 Exam scores data:")
print(f"  Sample size: {len(scores)}")
print(f"  Range: [{scores.min():.2f}, {scores.max():.2f}]")
print(f"  Mean: {scores.mean():.2f}")
print(f"  Std: {scores.std():.2f}")

# Fit truncated normal
dist_trunc = TruncatedNormal(a=0, b=100)
dist_trunc.fit(scores)

print(f"\n🎯 Estimated parameters:")
for name, value in dist_trunc.params.items():
    print(f"  {name:<10} = {value:>10.4f}")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Shifted Exponential
ax1 = axes[0]
ax1.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', 
         edgecolor='black', label='Data')
x1 = np.linspace(data.min(), data.max(), 300)
ax1.plot(x1, dist.pdf(x1), 'r-', linewidth=2, label='Fitted Distribution')
ax1.axvline(dist.params['shift'], color='green', linestyle='--', 
            linewidth=2, label=f'Estimated shift={dist.params["shift"]:.2f}')
ax1.set_xlabel('Value', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Shifted Exponential', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Truncated Normal
ax2 = axes[1]
ax2.hist(scores, bins=30, density=True, alpha=0.6, color='lightcoral', 
         edgecolor='black', label='Data')
x2 = np.linspace(0, 100, 300)
ax2.plot(x2, dist_trunc.pdf(x2), 'b-', linewidth=2, label='Fitted Distribution')
ax2.axvline(0, color='red', linestyle=':', linewidth=1.5, label='Bounds')
ax2.axvline(100, color='red', linestyle=':', linewidth=1.5)
ax2.set_xlabel('Score', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Truncated Normal (Exam Scores)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('custom_distributions.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: custom_distributions.png")

# ============================================================================
# Using Custom Distributions
# ============================================================================
print("\n" + "="*70)
print("USING CUSTOM DISTRIBUTIONS")
print("="*70)

print("\n📚 Example: Predictions with Shifted Exponential")

# Reliability analysis
time_points = [150, 200, 250]
print("\nSurvival probabilities (P(X > t)):")
for t in time_points:
    surv_prob = dist.sf(t)
    print(f"  P(lifetime > {t} hours) = {surv_prob:.4f} ({surv_prob*100:.2f}%)")

# Warranty period
warranty_pct = 0.9
warranty_time = dist.ppf(warranty_pct)
print(f"\n90% of units will last at least {warranty_time:.2f} hours")

print("\n📚 Example: Predictions with Truncated Normal")

# Grade boundaries
grade_boundaries = [0.5, 0.7, 0.85, 0.95]
grade_names = ['Pass', 'Good', 'Excellent', 'Outstanding']

print("\nGrade boundaries (percentiles):")
for pct, grade in zip(grade_boundaries, grade_names):
    score = dist_trunc.ppf(pct)
    print(f"  {grade:<15}: {score:>6.2f}  (top {(1-pct)*100:.0f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎓 ADVANCED TECHNIQUES LEARNED")
print("="*70)

print("\nKey Skills:")
print("  1. ✓ Inherit from ContinuousDistribution base class")
print("  2. ✓ Implement info property with DistributionInfo")
print("  3. ✓ Implement _fit_mle() and _fit_mom() methods")
print("  4. ✓ Convert parameters to scipy format with _get_scipy_params()")
print("  5. ✓ Override pdf/cdf/ppf for complex distributions")

print("\nCustom Distribution Use Cases:")
print("  → Domain-specific models (finance, engineering, biology)")
print("  → Distributions not in scipy/distfit-pro")
print("  → Modified standard distributions (truncated, shifted, mixture)")
print("  → Proprietary models for competitive advantage")

print("\n💡 Pro Tips:")
print("  - Start with scipy distribution if available")
print("  - Validate implementation with known datasets")
print("  - Add proper docstrings and type hints")
print("  - Consider numerical stability for optimization")
print("="*70)
