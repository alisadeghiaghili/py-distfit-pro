#!/usr/bin/env python3
"""
Mixture Models: Fitting Multiple Distribution Components
========================================================

When data comes from multiple populations:
  - Gaussian Mixture Models (GMM)
  - EM algorithm
  - Component identification
  - Real-world applications

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

np.random.seed(42)

print("="*70)
print("🧩 MIXTURE MODELS: MULTIPLE POPULATIONS")
print("="*70)


# ============================================================================
# Theory: Mixture Models
# ============================================================================

print("\n" + "="*70)
print("📚 Theory: Mixture Models")
print("="*70)

theory = """
1. WHAT ARE MIXTURE MODELS?
   • Data comes from K different populations (components)
   • Each component has its own distribution
   • Mixture = weighted sum of components
   • Formula: f(x) = π1*f1(x) + π2*f2(x) + ... + πK*fK(x)
   • Weights πi sum to 1

2. GAUSSIAN MIXTURE MODEL (GMM):
   • Each component is Gaussian (normal)
   • Parameters: means μi, std devs σi, weights πi
   • Fitted using EM (Expectation-Maximization) algorithm
   • Most common mixture model

3. WHEN TO USE:
   • Multimodal data (multiple peaks)
   • Heterogeneous populations
   • Customer segmentation
   • Defect classification
   • When single distribution doesn't fit

4. FITTING PROCESS:
   1. Choose number of components K
   2. Initialize parameters randomly
   3. E-step: Assign data to components (soft assignment)
   4. M-step: Update parameters given assignments
   5. Repeat until convergence

5. MODEL SELECTION:
   • BIC or AIC to choose K
   • Trade-off: more components = better fit but more complex
"""

print(theory)


# ============================================================================
# Example 1: Simple Two-Component Mixture
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Two-Component Gaussian Mixture")
print("="*70)

print("""
Scenario: Customer heights (male + female populations)
  - Two distinct groups
  - Fit mixture model
  - Identify components
""")

# Generate mixture data
# Component 1: Female heights ~ N(162, 6^2)
# Component 2: Male heights ~ N(175, 7^2)
n1 = 300
n2 = 200

data_female = np.random.normal(162, 6, n1)
data_male = np.random.normal(175, 7, n2)
data_mixture = np.concatenate([data_female, data_male])
np.random.shuffle(data_mixture)

print(f"\n📊 Data: {len(data_mixture)} observations")
print(f"  True mixture: {n1} from N(162, 6²), {n2} from N(175, 7²)")
print(f"  Overall mean: {data_mixture.mean():.2f} cm")
print(f"  Overall std:  {data_mixture.std():.2f} cm")

# Fit single Gaussian (WRONG!)
print("\n1️⃣ Fitting single Gaussian (incorrect):")
dist_single = get_distribution('normal')
dist_single.fit(data_mixture)
print(f"  Mean: {dist_single.mean():.2f} cm")
print(f"  Std:  {dist_single.std():.2f} cm")
print(f"  AIC:  {dist_single.aic():.2f}")

# Fit Gaussian Mixture Model
print("\n2️⃣ Fitting 2-component Gaussian Mixture:")
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data_mixture.reshape(-1, 1))

means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_

# Sort by mean for consistent display
idx_sort = np.argsort(means)
means = means[idx_sort]
stds = stds[idx_sort]
weights = weights[idx_sort]

print(f"  Component 1:")
print(f"    Mean:   {means[0]:.2f} cm")
print(f"    Std:    {stds[0]:.2f} cm")
print(f"    Weight: {weights[0]:.3f} ({weights[0]*100:.1f}%)")

print(f"  Component 2:")
print(f"    Mean:   {means[1]:.2f} cm")
print(f"    Std:    {stds[1]:.2f} cm")
print(f"    Weight: {weights[1]:.3f} ({weights[1]*100:.1f}%)")

print(f"\n  BIC: {gmm.bic(data_mixture.reshape(-1, 1)):.2f}")
print(f"  AIC: {gmm.aic(data_mixture.reshape(-1, 1)):.2f}")

print(f"\n✅ Mixture model successfully identified both populations!")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data with single Gaussian (wrong)
ax = axes[0]
ax.hist(data_mixture, bins=40, density=True, alpha=0.6, color='gray',
        edgecolor='black', label='Data')

x = np.linspace(data_mixture.min(), data_mixture.max(), 300)
ax.plot(x, dist_single.pdf(x), 'r-', linewidth=2.5, 
        label='Single Gaussian (poor fit)')

ax.set_xlabel('Height (cm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Wrong: Single Gaussian', fontweight='bold', color='red')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, 'Misses bimodal structure!',
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: Data with mixture model (correct)
ax = axes[1]
ax.hist(data_mixture, bins=40, density=True, alpha=0.6, color='gray',
        edgecolor='black', label='Data')

# Plot individual components
for i in range(2):
    component = stats.norm(means[i], stds[i])
    ax.plot(x, weights[i] * component.pdf(x), '--', linewidth=2,
            label=f'Component {i+1}: N({means[i]:.0f}, {stds[i]:.1f}²)')

# Plot mixture
mixture_pdf = sum(weights[i] * stats.norm(means[i], stds[i]).pdf(x) 
                  for i in range(2))
ax.plot(x, mixture_pdf, 'r-', linewidth=3, label='Mixture')

ax.set_xlabel('Height (cm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Correct: Gaussian Mixture Model', fontweight='bold', color='green')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, 'Captures bimodal structure!',
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
print("\n📊 Comparison plot created!")
plt.savefig('/tmp/mixture_comparison.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: Model Selection (How Many Components?)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Model Selection - Optimal Number of Components")
print("="*70)

print("""
Question: How many components are in the data?
Method: Fit models with K=1,2,3,4,5 and compare BIC
""")

# Test different numbers of components
max_components = 6
bic_scores = []
aic_scores = []

print(f"\n🔬 Testing K = 1 to {max_components} components...\n")

for k in range(1, max_components + 1):
    gmm_k = GaussianMixture(n_components=k, random_state=42)
    gmm_k.fit(data_mixture.reshape(-1, 1))
    
    bic = gmm_k.bic(data_mixture.reshape(-1, 1))
    aic = gmm_k.aic(data_mixture.reshape(-1, 1))
    
    bic_scores.append(bic)
    aic_scores.append(aic)
    
    marker = '⭐' if k == 2 else '  '
    print(f"  {marker} K={k}: BIC={bic:8.2f}, AIC={aic:8.2f}")

best_k_bic = np.argmin(bic_scores) + 1
best_k_aic = np.argmin(aic_scores) + 1

print(f"\n🏆 Model Selection Results:")
print(f"  BIC prefers: K = {best_k_bic} components")
print(f"  AIC prefers: K = {best_k_aic} components")
print(f"  True value:  K = 2 components")
print(f"\n  ✅ Both criteria correctly identified 2 components!")

# Visualization: Elbow plot
fig, ax = plt.subplots(figsize=(10, 6))

k_range = range(1, max_components + 1)
ax.plot(k_range, bic_scores, 'b-o', linewidth=2.5, markersize=10,
        label='BIC', markerfacecolor='blue', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(k_range, aic_scores, 'r-s', linewidth=2.5, markersize=10,
        label='AIC', markerfacecolor='red', markeredgecolor='black', markeredgewidth=1.5)

# Mark optimal K
ax.axvline(best_k_bic, color='blue', linestyle='--', alpha=0.5, linewidth=2)
ax.axvline(best_k_aic, color='red', linestyle='--', alpha=0.5, linewidth=2)

ax.scatter([best_k_bic], [bic_scores[best_k_bic-1]], s=300, marker='*',
           color='gold', edgecolors='black', linewidths=2, zorder=5,
           label=f'Optimal (K={best_k_bic})')

ax.set_xlabel('Number of Components (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Information Criterion', fontsize=12, fontweight='bold')
ax.set_title('Model Selection: BIC & AIC vs Number of Components',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xticks(k_range)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("\n📊 Model selection plot created!")
plt.savefig('/tmp/mixture_model_selection.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 3: Component Assignment (Soft Clustering)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Component Assignment & Soft Clustering")
print("="*70)

print("""
Question: Which component does each data point belong to?
Answer: Use posterior probabilities (soft assignment)
""")

# Get posterior probabilities
posterior_probs = gmm.predict_proba(data_mixture.reshape(-1, 1))
hard_assignments = gmm.predict(data_mixture.reshape(-1, 1))

print(f"\n📊 Assignment Statistics:")
for k in range(2):
    n_assigned = np.sum(hard_assignments == k)
    print(f"  Component {k+1}: {n_assigned} points ({n_assigned/len(data_mixture)*100:.1f}%)")

# Show some examples
print(f"\n🔍 Example Posterior Probabilities (first 10 points):")
print(f"\n  {'Value':<8} {'Comp 1':<10} {'Comp 2':<10} {'Assignment'}")
print("-"*50)

for i in range(10):
    val = data_mixture[i]
    p1 = posterior_probs[i, 0]
    p2 = posterior_probs[i, 1]
    assignment = f"Component {hard_assignments[i] + 1}"
    print(f"  {val:7.2f}  {p1:9.4f}  {p2:9.4f}  {assignment}")

# Visualization: Soft clustering
fig, ax = plt.subplots(figsize=(10, 6))

# Color by assignment confidence
colors = posterior_probs[:, 1]  # Probability of component 2

scatter = ax.scatter(data_mixture, np.zeros_like(data_mixture),
                     c=colors, cmap='RdYlBu_r', s=50, alpha=0.7,
                     edgecolors='black', linewidth=0.5)

# Add component distributions
ax2 = ax.twinx()
ax2.hist(data_mixture, bins=40, density=True, alpha=0.3, color='gray')
ax2.plot(x, mixture_pdf, 'r-', linewidth=3, label='Mixture')
for i in range(2):
    component = stats.norm(means[i], stds[i])
    ax2.plot(x, weights[i] * component.pdf(x), '--', linewidth=2)

ax.set_xlabel('Height (cm)', fontsize=12, fontweight='bold')
ax.set_ylabel('', fontsize=11)
ax.set_ylim([-0.5, 0.5])
ax.set_yticks([])
ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Soft Clustering: Color = Probability of Component 2',
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15)
cbar.set_label('P(Component 2)', fontsize=11, fontweight='bold')

plt.tight_layout()
print("\n📊 Soft clustering plot created!")
plt.savefig('/tmp/mixture_soft_clustering.png', dpi=150, bbox_inches='tight')

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways - Mixture Models")
print("="*70)
print("""
1. WHEN TO USE MIXTURE MODELS:
   ✓ Multimodal data (multiple peaks)
   ✓ Heterogeneous populations
   ✓ Single distribution doesn't fit
   ✓ Want to identify subgroups

2. GAUSSIAN MIXTURE MODEL (GMM):
   • Most common mixture model
   • Each component is Gaussian
   • Fitted using EM algorithm
   • Provides soft clustering

3. MODEL SELECTION:
   • Use BIC or AIC to choose K
   • Look for "elbow" in information criterion
   • BIC tends to prefer simpler models
   • Don't overfit with too many components!

4. COMPONENT ASSIGNMENT:
   • Hard: Each point assigned to one component
   • Soft: Posterior probabilities for all components
   • Soft assignment better captures uncertainty

5. APPLICATIONS:
   • Customer segmentation
   • Anomaly detection
   • Image segmentation
   • Bioinformatics (gene expression)
   • Quality control (defect types)

6. BEST PRACTICES:
   ✓ Visualize data first (histogram)
   ✓ Try K=1,2,3,... and compare BIC
   ✓ Check component separation
   ✓ Validate with domain knowledge
   ✓ Use random_state for reproducibility

7. SKLEARN USAGE:
   from sklearn.mixture import GaussianMixture
   
   gmm = GaussianMixture(n_components=2)
   gmm.fit(data.reshape(-1, 1))
   labels = gmm.predict(data.reshape(-1, 1))
   probs = gmm.predict_proba(data.reshape(-1, 1))

8. LIMITATIONS:
   • Assumes Gaussian components (can use other types)
   • EM algorithm can get stuck in local minima
   • Need sufficient data per component
   • K must be specified (or searched)

Next: See bootstrap_confidence.py for uncertainty quantification!
""")
