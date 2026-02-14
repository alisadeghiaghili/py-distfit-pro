#!/usr/bin/env python3
"""
Interactive Plots with Plotly
=============================

Create interactive visualizations using Plotly:
  - Zoom/pan capabilities
  - Hover information
  - Export to HTML
  - Professional dashboards

Requires: plotly
  Install: pip install plotly

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not installed. Install with: pip install plotly")

np.random.seed(42)

print("="*70)
print("🔥 INTERACTIVE PLOTS WITH PLOTLY")
print("="*70)

if not PLOTLY_AVAILABLE:
    print("\n❌ Plotly is required for this example.")
    print("   Install: pip install plotly")
    print("\nExiting...")
    exit(1)


# ============================================================================
# Example 1: Interactive PDF Plot
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Interactive PDF Plot")
print("="*70)

# Generate and fit data
data = np.random.normal(loc=100, scale=15, size=1000)
dist = get_distribution('normal')
dist.fit(data)

print(f"\n📊 Data: {len(data)} samples from N(100, 15²)")
print(f"🎨 Creating interactive PDF plot...")

# Create figure
fig = go.Figure()

# Histogram
fig.add_trace(go.Histogram(
    x=data,
    nbinsx=50,
    name='Data Histogram',
    opacity=0.6,
    marker_color='lightblue',
    marker_line_color='black',
    marker_line_width=1,
    histnorm='probability density',
    hovertemplate='Value: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>'
))

# Fitted PDF
x = np.linspace(data.min(), data.max(), 300)
y_pdf = dist.pdf(x)

fig.add_trace(go.Scatter(
    x=x,
    y=y_pdf,
    mode='lines',
    name='Fitted PDF',
    line=dict(color='red', width=3),
    hovertemplate='Value: %{x:.2f}<br>PDF: %{y:.4f}<extra></extra>'
))

# Mean line
mean_val = dist.mean()
fig.add_vline(
    x=mean_val,
    line_dash='dash',
    line_color='green',
    line_width=2,
    annotation_text=f'Mean = {mean_val:.2f}',
    annotation_position='top'
)

# Layout
fig.update_layout(
    title=dict(
        text='Interactive PDF Plot<br><sub>Hover for details | Click legend to toggle | Drag to zoom</sub>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Value',
    yaxis_title='Probability Density',
    template='plotly_white',
    hovermode='closest',
    showlegend=True,
    width=900,
    height=600,
    font=dict(size=12)
)

fig.write_html('/tmp/interactive_pdf.html')
print("✅ Interactive PDF plot created: /tmp/interactive_pdf.html")
print("   Open in browser to interact!")


# ============================================================================
# Example 2: Interactive CDF with Percentiles
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Interactive CDF with Percentiles")
print("="*70)

print(f"🎨 Creating interactive CDF plot...")

fig = go.Figure()

# Empirical CDF
data_sorted = np.sort(data)
empirical_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

fig.add_trace(go.Scatter(
    x=data_sorted,
    y=empirical_cdf,
    mode='markers',
    name='Empirical CDF',
    marker=dict(size=3, color='blue', opacity=0.4),
    hovertemplate='Value: %{x:.2f}<br>Cumulative Prob: %{y:.4f}<extra></extra>'
))

# Theoretical CDF
x_cdf = np.linspace(data.min(), data.max(), 500)
y_cdf = dist.cdf(x_cdf)

fig.add_trace(go.Scatter(
    x=x_cdf,
    y=y_cdf,
    mode='lines',
    name='Theoretical CDF',
    line=dict(color='red', width=3),
    hovertemplate='Value: %{x:.2f}<br>CDF: %{y:.4f}<extra></extra>'
))

# Add percentile markers
percentiles = [0.25, 0.5, 0.75, 0.95]
colors = ['orange', 'green', 'purple', 'red']

for p, color in zip(percentiles, colors):
    val = dist.ppf(p)
    
    # Vertical line
    fig.add_trace(go.Scatter(
        x=[val, val],
        y=[0, p],
        mode='lines',
        line=dict(color=color, dash='dot', width=1.5),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Horizontal line
    fig.add_trace(go.Scatter(
        x=[data.min(), val],
        y=[p, p],
        mode='lines',
        line=dict(color=color, dash='dot', width=1.5),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Marker
    fig.add_trace(go.Scatter(
        x=[val],
        y=[p],
        mode='markers',
        name=f'P{int(p*100)} = {val:.2f}',
        marker=dict(size=12, color=color, line=dict(width=2, color='white')),
        hovertemplate=f'P{int(p*100)}<br>Value: {val:.2f}<br>Probability: {p:.2f}<extra></extra>'
    ))

fig.update_layout(
    title='Interactive CDF with Percentiles<br><sub>Hover over markers for percentile info</sub>',
    xaxis_title='Value',
    yaxis_title='Cumulative Probability',
    template='plotly_white',
    hovermode='closest',
    width=900,
    height=600,
    yaxis=dict(range=[0, 1.05])
)

fig.write_html('/tmp/interactive_cdf.html')
print("✅ Interactive CDF plot created: /tmp/interactive_cdf.html")


# ============================================================================
# Example 3: Multi-Distribution Comparison Dashboard
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Interactive Distribution Comparison")
print("="*70)

# Generate slightly skewed data
data_compare = np.random.lognormal(mean=3, sigma=0.5, size=800)

print(f"📊 Data: {len(data_compare)} samples (right-skewed)")
print(f"🎨 Creating multi-distribution comparison dashboard...")

# Fit multiple distributions
dists_names = ['lognormal', 'gamma', 'weibull_min', 'expon']
dists_fitted = []

for dname in dists_names:
    d = get_distribution(dname)
    d.fit(data_compare)
    dists_fitted.append((dname, d))
    print(f"   ✓ Fitted {dname}: AIC = {d.aic():.2f}")

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('PDF Comparison', 'CDF Comparison', 
                    'AIC Comparison', 'Q-Q Plot (Best Model)'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'scatter'}]]
)

# 1. PDF Comparison
fig.add_trace(
    go.Histogram(x=data_compare, nbinsx=40, name='Data', 
                 histnorm='probability density', opacity=0.4,
                 marker_color='gray', showlegend=True),
    row=1, col=1
)

x_comp = np.linspace(data_compare.min(), data_compare.max(), 300)
colors_dist = ['red', 'blue', 'green', 'orange']

for (dname, d), color in zip(dists_fitted, colors_dist):
    fig.add_trace(
        go.Scatter(x=x_comp, y=d.pdf(x_comp), mode='lines',
                   name=dname, line=dict(color=color, width=2)),
        row=1, col=1
    )

# 2. CDF Comparison
data_comp_sorted = np.sort(data_compare)
emp_cdf_comp = np.arange(1, len(data_comp_sorted) + 1) / len(data_comp_sorted)

fig.add_trace(
    go.Scatter(x=data_comp_sorted, y=emp_cdf_comp, mode='markers',
               name='Empirical', marker=dict(size=2, color='gray'),
               showlegend=False),
    row=1, col=2
)

for (dname, d), color in zip(dists_fitted, colors_dist):
    fig.add_trace(
        go.Scatter(x=x_comp, y=d.cdf(x_comp), mode='lines',
                   name=dname, line=dict(color=color, width=2),
                   showlegend=False),
        row=1, col=2
    )

# 3. AIC Comparison
aic_values = [d.aic() for _, d in dists_fitted]
fig.add_trace(
    go.Bar(x=dists_names, y=aic_values, 
           marker_color=colors_dist, showlegend=False,
           text=[f'{aic:.1f}' for aic in aic_values],
           textposition='outside'),
    row=2, col=1
)

# 4. Q-Q Plot for best model
best_dist_idx = np.argmin(aic_values)
best_name, best_dist = dists_fitted[best_dist_idx]

percentiles_qq = np.linspace(0.01, 0.99, len(data_compare))
theoretical_qq = best_dist.ppf(percentiles_qq)
empirical_qq = np.sort(data_compare)

fig.add_trace(
    go.Scatter(x=theoretical_qq, y=empirical_qq, mode='markers',
               name=f'Best: {best_name}',
               marker=dict(size=5, color=colors_dist[best_dist_idx])),
    row=2, col=2
)

# 45° line
min_qq = min(theoretical_qq.min(), empirical_qq.min())
max_qq = max(theoretical_qq.max(), empirical_qq.max())
fig.add_trace(
    go.Scatter(x=[min_qq, max_qq], y=[min_qq, max_qq], mode='lines',
               line=dict(color='black', dash='dash', width=2),
               name='Perfect Fit', showlegend=False),
    row=2, col=2
)

# Update axes
fig.update_xaxes(title_text='Value', row=1, col=1)
fig.update_yaxes(title_text='Density', row=1, col=1)
fig.update_xaxes(title_text='Value', row=1, col=2)
fig.update_yaxes(title_text='Cumulative Prob', row=1, col=2)
fig.update_xaxes(title_text='Distribution', row=2, col=1)
fig.update_yaxes(title_text='AIC (lower = better)', row=2, col=1)
fig.update_xaxes(title_text='Theoretical Quantiles', row=2, col=2)
fig.update_yaxes(title_text='Sample Quantiles', row=2, col=2)

fig.update_layout(
    title_text=f'Distribution Comparison Dashboard<br><sub>Best: {best_name} (AIC={aic_values[best_dist_idx]:.2f})</sub>',
    template='plotly_white',
    height=800,
    width=1200,
    showlegend=True
)

fig.write_html('/tmp/interactive_dashboard.html')
print("✅ Interactive dashboard created: /tmp/interactive_dashboard.html")


# ============================================================================
# Example 4: 3D Surface Plot (PDF over parameter space)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: 3D Interactive Surface (Normal PDF)")
print("="*70)

print(f"🎨 Creating 3D surface plot...")

# Create parameter grid
mu_range = np.linspace(90, 110, 50)
sigma_range = np.linspace(5, 25, 50)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)

# Fixed point to evaluate PDF
x_point = 100

# Calculate PDF values
Z = np.zeros_like(MU)
for i in range(len(sigma_range)):
    for j in range(len(mu_range)):
        d_temp = get_distribution('normal')
        d_temp.params = {'loc': MU[i, j], 'scale': SIGMA[i, j]}
        Z[i, j] = d_temp.pdf(x_point)

fig = go.Figure(data=[go.Surface(
    x=MU,
    y=SIGMA,
    z=Z,
    colorscale='Viridis',
    hovertemplate='μ: %{x:.1f}<br>σ: %{y:.1f}<br>PDF: %{z:.6f}<extra></extra>'
)])

fig.update_layout(
    title=f'3D Surface: Normal PDF at x={x_point}<br><sub>Rotate with mouse | Zoom with scroll</sub>',
    scene=dict(
        xaxis_title='Mean (μ)',
        yaxis_title='Std Dev (σ)',
        zaxis_title=f'PDF(x={x_point})',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    width=900,
    height=700
)

fig.write_html('/tmp/interactive_3d_surface.html')
print("✅ 3D surface plot created: /tmp/interactive_3d_surface.html")


print("\n" + "="*70)
print("📁 Files Saved")
print("="*70)
print("""
Interactive HTML files saved to /tmp/:
  1. interactive_pdf.html
  2. interactive_cdf.html
  3. interactive_dashboard.html
  4. interactive_3d_surface.html

Open these files in your web browser for interactive exploration!
""")

print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. PLOTLY ADVANTAGES:
   ✓ Interactive zoom/pan
   ✓ Hover information
   ✓ Click to toggle traces
   ✓ Export to PNG/SVG
   ✓ Embed in HTML/web apps
   ✓ Professional appearance

2. USE CASES:
   • Exploratory analysis
   • Web dashboards
   • Presentations (live demos)
   • Sharing results (HTML)
   • Client-facing reports

3. PLOTLY vs MATPLOTLIB:
   Plotly:
     + Interactive out-of-the-box
     + Beautiful defaults
     + Easy web integration
     - Larger file sizes
     - Slower for huge datasets
   
   Matplotlib:
     + Publication-quality static plots
     + Fine-grained control
     + Better for print
     - Requires mpld3 for interactivity

4. INSTALLATION:
   pip install plotly
   pip install plotly kaleido  # For static export

5. BASIC USAGE:
   import plotly.graph_objects as go
   
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=x, y=y))
   fig.update_layout(title='My Plot')
   fig.show()  # Interactive in Jupyter
   fig.write_html('plot.html')  # Save

6. BEST PRACTICES:
   ✓ Add hover templates for context
   ✓ Use subplot titles
   ✓ Choose appropriate templates (plotly_white, etc.)
   ✓ Set reasonable figure sizes
   ✓ Use color consistently

7. ADVANCED FEATURES:
   • Animations (time series)
   • 3D plots (scatter, surface, mesh)
   • Maps (choropleth, scatter_geo)
   • Statistical charts (box, violin)
   • Real-time updates

Next: See 06_real_world/ for practical applications!
""")
