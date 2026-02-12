Frequently Asked Questions
==========================

General Questions
-----------------

**Q: Which distribution should I use?**

A: Start with the data characteristics:

- Symmetric, no bounds → Normal
- Positive, right-skewed → Lognormal, Gamma
- Bounded [0,1] → Beta
- Count data → Poisson, Binomial
- Time-to-event → Weibull, Exponential

Then test with GOF tests and compare AIC/BIC.

**Q: MLE vs Moments vs Quantile - which to use?**

A: 

- Default: MLE (most accurate)
- MLE fails: Moments (fast, always works)
- Outliers present: Quantile (robust)

**Q: How many bootstrap samples do I need?**

A:

- Quick check: 1000
- Publication: 5000-10000
- Critical decisions: 10000+

Fitting Issues
--------------

**Q: Fit failed with "ValueError: invalid parameters"**

A: Try:

1. Check data range matches distribution support
2. Remove NaN/Inf values
3. Try moments method instead of MLE
4. Scale data to reasonable range

**Q: All GOF tests reject my fit**

A:

1. Try different distributions
2. Check for outliers (may need removal/robust method)
3. Consider mixture distributions
4. Data may not follow any standard distribution

**Q: Bootstrap CI is very wide**

A: This indicates:

- High parameter uncertainty (normal with small samples)
- Poor fit (try different distribution)
- Heavy-tailed data (expected)

Performance
-----------

**Q: Bootstrap is slow**

A:

- Use n_jobs=-1 for parallel processing
- Reduce n_bootstrap (1000 is usually enough)
- Use parametric instead of non-parametric

**Q: Can I use GPU?**

A: Not currently. Planned for v2.0.

Weighted Data
-------------

**Q: What weights should I use?**

A:

- Survey: sampling weights
- Precision: 1/variance
- Frequency: counts

**Q: Weighted fit gives different results than unweighted**

A: This is expected! Weights change the emphasis on different observations.
