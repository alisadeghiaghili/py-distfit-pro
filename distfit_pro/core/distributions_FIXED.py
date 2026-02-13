# اینجا فقط 3 کلاس fix شده رو می‌ذارم به عنوان مرجع
# بعدا این رو توی فایل اصلی merge می‌کنیم

# FIXED WEIBULL:
class WeibullDistribution_FIXED:
    def _fit_mle(self, data, **kwargs):
        if np.any(data < 0):
            raise ValueError("Weibull distribution requires non-negative data")
        shape, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'c': shape, 'scale': scale}
    
    def _fit_mom(self, data, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self):
        c, scale = self._params['c'], self._params['scale']
        if c > 1:
            return scale * ((c - 1) / c) ** (1 / c)
        return 0.0
 
    def _get_scipy_params(self):
        return {'c': self._params['c'], 'loc': 0, 'scale': self._params['scale']}


# FIXED LOGNORMAL:
class LognormalDistribution_FIXED:
    def _fit_mle(self, data, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Lognormal distribution requires positive data")
        shape, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'s': shape, 'scale': scale}
    
    def _fit_mom(self, data, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Lognormal distribution requires positive data")
        log_data = np.log(data)
        s = np.std(log_data, ddof=1)
        scale = np.exp(np.mean(log_data))
        self._params = {'s': s, 'scale': scale}
    
    def mode(self):
        s, scale = self._params['s'], self._params['scale']
        return scale * np.exp(-s**2)
    
    def _get_scipy_params(self):
        return {'s': self._params['s'], 'loc': 0, 'scale': self._params['scale']}


# FIXED HYPERGEOMETRIC:
# scipy.hypergeom(M, n, N) where:
#   M = total population
#   n = number of success states in population  
#   N = number of draws
class HypergeometricDistribution_FIXED:
    def _fit_mle(self, data, N, n, **kwargs):
        # User provides: N=population, n=draws
        # We need to estimate success_states from data mean
        if N is None or n is None:
            raise ValueError("Hypergeometric requires N (population) and n (draws)")
        
        # E[X] = n * success_states / N
        # So: success_states = E[X] * N / n
        mean_data = np.mean(data)
        success_states = int(np.round(mean_data * N / n))
        
        # Store as M=N, n=success_states, N=n for scipy compatibility
        self._params = {'M': N, 'n': success_states, 'N': n}
    
    def _fit_mom(self, data, N, n, **kwargs):
        self._fit_mle(data, N=N, n=n, **kwargs)
    
    def mode(self):
        M, n, N = self._params['M'], self._params['n'], self._params['N']
        return np.floor((N + 1) * (n + 1) / (M + 2))
    
    def _get_scipy_params(self):
        return {'M': self._params['M'], 'n': self._params['n'], 'N': self._params['N']}
