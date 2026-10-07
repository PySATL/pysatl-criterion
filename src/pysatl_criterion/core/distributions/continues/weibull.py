"""Weibull distribution with zero location."""

from scipy.stats import weibull_min


def generate_weibull(size, *, shape=5, scale=1, random_state=None):
    """Generate samples with an explicit scale and optional random state."""
    return weibull_min.rvs(c=shape, scale=scale, size=size, random_state=random_state)


def generate_weibull_cdf(rvs, *, shape=5, scale=1):
    """Evaluate cdf for the specified distribution."""
    return weibull_min.cdf(rvs, c=shape, scale=scale)


def generate_weibull_logcdf(rvs, *, shape=5, scale=1):
    """Evaluate logcdf for the specified distribution."""
    return weibull_min.logcdf(rvs, c=shape, scale=scale)


def generate_weibull_logsf(rvs, *, shape=5, scale=1):
    """Evaluate logsf for the specified distribution."""
    return weibull_min.logsf(rvs, c=shape, scale=scale)
