"""Exponentiated weibull distribution with zero location."""

from scipy.stats import exponweib


def generate_exponentiated_weibull(size, *, exponent=1, shape=5, scale=1, random_state=None):
    """Generate samples with an explicit scale and optional random state."""
    return exponweib.rvs(c=shape, a=exponent, scale=scale, size=size, random_state=random_state)


def generate_exponentiated_weibull_cdf(rvs, *, exponent=1, shape=5, scale=1):
    """Evaluate cdf for the specified distribution."""
    return exponweib.cdf(rvs, c=shape, a=exponent, scale=scale)


def generate_exponentiated_weibull_logcdf(rvs, *, exponent=1, shape=5, scale=1):
    """Evaluate logcdf for the specified distribution."""
    return exponweib.logcdf(rvs, c=shape, a=exponent, scale=scale)


def generate_exponentiated_weibull_logsf(rvs, *, exponent=1, shape=5, scale=1):
    """Evaluate logsf for the specified distribution."""
    return exponweib.logsf(rvs, c=shape, a=exponent, scale=scale)
