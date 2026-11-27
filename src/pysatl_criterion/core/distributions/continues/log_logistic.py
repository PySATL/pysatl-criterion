"""Log-logistic distribution with zero location."""

from scipy.stats import fisk


def generate_log_logistic(size, *, alpha=1.0, beta=1.0, random_state=None):
    """Generate samples with explicit scale, shape, and optional random state."""
    return fisk.rvs(c=beta, scale=alpha, size=size, random_state=random_state)
