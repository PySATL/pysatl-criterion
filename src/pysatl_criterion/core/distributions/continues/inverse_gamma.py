"""Inverse gamma distribution with zero location."""

from scipy.stats import invgamma


def generate_inverse_gamma(size, *, alpha=1.0, beta=1.0, random_state=None):
    """Generate samples with explicit shape, scale, and optional random state."""
    return invgamma.rvs(a=alpha, scale=beta, size=size, random_state=random_state)
