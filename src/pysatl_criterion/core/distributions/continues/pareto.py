"""Pareto distribution."""

from scipy.stats import pareto


def generate_pareto(size, *, shape=1.0, scale=1.0, random_state=None):
    """Generate samples with explicit shape, scale, and optional random state."""
    return pareto.rvs(b=shape, scale=scale, size=size, random_state=random_state)
