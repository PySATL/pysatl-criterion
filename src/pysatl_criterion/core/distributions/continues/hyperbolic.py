"""Hyperbolic distribution."""

from scipy.stats import genhyperbolic


def generate_hyperbolic(size, *, alpha=1.0, beta=0.0, delta=1.0, mu=0.0, random_state=None):
    """Generate samples with explicit parameters and optional random state."""
    return genhyperbolic.rvs(
        p=1,
        a=alpha * delta,
        b=beta * delta,
        loc=mu,
        scale=delta,
        size=size,
        random_state=random_state,
    )
