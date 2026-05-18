import math

import numpy as np
from scipy.stats import lognorm


def generate_lognorm(size, mu=0, s=1):
    """
    Generate random samples from the log-normal distribution.

    This function uses ``scipy.stats.lognorm.rvs`` to draw random values
    from a log-normal distribution. If a random variable X is log-normally
    distributed, then ln(X) follows a normal distribution.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    mu : float, optional
        Mean of the underlying normal distribution (in log-space).
        The scale parameter is computed as ``exp(mu)``. Default is 0.
    s : float, optional
        Variance (or dispersion parameter) of the underlying normal
        distribution. The standard deviation used by SciPy is
        ``sqrt(s)``. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the log-normal distribution.

    Raises
    ------
    ValueError
        If ``s < 0``.

    Notes
    -----
    The log-normal distribution is commonly used to model positive-valued
    skewed data such as incomes, lifetimes, and biological measurements.

    Examples
    --------
    >>> generate_lognorm(5, mu=0.0, s=1.0)
    array([...])
    """
    scale = math.exp(mu)
    return lognorm.rvs(s=np.sqrt(s), size=size, scale=scale)
