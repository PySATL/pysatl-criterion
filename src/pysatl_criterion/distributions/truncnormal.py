import numpy as np
from scipy.stats import truncnorm


def generate_truncnorm(size, mean=0, var=1, a=-10, b=10):
    """
    Generate random samples from the truncated normal distribution.

    This function uses ``scipy.stats.truncnorm.rvs`` to draw samples
    from a normal distribution truncated to the interval [a, b],
    with given mean and variance.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    mean : float, optional
        Mean (μ) of the underlying normal distribution. Default is 0.
    var : float, optional
        Variance (σ²) of the underlying normal distribution.
        Must be non-negative. Default is 1.
    a : float, optional
        Lower truncation bound. Default is -10.
    b : float, optional
        Upper truncation bound. Default is 10.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the truncated normal distribution.

    Raises
    ------
    ValueError
        If ``var < 0`` or ``a >= b``.

    Notes
    -----
    The truncated normal distribution is useful when values are naturally
    bounded but still follow approximately Gaussian behavior inside the
    interval.

    Examples
    --------
    >>> generate_truncnorm(5, mean=0, var=1, a=-2, b=2)
    array([...])
    """
    return truncnorm.rvs(a=a, b=b, size=size, loc=mean, scale=np.sqrt(var))
