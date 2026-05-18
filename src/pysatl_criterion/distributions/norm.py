import numpy as np
from scipy.stats import norm


def generate_norm(size, mean=0, var=1):
    """
    Generate random samples from the normal (Gaussian) distribution.

    This function uses ``scipy.stats.norm.rvs`` to draw random values
    from a normal distribution parameterized by mean and variance.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    mean : float, optional
        Mean (μ) of the normal distribution. Default is 0.
    var : float, optional
        Variance (σ²) of the normal distribution. Must be non-negative.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the normal distribution.

    Raises
    ------
    ValueError
        If ``var < 0``.

    Notes
    -----
    The normal distribution is one of the most commonly used probability
    distributions in statistics due to the Central Limit Theorem.

    Examples
    --------
    >>> generate_norm(5, mean=0, var=1)
    array([...])
    """
    return norm.rvs(size=size, loc=mean, scale=np.sqrt(var))


def cdf_norm(rvs, mean=0, var=1):
    """
    Compute the cumulative distribution function (CDF) of the normal distribution.

    This function uses ``scipy.stats.norm.cdf`` to evaluate the probability
    that a normally distributed random variable is less than or equal to
    given values.

    Parameters
    ----------
    rvs : array-like
        Input values at which to evaluate the CDF.
    mean : float, optional
        Mean (μ) of the normal distribution. Default is 0.
    var : float, optional
        Variance (σ²) of the normal distribution. Must be non-negative.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        CDF values corresponding to the input ``rvs``.

    Raises
    ------
    ValueError
        If ``var < 0``.

    Examples
    --------
    >>> cdf_norm([0, 1], mean=0, var=1)
    array([...])
    """
    return norm.cdf(rvs, loc=mean, scale=np.sqrt(var))


def pdf_norm(rvs, mean=0, var=1):
    """
    Compute the probability density function (PDF) of the normal distribution.

    This function uses ``scipy.stats.norm.pdf`` to evaluate the density
    of the normal distribution at given values.

    Parameters
    ----------
    rvs : array-like
        Input values at which to evaluate the PDF.
    mean : float, optional
        Mean (μ) of the normal distribution. Default is 0.
    var : float, optional
        Variance (σ²) of the normal distribution. Must be non-negative.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        PDF values corresponding to the input ``rvs``.

    Raises
    ------
    ValueError
        If ``var < 0``.

    Examples
    --------
    >>> pdf_norm([0, 1], mean=0, var=1)
    array([...])
    """
    return norm.pdf(rvs, loc=mean, scale=np.sqrt(var))
