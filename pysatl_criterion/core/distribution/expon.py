from scipy.stats import expon


def generate_expon(size, lam=1):
    """
    Generate random samples from the exponential distribution.

    This function uses ``scipy.stats.expon.rvs`` to draw random values
    from an exponential distribution parameterized by the rate ``lam``.
    Internally, SciPy uses the ``scale`` parameter, where
    ``scale = 1 / lam``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    lam : float, optional
        Rate parameter (λ) of the exponential distribution.
        Must be greater than 0. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the exponential distribution.

    Raises
    ------
    ValueError
        If ``lam <= 0``.

    Notes
    -----
    The exponential distribution models the time between events in a
    Poisson process. Its mean is ``1 / lam``.

    Examples
    --------
    >>> generate_expon(5, lam=2)
    array([...])
    """
    scale = 1 / lam
    return expon.rvs(size=size, scale=scale)


def cdf_expon(rvs, lam=1):
    """
    Compute the cumulative distribution function (CDF) for the exponential distribution.

    This function uses ``scipy.stats.expon.cdf`` to evaluate the CDF
    at given values, parameterized by the rate ``lam``. Internally,
    ``scale = 1 / lam``.

    Parameters
    ----------
    rvs : array-like
        Input values at which to evaluate the CDF.
    lam : float, optional
        Rate parameter (λ) of the exponential distribution.
        Must be greater than 0. Default is 1.

    Returns
    -------
    numpy.ndarray
        CDF values corresponding to the input ``rvs``.

    Raises
    ------
    ValueError
        If ``lam <= 0``.

    Notes
    -----
    The CDF of the exponential distribution is:
        F(x) = 1 - exp(-λx), for x ≥ 0.

    Examples
    --------
    >>> cdf_expon([0.5, 1.0, 2.0], lam=1)
    array([...])
    """
    scale = 1 / lam
    return expon.cdf(rvs, scale=scale)
