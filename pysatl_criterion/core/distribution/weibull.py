from scipy.stats import exponweib


def generate_weibull(size, a=1, k=5):
    """
    Generate random samples from the exponentiated Weibull distribution.

    This function uses ``scipy.stats.exponweib.rvs`` to draw random values
    from an exponentiated Weibull distribution with parameters ``a`` and ``k``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    a : float, optional
        Shape parameter of the distribution. Default is 1.
    k : float, optional
        Shape parameter controlling tail behavior. Default is 5.

    Returns
    -------
    numpy.ndarray
        Random samples drawn from the exponentiated Weibull distribution.

    Notes
    -----
    The exponentiated Weibull distribution generalizes the Weibull
    distribution and can model a wider range of hazard rate behaviors.

    Examples
    --------
    >>> generate_weibull(5, a=1, k=2)
    array([...])
    """
    return exponweib.rvs(a=a, c=k, size=size)


def generate_weibull_cdf(rvs, a=1, k=5):
    """
    Compute the cumulative distribution function (CDF) of the exponentiated Weibull distribution.

    Parameters
    ----------
    rvs : array-like
        Input values at which to evaluate the CDF.
    a : float, optional
        Shape parameter. Default is 1.
    k : float, optional
        Shape parameter. Default is 5.

    Returns
    -------
    numpy.ndarray
        CDF values corresponding to the input data.

    Examples
    --------
    >>> generate_weibull_cdf([0.5, 1.0], a=1, k=2)
    array([...])
    """
    return exponweib.cdf(rvs, a=a, c=k)


def generate_weibull_logcdf(rvs, a=1, k=5):
    """
    Compute the log cumulative distribution function (log-CDF)
    of the exponentiated Weibull distribution.

    Parameters
    ----------
    rvs : array-like
        Input values at which to evaluate the log-CDF.
    a : float, optional
        Shape parameter. Default is 1.
    k : float, optional
        Shape parameter. Default is 5.

    Returns
    -------
    numpy.ndarray
        Log-CDF values.

    Examples
    --------
    >>> generate_weibull_logcdf([0.5, 1.0], a=1, k=2)
    array([...])
    """
    return exponweib.logcdf(rvs, a=a, c=k)


def generate_weibull_logsf(rvs, a=1, k=5):
    """
    Compute the log survival function (log-SF) of the exponentiated Weibull distribution.

    Parameters
    ----------
    rvs : array-like
        Input values at which to evaluate the log-SF.
    a : float, optional
        Shape parameter. Default is 1.
    k : float, optional
        Shape parameter. Default is 5.

    Returns
    -------
    numpy.ndarray
        Log-survival function values.

    Notes
    -----
    The survival function is defined as S(x) = 1 - F(x).

    Examples
    --------
    >>> generate_weibull_logsf([0.5, 1.0], a=1, k=2)
    array([...])
    """
    return exponweib.logsf(rvs, a=a, c=k)
