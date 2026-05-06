from scipy.stats import chi2


def generate_chi2(size, df=2):
    """
    Generate random samples from the Chi-square distribution.

    This function uses ``scipy.stats.chi2.rvs`` to draw random values
    from a Chi-square distribution with the specified degrees of freedom.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    df : float, optional
        Degrees of freedom of the Chi-square distribution.
        Must be greater than 0. Default is 2.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Chi-square distribution.

    Raises
    ------
    ValueError
        If ``df <= 0``.

    Notes
    -----
    The Chi-square distribution is a special case of the Gamma distribution
    and is commonly used in statistical hypothesis testing and confidence
    interval estimation.

    Examples
    --------
    >>> generate_chi2(5, df=3)
    array([...])
    """
    return chi2.rvs(df=df, size=size)
