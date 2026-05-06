from scipy.stats import gumbel_r


def generate_gumbel(size, mu=0, beta=1):
    """
    Generate random samples from the Gumbel (right-skewed) distribution.

    This function uses ``scipy.stats.gumbel_r.rvs`` to draw random values
    from a Gumbel distribution parameterized by location ``mu`` and
    scale ``beta``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    mu : float, optional
        Location parameter of the distribution. Default is 0.
    beta : float, optional
        Scale parameter of the distribution. Must be greater than 0.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Gumbel distribution.

    Raises
    ------
    ValueError
        If ``beta <= 0``.

    Notes
    -----
    The Gumbel distribution is commonly used to model the distribution
    of extreme values (e.g., maxima of samples). It is a special case
    of the generalized extreme value distribution.

    Examples
    --------
    >>> generate_gumbel(5, mu=0.0, beta=2.0)
    array([...])
    """
    return gumbel_r.rvs(size=size, loc=mu, scale=beta)
