from scipy.stats import logistic


def generate_logistic(size, t=0, s=1):
    """
    Generate random samples from the logistic distribution.

    This function uses ``scipy.stats.logistic.rvs`` to draw random values
    from a logistic distribution parameterized by location ``t`` and
    scale ``s``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    t : float, optional
        Location parameter (mean/median) of the distribution.
        Default is 0.
    s : float, optional
        Scale parameter of the distribution. Must be greater than 0.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the logistic distribution.

    Raises
    ------
    ValueError
        If ``s <= 0``.

    Notes
    -----
    The logistic distribution is symmetric and similar in shape to the
    normal distribution but has heavier tails. It is commonly used in
    logistic regression and growth models.

    Examples
    --------
    >>> generate_logistic(5, t=0.0, s=1.0)
    array([...])
    """
    return logistic.rvs(size=size, loc=t, scale=s)
