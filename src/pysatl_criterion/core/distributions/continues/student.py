from scipy.stats import t


def generate_t(size, df=2, random_state=None, *, loc=0, scale=1):
    """
    Generate random samples from the Student's t-distribution.

    This function uses ``scipy.stats.t.rvs`` to draw random values
    from a Student's t-distribution with the specified degrees of freedom.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    df : float, optional
        Degrees of freedom of the distribution. Must be greater than 0.
        Default is 2.
    random_state : object, optional
        Random state forwarded to ``scipy.stats.t.rvs``.
    loc : float, optional
        Location parameter. Default is 0.
    scale : float, optional
        Positive scale parameter. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Student's t-distribution.

    Raises
    ------
    ValueError
        If ``df <= 0``.

    Notes
    -----
    The Student's t-distribution is widely used in hypothesis testing
    when the sample size is small and the population variance is unknown.
    It has heavier tails than the normal distribution.

    Examples
    --------
    >>> generate_t(5, df=3)
    array([...])
    """
    return t.rvs(df=df, loc=loc, scale=scale, size=size, random_state=random_state)
