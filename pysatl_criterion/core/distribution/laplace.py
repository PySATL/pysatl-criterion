from scipy.stats import laplace


def generate_laplace(size, t=0, s=1):
    """
    Generate random samples from the Laplace (double exponential) distribution.

    This function uses ``scipy.stats.laplace.rvs`` to draw random values
    from a Laplace distribution parameterized by location ``t`` and
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
        Array of random samples drawn from the Laplace distribution.

    Raises
    ------
    ValueError
        If ``s <= 0``.

    Notes
    -----
    The Laplace distribution is symmetric around its mean and has
    heavier tails than the normal distribution. It is often used in
    robust statistics and signal processing.

    Examples
    --------
    >>> generate_laplace(5, t=0.0, s=1.0)
    array([...])
    """
    return laplace.rvs(size=size, loc=t, scale=s)
