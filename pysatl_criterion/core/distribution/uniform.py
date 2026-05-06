from scipy.stats import uniform


def generate_uniform(size, a=0, b=1):
    """
    Generate random samples from the uniform distribution.

    This function uses ``scipy.stats.uniform.rvs`` to draw random values
    uniformly distributed over the interval [a, b].

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    a : float, optional
        Lower bound of the distribution interval. Default is 0.
    b : float, optional
        Upper bound of the distribution interval. Must be greater than ``a``.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the uniform distribution.

    Raises
    ------
    ValueError
        If ``b <= a``.

    Notes
    -----
    The uniform distribution assigns equal probability density to all values
    within the interval [a, b].

    Examples
    --------
    >>> generate_uniform(5, a=0, b=10)
    array([...])
    """
    scale = b - a
    return uniform.rvs(size=size, loc=a, scale=scale)
