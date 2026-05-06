from scipy.stats import gamma


def generate_gamma(size, alfa=0, beta=1):
    """
    Generate random samples from the Gamma distribution.

    This function uses ``scipy.stats.gamma.rvs`` to draw random values
    from a Gamma distribution parameterized by shape ``alfa`` and rate ``beta``.
    Internally, SciPy uses the ``scale`` parameter, where
    ``scale = 1 / beta``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    alfa : float, optional
        Shape parameter (α) of the Gamma distribution.
        Must be greater than 0. Default is 0.
    beta : float, optional
        Rate parameter (β) of the Gamma distribution.
        Must be greater than 0. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Gamma distribution.

    Raises
    ------
    ValueError
        If ``alfa <= 0`` or ``beta <= 0``.

    Notes
    -----
    The Gamma distribution is commonly used to model waiting times and
    is a generalization of the exponential and Chi-square distributions.

    Examples
    --------
    >>> generate_gamma(5, alfa=2.0, beta=3.0)
    array([...])
    """
    scale = 1 / beta
    return gamma.rvs(a=alfa, size=size, scale=scale)
