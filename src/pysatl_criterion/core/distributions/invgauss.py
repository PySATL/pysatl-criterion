from scipy.stats import invgauss


def generate_invgauss(size, mu=0, lam=1):
    """
    Generate random samples from the inverse Gaussian (Wald) distribution.

    This function uses ``scipy.stats.invgauss.rvs`` to draw random values
    from an inverse Gaussian distribution. In SciPy, the distribution is
    parameterized by a shape parameter ``mu`` and a ``scale`` parameter.
    Here, ``lam`` is passed as the scale.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    mu : float, optional
        Shape parameter of the distribution. Must be greater than 0.
        Default is 0.
    lam : float, optional
        Scale parameter of the distribution. Must be greater than 0.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the inverse Gaussian distribution.

    Raises
    ------
    ValueError
        If ``mu <= 0`` or ``lam <= 0``.

    Notes
    -----
    The inverse Gaussian distribution is used to model positive-valued,
    right-skewed data and appears in contexts such as first-passage times
    in stochastic processes.

    Examples
    --------
    >>> generate_invgauss(5, mu=1.0, lam=2.0)
    array([...])
    """
    return invgauss.rvs(mu, size=size, scale=lam)
