from scipy.stats import beta


def generate_beta(size, a=0.0, b=1.0):
    """
    Generate random samples from the Beta distribution.

    This function draw random values
    from a Beta distribution parameterized by shape parameters ``a`` and ``b``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    a : float, optional
        First shape parameter (alpha) of the Beta distribution.
        Must be greater than 0. Default is 0.0.
    b : float, optional
        Second shape parameter (beta) of the Beta distribution.
        Must be greater than 0. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Beta distribution.

    Raises
    ------
    ValueError
        If ``a <= 0`` or ``b <= 0``.

    Notes
    -----
    The Beta distribution is defined on the interval [0, 1] and is often used
    to model probabilities and proportions.

    Examples
    --------
    >>> generate_beta(5, a=2.0, b=5.0)
    array([...])
    """
    return beta.rvs(a=a, b=b, size=size)
