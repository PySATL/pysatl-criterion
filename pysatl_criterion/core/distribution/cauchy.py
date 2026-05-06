from scipy.stats import cauchy


def generate_cauchy(size, t=0.5, s=0.5):
    """
    Generate random samples from the Cauchy distribution.

    This function uses ``scipy.stats.cauchy.rvs`` to draw random values
    from a Cauchy (Lorentz) distribution with specified location and scale.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    t : float, optional
        Location parameter (median and peak of the distribution).
        Default is 0.5.
    s : float, optional
        Scale parameter (controls the width of the distribution).
        Must be positive. Default is 0.5.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Cauchy distribution.

    Raises
    ------
    ValueError
        If ``s <= 0``.

    Notes
    -----
    The Cauchy distribution is a heavy-tailed distribution with undefined
    mean and variance. It is often used in cases where extreme values
    (outliers) are more probable than in a normal distribution.

    Examples
    --------
    >>> generate_cauchy(5, t=0.0, s=1.0)
    array([...])
    """
    return cauchy.rvs(size=size, loc=t, scale=s)
