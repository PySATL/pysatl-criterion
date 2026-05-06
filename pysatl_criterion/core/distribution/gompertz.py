from scipy.stats import gompertz


def generate_gompertz(size, eta=0, b=1):
    """
    Generate random samples from the Gompertz distribution.

    This function uses ``scipy.stats.gompertz.rvs`` to draw random values
    from a Gompertz distribution parameterized by shape ``eta`` and scale ``b``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    eta : float, optional
        Shape parameter of the Gompertz distribution.
        Must be greater than 0. Default is 0.
    b : float, optional
        Scale parameter of the distribution.
        Must be greater than 0. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Gompertz distribution.

    Raises
    ------
    ValueError
        If ``eta <= 0`` or ``b <= 0``.

    Notes
    -----
    The Gompertz distribution is often used in survival analysis and
    actuarial science to model lifetimes and failure rates that increase
    exponentially over time.

    Examples
    --------
    >>> generate_gompertz(5, eta=1.5, b=2.0)
    array([...])
    """
    return gompertz.rvs(eta, size=size, scale=b)
