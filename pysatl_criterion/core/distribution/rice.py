from scipy.stats import rice


def generate_rice(size, nu=0, sigma=1):
    """
    Generate random samples from the Rice distribution.

    This function uses ``scipy.stats.rice.rvs`` to draw random values
    from a Rice (Rician) distribution, commonly used to model signal
    strength with a dominant component plus noise.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    nu : float, optional
        Non-centrality parameter (distance from origin of the deterministic
        component). Default is 0.
    sigma : float, optional
        Scale parameter controlling the spread (noise level).
        Must be positive. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Rice distribution.

    Raises
    ------
    ValueError
        If ``sigma <= 0``.

    Notes
    -----
    The Rice distribution is often used in communications engineering,
    especially in wireless fading channels where there is a line-of-sight
    signal plus random multipath components.

    Examples
    --------
    >>> generate_rice(5, nu=2.0, sigma=1.0)
    array([...])
    """
    return rice.rvs(nu, size=size, scale=sigma)
