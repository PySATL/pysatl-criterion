from scipy.stats import tukeylambda


def generate_tukey(size, lam=2):
    """
    Generate random samples from the Tukey Lambda distribution.

    This function uses ``scipy.stats.tukeylambda.rvs`` to draw random values
    from a Tukey Lambda distribution, which is a flexible family of
    distributions controlled by the shape parameter ``lam``.

    Parameters
    ----------
    size : int or tuple of int
        Output shape of generated samples. If an integer is provided,
        returns a 1D array of length ``size``. If a tuple is provided,
        returns an array with the specified shape.
    lam : float, optional
        Shape parameter (λ) controlling tail behavior and kurtosis.
        Default is 2.

    Returns
    -------
    numpy.ndarray
        Array of random samples drawn from the Tukey Lambda distribution.

    Notes
    -----
    The Tukey Lambda distribution is a flexible family that can approximate
    normal, uniform, and heavy-tailed distributions depending on λ:
    - λ ≈ 0 → approximately normal
    - λ < 0 → heavy-tailed distributions
    - λ > 0 → bounded distributions

    Examples
    --------
    >>> generate_tukey(5, lam=0.14)
    array([...])
    """
    return tukeylambda.rvs(lam, size=size)
