import numpy as np
from scipy.stats import moment as scipy_moment


def moment(a, mom=1, center=None):
    """
    Compute the (raw or central) moment of a dataset.

    This function wraps ``scipy.stats.moment`` and allows computation of
    either raw moments (when ``center=None``) or centered moments.

    Parameters
    ----------
    a : array-like
        Input data.
    mom : int, optional
        Order of the moment to compute (e.g., 1 = mean deviation,
        2 = variance-related moment). Default is 1.
    center : float or array-like or None, optional
        Center of the moment calculation. If None, raw moment is computed.
        If provided, computes central moment around this value.

    Returns
    -------
    numpy.ndarray
        The computed moment(s).

    Notes
    -----
    When ``center=None``, SciPy computes raw moments.
    When ``center`` is specified, it computes central moments.

    Examples
    --------
    >>> moment([1, 2, 3], mom=2)
    array([...])
    """
    return scipy_moment(a=a, moment=mom, center=center)


def central_moment(a, mom=1):
    """
    Compute the central moment of a dataset.

    The central moment is computed relative to the sample mean:
    """
    mu = np.mean(a, axis=0)
    return scipy_moment(a=a, moment=mom, center=mu)
