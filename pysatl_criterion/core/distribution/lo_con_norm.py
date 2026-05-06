import numpy as np
from scipy.stats import norm


def generate_lo_con_norm(size, p=0.5, a=0) -> list[float]:
    """
    Generate samples from a two-component mixture of normal distributions.

    Each observation is drawn independently from:
    - N(a, 1) with probability ``p``
    - N(0, 1) with probability ``1 - p``

    This produces a location-contaminated normal distribution.

    Parameters
    ----------
    size : int
        Number of random samples to generate.
    p : float, optional
        Probability of sampling from the shifted normal distribution N(a, 1).
        Must satisfy ``0 <= p <= 1``. Default is 0.5.
    a : float, optional
        Mean (location shift) of the contaminated component. Default is 0.

    Returns
    -------
    list of float
        Generated samples as a Python list.

    Raises
    ------
    ValueError
        If ``p`` is not in the interval [0, 1] or ``size`` is negative.

    Notes
    -----
    This is a simple mixture model often used to simulate contamination
    or outliers in normally distributed data.

    Examples
    --------
    >>> generate_lo_con_norm(5, p=0.3, a=2.0)
    [ ... ]
    """
    result = []
    for i in range(size):
        choice = np.random.choice(np.arange(2), p=[p, 1 - p])
        if choice == 0:
            item = norm.rvs(size=1, loc=a)
        else:
            item = norm.rvs(size=1)
        result.append(item)

    return np.concatenate(result, axis=0).tolist()
