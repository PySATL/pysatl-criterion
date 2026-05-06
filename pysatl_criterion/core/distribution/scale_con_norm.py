import numpy as np
from scipy.stats import norm


def generate_scale_con_norm(size, p=0.5, b=0) -> list[float]:
    """
    Generate samples from a scale-contaminated normal distribution.

    Each observation is independently drawn from:
    - N(0, b²) with probability ``p``
    - N(0, 1) with probability ``1 - p``

    This creates a mixture distribution where the variance is randomly
    inflated (or deflated) for a subset of samples.

    Parameters
    ----------
    size : int
        Number of random samples to generate.
    p : float, optional
        Probability of sampling from the scaled normal distribution N(0, b²).
        Must satisfy ``0 <= p <= 1``. Default is 0.5.
    b : float, optional
        Standard deviation of the contaminated component.
        Must be non-negative. Default is 0.

    Returns
    -------
    list of float
        Generated samples as a Python list.

    Raises
    ------
    ValueError
        If ``p`` is not in [0, 1] or ``b < 0``.

    Notes
    -----
    This model is commonly used to simulate heteroscedastic noise or
    variance contamination in otherwise standard normal data.

    Examples
    --------
    >>> generate_scale_con_norm(5, p=0.3, b=2.0)
    [ ... ]
    """
    result = []
    for i in range(size):
        choice = np.random.choice(np.arange(2), p=[p, 1 - p])
        if choice == 0:
            item = norm.rvs(size=1, scale=b)
        else:
            item = norm.rvs(size=1)
        result.append(item)

    return np.concatenate(result, axis=0).tolist()
