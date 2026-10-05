import numpy as np
from scipy.stats import norm

from pysatl_criterion.core.distributions._random import resolve_random_state


def generate_mix_con_norm(size, p=0.5, a=0, b=1, random_state=None) -> list[float]:
    """
    Generate samples from a two-component mixture of normal distributions.

    Each observation is independently drawn from:
    - N(a, b^2) with probability ``p``
    - N(0, 1) with probability ``1 - p``

    This defines a contaminated normal (mixture) distribution.

    Parameters
    ----------
    size : int
        Number of random samples to generate.
    p : float, optional
        Probability of sampling from the contaminated normal distribution
        N(a, b^2). Must satisfy ``0 <= p <= 1``. Default is 0.5.
    a : float, optional
        Mean of the contaminated normal component. Default is 0.
    b : float, optional
        Standard deviation of the contaminated normal component.
        Must be positive. Default is 1.

    Returns
    -------
    list of float
        Generated samples as a Python list.

    Raises
    ------
    ValueError
        If ``p`` is not in [0, 1] or ``b <= 0``.

    Notes
    -----
    This mixture model is commonly used to simulate data with outliers
    or heterogeneous variance structures.

    Examples
    --------
    >>> generate_mix_con_norm(5, p=0.3, a=2, b=0.5)
    [ ... ]
    """
    resolved_random_state = resolve_random_state(random_state)
    choice_random_state = np.random if resolved_random_state is None else resolved_random_state
    result = []
    for i in range(size):
        choice = choice_random_state.choice(np.arange(2), p=[p, 1 - p])
        if choice == 0:
            item = norm.rvs(size=1, loc=a, scale=b, random_state=resolved_random_state)
        else:
            item = norm.rvs(size=1, random_state=resolved_random_state)
        result.append(item)

    return np.concatenate(result, axis=0).tolist()
