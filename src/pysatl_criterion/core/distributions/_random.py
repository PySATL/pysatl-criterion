"""Internal helpers for reproducible distribution sampling."""

import numpy as np


def resolve_random_state(random_state):
    """Return a state object that can be shared by NumPy and SciPy draws."""
    if random_state is None or isinstance(
        random_state, (np.random.Generator, np.random.RandomState)
    ):
        return random_state

    return np.random.default_rng(random_state)
