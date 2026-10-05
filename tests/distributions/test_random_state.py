"""Reproducibility tests for random-value generators."""

from collections.abc import Callable

import numpy as np
import pytest

from pysatl_criterion.core.distributions.beta import generate_beta
from pysatl_criterion.core.distributions.expon import generate_expon
from pysatl_criterion.core.distributions.gamma import generate_gamma
from pysatl_criterion.core.distributions.lognormal import generate_lognorm
from pysatl_criterion.core.distributions.norm import generate_norm
from pysatl_criterion.core.distributions.student import generate_t
from pysatl_criterion.core.distributions.uniform import generate_uniform
from pysatl_criterion.core.distributions.weibull import generate_weibull


@pytest.mark.parametrize(
    ("generator", "parameters"),
    [
        (generate_norm, {"mean": 1.0, "var": 2.0}),
        (generate_expon, {"lam": 2.0}),
        (generate_weibull, {"a": 1.0, "k": 2.0}),
        (generate_gamma, {"alfa": 2.0, "beta": 1.0}),
        (generate_beta, {"a": 2.0, "b": 3.0}),
        (generate_lognorm, {"s": 0.5, "mu": 1.0}),
        (generate_t, {"df": 5.0}),
        (generate_uniform, {"a": -1.0, "b": 2.0}),
    ],
)
def test_generator_accepts_random_state_and_is_reproducible(
    generator: Callable[..., np.ndarray], parameters: dict[str, float]
) -> None:
    first = generator(size=20, random_state=np.random.default_rng(123), **parameters)
    second = generator(size=20, random_state=np.random.default_rng(123), **parameters)

    assert np.array_equal(first, second)
