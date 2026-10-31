"""Small exact likelihoods detect errors hidden by nonnegativity-only checks."""

import numpy as np
import pytest

from pysatl_criterion.estimation.maximum_likelihood import (
    function_for_maximum_likelihood as likelihoods,
)


@pytest.mark.parametrize(
    "name,sample,expected",
    [
        ("uniform_continuous", [1, 2, 4], 1 / 27),
        ("normal", [-1, 1], np.exp(-1) / (2 * np.pi)),
        ("lognormal", [np.exp(-1), np.e], np.exp(-1) / (2 * np.pi)),
        ("exponential", [0, 1, 2], np.exp(-3)),
        ("pareto", [1, np.e, np.e**2], np.exp(-6)),
        ("laplace", [-1, 0, 2], np.exp(-3) / 8),
        ("bernoulli", [0, 1, 1], 4 / 27),
        ("poisson", [0, 1, 2], np.exp(-3) / 2),
        ("geometric", [1, 2, 3], 1 / 64),
    ],
)
def test_likelihood_matches_analytical_value(name, sample, expected):
    function = getattr(likelihoods, f"likelihood_function_{name}")
    assert function(np.array(sample)) == pytest.approx(expected, rel=1e-12, abs=0)


def test_binomial_likelihood_includes_binomial_coefficients():
    assert likelihoods.likelihood_function_binomial(np.array([0, 2, 4]), 4) == pytest.approx(
        6 / 4096, rel=1e-12, abs=0
    )
