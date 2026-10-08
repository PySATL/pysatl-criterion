"""Independent integral and pairwise references for the conditional-moment test."""

import numpy as np
import pytest
from scipy import integrate, special, stats

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.statistics.alternative import RightAlternative
from pysatl_criterion.statistics.goodness_of_fit import EbnerLiebenbergBetaGofStatistic


@pytest.fixture
def statistic():
    return EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse({}))


@pytest.mark.parametrize("a,b", [(0.3, 0.4), (1, 1), (2, 5), (50, 70), (0.5, 8)])
def test_matches_integral_and_equation_three(statistic, a, b):
    x = stats.beta.ppf(np.linspace(0.03, 0.97, 25), a, b)
    a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    c = (a + b) * x - a
    n = len(x)

    def integrand(t):
        empirical = np.mean(c * (x >= t))
        theoretical = np.exp(a * np.log(t) + b * np.log1p(-t) - special.betaln(a, b))
        return n * (empirical - theoretical) ** 2

    integral = integrate.quad(integrand, 0, 1, points=x, epsabs=1e-10, limit=150)[0]
    pairwise = (
        np.sum(c[:, None] * c[None, :] * np.minimum.outer(x, x)) / n
        - 2
        * special.beta(a + 1, b + 1)
        / special.beta(a, b)
        * np.sum(c * stats.beta.cdf(x, a + 1, b + 1))
        + n * np.exp(special.betaln(2 * a + 1, 2 * b + 1) - 2 * special.betaln(a, b))
    )
    actual = statistic.execute_statistic(x)
    assert actual == pytest.approx(integral, rel=1e-8, abs=1e-10)
    assert actual == pytest.approx(pairwise, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize(
    "x",
    [
        [],
        [0.2],
        [0.2, 0.2],
        [0, 0.5],
        [0.5, 1],
        [-0.1, 0.5],
        [0.5, 1.1],
        [np.nan, 0.2],
        [np.inf, 0.2],
        [[0.2, 0.3]],
        [0.2 + 1j, 0.3],
        np.ma.array([0.2, 0.3], mask=[True, False]),
    ],
)
def test_invalid_samples(statistic, x):
    with pytest.raises(ValueError):
        statistic.execute_statistic(x)


def test_composite_contract_and_refitting(statistic):
    assert statistic.hypothesis().parameters() == {}
    assert isinstance(statistic.alternative(), RightAlternative)
    assert statistic.short_code() == "EL"
    for fixed in ({"a": 2}, {"b": 5}, {"a": 2, "b": 5}):
        with pytest.raises(ValueError, match="Unsupported"):
            EbnerLiebenbergBetaGofStatistic(Beta.DEFAULT.parse(fixed))
    x = np.array([0.1, 0.2, 0.2, 0.3, 0.5, 0.7])
    original = x.copy()
    value = statistic.execute_statistic(x)
    assert statistic.execute_statistic(x[::-1]) == pytest.approx(value)
    statistic.execute_statistic([0.05, 0.1, 0.15, 0.2])
    assert statistic.execute_statistic(x) == value
    np.testing.assert_array_equal(x, original)
    assert statistic.hypothesis().parameters() == {}
