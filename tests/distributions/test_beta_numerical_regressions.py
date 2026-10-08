"""Regressions for calibration identity and cancellation in Beta statistics."""

from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import special, stats

from pysatl_criterion.distribution.distributions import BetaDistributionDescriptor as Beta
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    StorageLimitDistributionResolver,
)
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.statistics.alternative import AlternativeType
from pysatl_criterion.statistics.goodness_of_fit.beta import (
    AndersonDarlingBetaGofStatistic,
    Chi2PearsonBetaGofStatistic,
    EbnerLiebenbergBetaGofStatistic,
    KolmogorovSmirnovBetaGofStatistic,
    MomentBasedBetaGofStatistic,
    NeymanSmoothBetaGofStatistic,
    RaschkeBetaGofStatistic,
    SkewnessKurtosisBetaGofStatistic,
)


@pytest.mark.parametrize(
    "cls,fixed,options",
    [
        (EbnerLiebenbergBetaGofStatistic, {}, {}),
        (RaschkeBetaGofStatistic, {}, {}),
        (NeymanSmoothBetaGofStatistic, {"a": 2, "b": 5}, {"k": 1}),
        (NeymanSmoothBetaGofStatistic, {"a": 2, "b": 5}, {"k": 9}),
        (
            KolmogorovSmirnovBetaGofStatistic,
            {"a": 2, "b": 5},
            {"alternative_type": AlternativeType.RIGHT},
        ),
        (
            KolmogorovSmirnovBetaGofStatistic,
            {"a": 2, "b": 5},
            {"alternative_type": AlternativeType.LEFT},
        ),
        (Chi2PearsonBetaGofStatistic, {"a": 2, "b": 5}, {"lambda_": 0}),
    ],
)
def test_unsafe_calibration_rejected_before_storage_lookup(cls, fixed, options):
    storage = Mock()
    statistic = cls(Beta.DEFAULT.parse(fixed), **options)
    with pytest.raises(ValueError, match="calibration"):
        StorageLimitDistributionResolver(storage).resolve(statistic, 100)
    storage.get.assert_not_called()


@pytest.mark.parametrize(
    "cls",
    [
        NeymanSmoothBetaGofStatistic,
        KolmogorovSmirnovBetaGofStatistic,
        Chi2PearsonBetaGofStatistic,
        MomentBasedBetaGofStatistic,
    ],
)
def test_default_specified_calibration_still_loads(cls):
    fixed = {"a": 2, "b": 5}
    statistic = cls(Beta.DEFAULT.parse(fixed))
    model = LimitDistributionModel(statistic.code(), fixed, 10, 3, [0.1, 0.2, 0.3])
    storage = Mock()
    storage.get.return_value = model
    assert StorageLimitDistributionResolver(storage).resolve(statistic, 10) == [0.1, 0.2, 0.3]
    storage.get.assert_called_once_with(model.key)


@pytest.mark.parametrize("shape", [1e-18, 1e-17, 1e-16, 1e-10, 0.1, 2, 1e8])
def test_moment_based_symmetric_covariance_reference(shape):
    x = np.array([0.1, 0.3, 0.6, 0.8])
    variance = 1 / (4 * (2 * shape + 1))
    # For symmetric Beta the cross-covariance vanishes; Var(Z^2) is exact below.
    variance_of_squared_z = 4 * shape / (2 * shape + 3)
    expected = len(x) * (
        (x.mean() - 0.5) ** 2 / variance
        + (x.var(ddof=1) / variance - 1) ** 2 / variance_of_squared_z
    )
    value = MomentBasedBetaGofStatistic(Beta.DEFAULT.parse({"a": shape, "b": shape}))
    assert value.execute_statistic(x) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("shape", [1e-18, 1e-16, 1e-10, 0.1, 2, 1e8])
def test_skewness_kurtosis_exact_rational_symmetric_reference(shape):
    x = np.array([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
    s = 2 * Fraction(shape)
    # Closed-form symmetric moments computed as exact rationals, independently
    # of the production recurrence, polynomial multiplication and Decimal solve.
    m4 = 3 * (s + 1) / (s + 3)
    m6 = 15 * (s + 1) ** 2 / ((s + 3) * (s + 5))
    m8 = 105 * (s + 1) ** 3 / ((s + 3) * (s + 5) * (s + 7))
    v3 = float(m6 - 6 * m4 + 9)
    v4 = float(m8 - 4 * m4 * m6 + 4 * m4**3 - m4**2)
    expected = len(x) * (
        stats.skew(x, bias=False) ** 2 / v3
        + (stats.kurtosis(x, bias=False) - float(m4 - 3)) ** 2 / v4
    )
    statistic = SkewnessKurtosisBetaGofStatistic(Beta.DEFAULT.parse({"a": shape, "b": shape}))
    actual = statistic.execute_statistic(x)
    assert np.isfinite(actual) and actual >= 0
    assert actual == pytest.approx(expected, rel=1e-12)


def _integer_beta_logcdf(x, a, b):
    # I_x(a,b) = P(Binomial(a+b-1,x) >= a), computed as a finite log-sum.
    n = a + b - 1
    j = np.arange(a, n + 1)
    terms = (
        special.gammaln(n + 1)
        - special.gammaln(j + 1)
        - special.gammaln(n - j + 1)
        + j * np.log(x)
        + (n - j) * np.log1p(-x)
    )
    return special.logsumexp(terms)


@pytest.mark.parametrize(
    "a,b,x",
    [
        (50, 70, [1e-12, 0.2, 0.4, 0.6]),
        (70, 50, [0.4, 0.6, 0.8, 1 - 1e-12]),
        (500, 700, [0.01, 0.3, 0.4, 0.99]),
        (50, 1, [np.nextafter(0.0, 1.0), 0.1, 0.3]),
    ],
)
def test_ad_underflow_matches_binomial_identity(a, b, x):
    x = np.sort(x)
    n = len(x)
    left = np.array([_integer_beta_logcdf(t, a, b) for t in x])
    # Upper tail binomial sum directly, avoiding 1-x rounding for tiny x.
    j = np.arange(b, a + b)
    log_choose = special.gammaln(a + b) - special.gammaln(j + 1) - special.gammaln(a + b - j)
    right = np.array(
        [special.logsumexp(log_choose + j * np.log1p(-t) + (a + b - 1 - j) * np.log(t)) for t in x]
    )
    expected = -n - np.dot((2 * np.arange(1, n + 1) - 1) / n, left + right[::-1])
    statistic = AndersonDarlingBetaGofStatistic(Beta.DEFAULT.parse({"a": a, "b": b}))
    assert statistic.execute_statistic(x) == pytest.approx(expected, rel=1e-11)


@pytest.mark.parametrize("endpoint", [0.0, 1.0])
def test_ad_true_endpoints_remain_infinite(endpoint):
    statistic = AndersonDarlingBetaGofStatistic(Beta.DEFAULT.parse({"a": 50, "b": 70}))
    assert np.isposinf(statistic.execute_statistic([endpoint, 0.4]))
