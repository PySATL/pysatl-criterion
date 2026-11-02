"""Regressions for calibration identity and cancellation in Beta statistics."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy import special

from pysatl_criterion.distribution import BetaDistributionDescriptor as Beta
from pysatl_criterion.hypothesis_testing.limit_distribution.base import (
    StorageLimitDistributionResolver,
)
from pysatl_criterion.persistence.models.limit_distribution import LimitDistributionModel
from pysatl_criterion.statistics.goodness_of_fit.continuous.beta import (
    AndersonDarlingBetaGofStatistic,
    Chi2PearsonBetaGofStatistic,
    EbnerLiebenbergBetaGofStatistic,
    KolmogorovSmirnovBetaGofStatistic,
    NeymanSmoothBetaGofStatistic,
    RaschkeBetaGofStatistic,
)


@pytest.mark.parametrize(
    "cls,fixed,options",
    [
        (EbnerLiebenbergBetaGofStatistic, {}, {}),
        (RaschkeBetaGofStatistic, {}, {}),
        (NeymanSmoothBetaGofStatistic, {"a": 2, "b": 5}, {"k": 1}),
        (NeymanSmoothBetaGofStatistic, {"a": 2, "b": 5}, {"k": 9}),
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
    ],
)
def test_default_specified_calibration_still_loads(cls):
    fixed = {"a": 2, "b": 5}
    statistic = cls(Beta.DEFAULT.parse(fixed))
    model = LimitDistributionModel.from_statistic(statistic, 10, [0.1, 0.2, 0.3])
    storage = Mock()
    storage.get.return_value = model
    assert StorageLimitDistributionResolver(storage).resolve(statistic, 10) == [0.1, 0.2, 0.3]
    storage.get.assert_called_once_with(model.key)


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
